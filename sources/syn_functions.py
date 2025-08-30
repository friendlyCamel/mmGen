import os
import cupy as cp
import numexpr as ne
import open3d as o3d
from time import time
from tqdm import tqdm
from syn_define import *
from radar_define import *
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d


# 可视化场景
def vis_scene(radar_location, targets):
    import copy
    import os
    FOR = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=radar_location)
    visual_mesh = []
    visual_mesh.append(FOR)
    for pic in range(syn_frame_num):
        for target in targets:
            if (target['type']=='person'):
                obj_files = [file for file in sorted(os.listdir(target['model_dir'])) if file.endswith('.obj')]
                loc = np.load(target['loc_path'])
                loc[:, 1] -= 0.15
                mesh_in = o3d.io.read_triangle_mesh(target['model_dir']+"/"+obj_files[int(pic)])
                R = mesh_in.get_rotation_matrix_from_xyz(target['rotation'])
                mesh_in = mesh_in.rotate(R, center=(0, 0, 0))
                mesh_in = mesh_in.translate(loc[int(pic)])
            else:
                mesh_in = o3d.io.read_triangle_mesh(target['model'])
                mesh_in = mesh_in.translate(target['location'])
                R = mesh_in.get_rotation_matrix_from_xyz(target['rotation'])
                mesh_in = mesh_in.rotate(R, center=target['rotation'])
                vertices = np.array(mesh_in.vertices)
                mesh_in.vertices = o3d.utility.Vector3dVector(vertices)
            mesh_normal = copy.deepcopy(mesh_in)
            mesh_normal.compute_vertex_normals()
            visual_mesh.append(mesh_normal)
        o3d.visualization.draw_geometries(visual_mesh, point_show_normal=False, mesh_show_back_face=True)
        visual_mesh.pop()
        visual_mesh.pop()

def get_loc(person, HPR_R, frame_start):
    """
        读取一个sample下的所有45个mesh，插值获得每个chirp时间点时的位置，为减少计算
    """
    mesh_dir = person['model_dir']
    loc = np.load(person['loc_path'])
    loc[:, 1] -= 0.15                           # 深度相机和毫米波雷达的深度视察
    init_loc = np.zeros([2, triangles_num, 3])
    init_area = np.zeros([2, triangles_num])
    init_norm = np.zeros([2, triangles_num, 3])
    obj_files = [file for file in sorted(os.listdir(mesh_dir)) if file.endswith('.obj')]
    origin_ = np.zeros((1, 3), dtype=float)  # 代表原点
    print("------------------------------------------------------------------------------------------------")
    for index, path in enumerate(obj_files):
        # 这里选择近邻的2帧
        if index < frame_start:
            continue
        if index >= frame_start + 2:
            break
        mesh = o3d.io.read_triangle_mesh(mesh_dir+'/'+path)
        R = mesh.get_rotation_matrix_from_xyz(person['rotation'])
        mesh = mesh.rotate(R, center=(0, 0, 0))
        mesh = mesh.translate(loc[int(index)])
        triangles = np.array(mesh.triangles)
        vertices = np.array(mesh.vertices)
        v1 = vertices[triangles[:, 0]]
        v2 = vertices[triangles[:, 1]]
        v3 = vertices[triangles[:, 2]]
        t1 = time()
        init_area[index - frame_start] = cal_area_batch(v1, v2, v3)
        init_norm[index - frame_start] = cal_norm_batch(v1, v2, v3)
        init_loc[index - frame_start, :, :] = (v1 + v2 + v3)/3
        t2 = time()
        print(f"Calculate the norm, area and loc for one frame: {t2 - t1:.6f}秒")
    tic = time()
    mesh_time_points = np.linspace(0, mesh_period, 2)
    chirp_time_points = np.linspace(0,mesh_period, mesh_frame_num * chirp_num)
    f = interp1d(mesh_time_points, init_loc, kind='linear', axis=0)
    res_loc = f(chirp_time_points).reshape([mesh_frame_num, chirp_num, triangles_num, 3])
    toc = time()
    print(f"Interp1d cost:{toc - tic:.6f}秒")
    res_loc = res_loc[0, :, :, :]
    # 这里假设一帧的时间内遮挡情况不变，反射的rcs值也不变，仅仅是距离变化导致相位的变化。
    tic = time()
    triangle_pos = res_loc[0]
    triangle_dis = np.sqrt(np.sum(triangle_pos**2, axis=1, keepdims=True))
    max_dis = np.max(triangle_dis)
    triangle_trans = triangle_pos + 2 * (triangle_pos / triangle_dis) * (max_dis * np.power(10, HPR_R) - triangle_dis)
    triangle_trans = np.concatenate((triangle_trans, origin_), axis=0)
    point_index = ConvexHull(triangle_trans).vertices[:-1]  # 去掉原点
    # 降采样并记录下这一帧选择的反射点
    idx = np.linspace(0, len(point_index) - 1, downsample_triangles_num, dtype=int)
    triangle_point_index = np.array(point_index)[idx]
    # 反射点位置，area以及norm
    visable_points = res_loc[:, triangle_point_index, :]
    init_area_perframe = init_area[0]
    init_norm_perframe = init_norm[0]
    visable_points_area = np.repeat(init_area_perframe[np.newaxis, triangle_point_index], chirp_num, axis=0)
    visable_points_norm = np.repeat(init_norm_perframe[np.newaxis, triangle_point_index, :], chirp_num, axis=0)
    toc = time()
    print(f"HPR cost:{toc - tic:.6f}秒")
    return visable_points, visable_points_area, visable_points_norm
def cal_area(p1, p2, p3):
    a = p1 - p2
    b = p2 - p3
    c = p3 - p1
    aLen = np.sqrt(np.dot(a, a))
    bLen = np.sqrt(np.dot(b, b))
    cLen = np.sqrt(np.dot(c, c))
    s = (aLen + bLen + cLen)/2
    return np.sqrt(s*(s - aLen)*(s - bLen)*(s - cLen))
def cal_area_batch(p1, p2, p3):
    # 计算每条边的向量
    a = p1 - p2
    b = p2 - p3
    c = p3 - p1
    # 计算每条边的长度
    aLen = np.linalg.norm(a, axis=1)
    bLen = np.linalg.norm(b, axis=1)
    cLen = np.linalg.norm(c, axis=1)
    # 计算半周长
    s = (aLen + bLen + cLen) / 2
    # 计算面积
    areas = np.sqrt(s * (s - aLen) * (s - bLen) * (s - cLen))
    return areas
def cal_norm(p1, p2, p3):
    a = p1 - p2
    b = p2 - p3
    vect = np.cross(a, b)
    stand_vec = (p1 + p2 + p3) / 3
    # 取一个朝向原点的法向量
    if np.dot(vect, stand_vec) > 0:
        vect = np.cross(-a, b)
    norm = vect / np.sqrt(np.dot(vect, vect))
    return norm
def cal_norm_batch(p1, p2, p3):
    a = p1 - p2
    b = p2 - p3
    # 计算法向量
    vect = np.cross(a, b)
    # 计算标准化法向量
    norms = np.linalg.norm(vect, axis=1, keepdims=True)
    normalized_vect = vect / norms
    # 计算三角形重心
    stand_vec = (p1 + p2 + p3) / 3
    # 取一个朝向原点的法向量
    dot_product = np.einsum('ij,ij->i', vect, stand_vec)
    mask = dot_product > 0
    normalized_vect[mask] = -normalized_vect[mask]
    return normalized_vect
def jihua_v(theta, epsi_0, sigma):
    epsi = epsi_0 - 1j*60*0.005*sigma
    theta = theta.flatten()
    return (epsi*np.sin(np.radians(theta)) - np.sqrt(epsi - np.cos(np.radians(theta))**2)) / (epsi*np.sin(np.radians(theta)) + np.sqrt(epsi - np.cos(np.radians(theta))**2))
def jihua_h(theta, epsi_0, sigma):
    epsi = epsi_0 - 1j*60*0.005*sigma
    theta = theta.flatten()
    return (np.sin(np.radians(theta)) - np.sqrt(epsi - np.cos(np.radians(theta))**2)) / (np.sin(np.radians(theta)) + np.sqrt(epsi - np.cos(np.radians(theta))**2))

def person_once(radar, persons, frame_start):
    # persons_loc [persons_num, frames, chirps, points, 3]
    theta_0 = 2.0277
    fai_0 = 0.6903
    sigma = 1.11
    HPR_R = 3.5
    pi = np.pi
    for person in persons:
        # (chirp, point, 3) (chirp, point), (chirp, point, 3)
        persons_loc, person_area, person_norm = get_loc(person, HPR_R, frame_start)
    newshape = (1, persons_loc.shape[1])
    radius_per_vertex = np.full(newshape, 0.02)
    sinal_res = np.zeros([cal_frame_num, tx_num, rx_num, chirp_num, sample_num]).astype(np.complex128)
    vir_ant = np.zeros((3, 4, 2, 3), dtype=float)
    for tx_index in range(radar.radar_prop['transmitter'].txchannel_prop['size']):
        for rx_index in range(radar.radar_prop['receiver'].rxchannel_prop['size']):
            vir_ant[tx_index, rx_index, 0] = radar.radar_prop['transmitter'].txchannel_prop['locations'][tx_index] + radar.radar_prop['location']
            vir_ant[tx_index, rx_index, 1] = radar.radar_prop['receiver'].rxchannel_prop['locations'][rx_index] + radar.radar_prop['location']
    vir_ant = vir_ant.reshape(12, 2, -1)
    vir_ant_tx = vir_ant[:, 0, :].reshape(12, 1, 1, -1)
    vir_ant_rx = vir_ant[:, 1, :].reshape(12, 1, 1, -1)
    tt2 = time()
    # (antenna, chirp, points, xyz)
    persons_loc = persons_loc.reshape((1, chirp_num, -1, 3))
    person_norm = person_norm.reshape((1, chirp_num, -1, 3))
    person_area = person_area.reshape((1, chirp_num, -1, 1))
    distance = np.sqrt(np.sum((persons_loc-vir_ant_tx)**2, axis=3)) + np.sqrt(np.sum((persons_loc-vir_ant_rx)**2, axis=3))
    delay = ne.evaluate("distance / light_v")
    tem1 = vir_ant_tx[:, :, :, 2]
    tem2 = persons_loc[:, :, :, 2]
    theta = ne.evaluate("arcsin((tem1 - tem2) / distance * 2)")
    tem1 = vir_ant_tx[:, :, :, 0]
    tem2 = persons_loc[:, :, :, 0]
    fai = ne.evaluate("arcsin(tem1 - tem2)")
    a_gain = ne.evaluate("10**((96.148*exp(-(theta**2 / 2 / theta_0**2)) - 100)/10)")
    e_gain = ne.evaluate("10**((95.363*exp(-(fai**2 / 2 / fai_0**2)) - 100)/10)")
    G = ne.evaluate("a_gain*e_gain").reshape(12, chirp_num, -1, 1)*10**5.8
    tem1 = ne.evaluate("sum((persons_loc - vir_ant_tx)**2, axis=3)")[:, :, :, np.newaxis]
    r_in_stand = ne.evaluate("(persons_loc - vir_ant_tx) / sqrt(tem1)")
    tem2 = ne.evaluate("sum((persons_loc - vir_ant_tx)*person_norm, axis=3)")[:, :, :, np.newaxis]
    rb = ne.evaluate("r_in_stand - 2 * (tem2/sqrt(tem1))*person_norm")
    tem1 = ne.evaluate("sum((vir_ant_rx - persons_loc) * rb, axis=3)")[:, :, :, np.newaxis]
    tem2 = ne.evaluate("sum((vir_ant_rx - persons_loc)**2, axis=3)")[:, :, :, np.newaxis]
    cos_theta = ne.evaluate("tem1 / sqrt(tem2)")
    # 使用numpy的clip函数来限制数值范围
    cos_theta = np.clip(cos_theta, -1, 1)
    theta_to_strongest = ne.evaluate("arccos(cos_theta)")
    N = ne.evaluate("exp(-theta_to_strongest**2 / 2 / sigma**2)")
    A = person_area
    rcs_sqrt = ne.evaluate("G*N*A").squeeze()
    Gtx = 10**(radar.radar_prop['transmitter'].txchannel_prop['antenna_gains'][0] / 20)
    Grx = 10**(radar.radar_prop['receiver'].rxchannel_prop['antenna_gains'][0] / 20)
    tem1 = np.sqrt(radar.radar_prop['transmitter'].rf_prop['tx_power'])
    A_1 = ne.evaluate("Gtx * Grx * wavelength * tem1 * rcs_sqrt / ((4 * pi)**1.5 * distance**2)")
    tm_shape = list(A_1.shape)
    tm_shape.append(sample_num)
    t_m = np.linspace(0, chirp_ramp_time, sample_num).reshape([1, 1, 1, sample_num])
    t_m = np.broadcast_to(t_m, tm_shape)
    delay = delay.reshape(12, chirp_num, -1, 1)
    A_1 = A_1.reshape(12, chirp_num, -1, 1)
    if_sig_c = ne.evaluate("A_1 * exp(-1j*2*pi* (f_0*delay - frequency_slope*delay**2/2 + frequency_slope*t_m*delay))")
    tt3 = time()
    print(f"Calculate human reflection IF signal cost:{tt3 - tt2:.6f}秒")
    sinal_res[0, :, :, :, :] = ne.evaluate("sum(if_sig_c, axis=2)").reshape((3, 4, chirp_num, -1))
    # persons_loc [persons_num, frames, chirps, points, 3]
    # radius_per_vertex (1, points) 0.02
    # person_area (chirp, points)
    # person_norm (chirp, points, 3)
    return sinal_res, persons_loc, radius_per_vertex, person_area, person_norm

def env_sim(radar, env):
    # 拟合TI雷达用户手册中的雷达角度增益曲线，估计出俯仰角、方位角曲线增益的函数参数
    theta_0 = 2.0277
    fai_0 = 0.6903
    sigma = 1.11
    # 在利用votenet检测不同物体的类别和bounding box后，查阅不同物品的对应的反射参数
    materials = [
        {
            "name": "table0",
            "x_range": (0.46, 1.2),
            "y_range": (0.902, 2.54),
            "z_range": (-0.36, -0.1),
            "theta": 0,
            "score": 0.032,
            "epsi": 5,
            "sigma_jihua": 1e-10,
            "sigma": 1.2,
            "an": 1.8
        },
        {
            "name": "board1",
            "x_range": (-1.1577, -0.7759),
            "y_range": (1.9591, 3.1356),
            "z_range": (-0.5276, 0.8232),
            "theta": 0,
            "score": 0.03,
            "epsi": 3,
            "sigma_jihua": 5e-17,
            "sigma": 0.7,
            "an": 25
        },
        {
            "name": "ground2",
            "x_range": (-1.8, 2.2),
            "y_range": (0, 3),
            "z_range": (-1.05, -0.75),
            "theta": 0,
            "score": 1,
            "epsi": 4,
            "sigma_jihua": 1e-15,
            "sigma": 1.95,
            "an": 0.02
        },
        # 其他物品或材料示意
        # {
        #     "name": "wall3",
        #     "x_range": (1.463, 1.594),
        #     "y_range": (1.413, 2.372),
        #     "z_range": (-2.0988, 0.12),
        #     "theta": 0,
        #     "score": 0.0045,
        #     "epsi": 5.31,
        #     "sigma_jihua": 3.26e-2,
        #     "sigma": 5.28,
        #     "an": 33
        # },
        # {
        #     "name": "sofa4",
        #     "x_range": (0.3032, 1.2122),
        #     "y_range": (1.24613, 2.17027),
        #     "z_range": (-1.321583, -0.617317),
        #     "theta": 0,
        #     "score": 0.45,
        #     "epsi": 4.8,
        #     "sigma_jihua": 1e-11,
        #     "sigma": 1.95,
        #     "an": 0.1
        # },
        # {
        #     "name": "flowerpot5",
        #     "x_range": (-1.72115, -1.34485),
        #     "y_range": (1.857, 2.185),
        #     "z_range": (-1.2362, -0.7592),
        #     "theta": 0,
        #     "score": 0.3,
        #     "epsi": 6.5,
        #     "sigma_jihua": 2e-7,
        #     "sigma": 1.95,
        #     "an": 10
        # }
    ]
    def is_point_in_object(point, obj):
        x, y, z = point
        return (obj['x_range'][0] <= x <= obj['x_range'][1] and
            obj['y_range'][0] <= y <= obj['y_range'][1] and
            obj['z_range'][0] <= z <= obj['z_range'][1])

    def env_once(radar, triangle_pos, triangle_area, triangle_norm, triangle_mat):
        triangle_pos = triangle_pos.reshape(-1, 3)
        triangle_area = triangle_area.reshape(-1, 1)
        triangle_norm = triangle_norm.reshape(-1, 3)
        triangle_mat = triangle_mat.reshape(-1, 4)
        signal_res = np.zeros([cal_frame_num, tx_num, rx_num, sample_num]).astype(np.complex128)
        t1 = time()
        for tx_index in range(radar.radar_prop['transmitter'].txchannel_prop['size']):
            for rx_index in range(radar.radar_prop['receiver'].rxchannel_prop['size']):
                tx_loc = radar.radar_prop['transmitter'].txchannel_prop['locations'][tx_index] + radar.radar_prop['location']
                rx_loc = radar.radar_prop['receiver'].rxchannel_prop['locations'][rx_index] + radar.radar_prop['location']
                distance = np.sqrt(np.sum(abs(triangle_pos - tx_loc)**2,axis=1)) + np.sqrt(np.sum(abs(triangle_pos - rx_loc)**2, axis=1))
                delay = distance / light_v
                # 计算RCS等信息
                theta = np.arcsin((tx_loc[2] - triangle_pos[:, 2]) / distance * 2)
                fai = np.arcsin((tx_loc[0] - triangle_pos[:, 0]) / distance * 2)
                a_gain = 10**((96.148*np.exp(-(theta**2 / 2 / theta_0**2)) - 100)/10)
                e_gain = 10**((95.363*np.exp(-(fai**2 / 2 / fai_0**2)) - 100)/10)
                G = (a_gain*e_gain).reshape(-1, 1)*10**6.3
                r_in_stand = (triangle_pos - tx_loc) / np.sqrt(np.sum((triangle_pos - tx_loc)**2, axis=1)).reshape(-1, 1)
                rb = r_in_stand - 2 * (np.sum((triangle_pos - tx_loc)*triangle_norm, axis=1)/np.sqrt(np.sum((triangle_pos - tx_loc)**2, axis=1))).reshape(-1, 1)*triangle_norm
                theta_to_strongest = np.arccos(np.sum((rx_loc - triangle_pos) * rb, axis=1) / np.sqrt(np.sum((rx_loc - triangle_pos)**2, axis=1)))
                N = (triangle_mat[:, 3] * np.exp(-theta_to_strongest**2 / 2 / triangle_mat[:, 2]**2)).reshape(-1, 1)
                A = triangle_area
                rcs_sqrt = G * A * N
                angle_ = abs(np.arccos(np.sum(rb*triangle_norm, axis=-1, keepdims=True)))
                # 极化方向选择或同时计算 jihua_h(angle_, triangle_mat[:, 0], triangle_mat[:, 1])
                triangle_rev = jihua_v(angle_, triangle_mat[:, 0], triangle_mat[:, 1])
                Gtx = 10**(radar.radar_prop['transmitter'].txchannel_prop['antenna_gains'][tx_index] / 20)
                Grx = 10**(radar.radar_prop['receiver'].rxchannel_prop['antenna_gains'][rx_index] / 20)
                A_1 =   Gtx * \
                        Grx * \
                        wavelength * np.sqrt(radar.radar_prop['transmitter'].rf_prop['tx_power'])  / \
                        ((4 * np.pi)**1.5 * distance**2) \
                        * rcs_sqrt.flatten() \
                        * triangle_rev
                tm_shape = list(A_1.shape)
                # del tm_shape[-1]
                tm_shape.append(sample_num)
                t_m = np.linspace(0, chirp_ramp_time, sample_num).reshape([1,sample_num])
                t_m = np.broadcast_to(t_m, tm_shape)

                if_sig = np.zeros_like(t_m, dtype=np.complex128)
                for sample_index in range(sample_num):
                    if_sig[:,sample_index] = A_1.reshape(-1) * np.exp(-1j*2*np.pi* (f_0*delay - frequency_slope*delay**2/2 + frequency_slope*t_m[:,sample_index]*delay))
                signal_res[:, tx_index, rx_index, :] = if_sig.sum(axis=0)
        signal_res = np.repeat(signal_res[:, :, :, np.newaxis, :], chirp_num, axis=3)
        t2 = time()
        print("------------------------------------------------------------------------------------------------")
        print(f"Calculate Env reflection IF signal cost: {t2 - t1:.6f} 秒")
        return signal_res
    mesh_path = env[0]['model']
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh = mesh.translate(env[0]['location'])
    R = mesh.get_rotation_matrix_from_xyz(env[0]['rotation'])
    mesh = mesh.rotate(R, center=env[0]['rotation'])
    triangles = np.array(mesh.triangles)
    vertices = np.array(mesh.vertices)


    # 在这里定义需要保存在每个投影面像素的信息
    res_h = 128
    res_v = 64
    triangle_area = np.zeros((res_v, res_h), dtype=float)
    triangle_norm = np.zeros((res_v, res_h, 3), dtype=float)
    triangle_pos = np.zeros((res_v, res_h, 3), dtype=float)
    triangle_dis = np.zeros((res_v, res_h), dtype=float)
    triangle_dis[:, :] = 1000
    triangle_flag = np.zeros((res_v, res_h))
    triangle_mat = np.zeros((res_v, res_h, 4), dtype=float)
    triangle_mat[:, :, 0] = 1    # 默认值
    triangle_mat[:, :, 1] = 1
    triangle_mat[:, :, 2] = 1.95
    triangle_mat[:, :, 3] = 0.02

    # 显示地遍历所有的投影面。一旦在这里初始化投影面的信息，后续根据每帧的人体变化，只遍历投影面去修改其属性即可，不用在遍历所有mesh的三角表面了。
    # 投影面位置 3m
    drop = 3
    res_drop_v = np.tan(np.radians(60)) * drop / res_v * 2
    res_drop_h = np.tan(np.radians(60)) * drop / res_h * 2

    for tri_index, (x_i, y_i, z_i) in tqdm(enumerate(triangles)):
        vertices_mid = (vertices[x_i] + vertices[y_i] + vertices[z_i]) / 3
        distance = np.sqrt(np.dot(vertices_mid, vertices_mid))
        scale_ = abs(vertices_mid[1] / drop)
        x_in_projection = int(vertices_mid[0] / scale_ / res_drop_v) + int(res_v/2)
        z_in_projection = int(vertices_mid[2] / scale_ / res_drop_h) + int(res_h/2)
        if triangle_dis[x_in_projection, z_in_projection] > distance:
            triangle_flag[x_in_projection, z_in_projection] = 1
            triangle_dis[x_in_projection, z_in_projection] = distance
            triangle_pos[x_in_projection, z_in_projection] = vertices_mid
            triangle_area[x_in_projection, z_in_projection] = cal_area(vertices[x_i], vertices[y_i], vertices[z_i])
            triangle_norm[x_in_projection, z_in_projection] = cal_norm(vertices[x_i], vertices[y_i], vertices[z_i])
            for material in materials:
                if is_point_in_object(vertices_mid, material):
                    triangle_mat[x_in_projection, z_in_projection, 0] = material["epsi"]
                    triangle_mat[x_in_projection, z_in_projection, 1] = material["sigma_jihua"]
                    triangle_mat[x_in_projection, z_in_projection, 2] = material["sigma"]
                    triangle_mat[x_in_projection, z_in_projection, 3] = material["an"]
    triangle_area_corr = triangle_area[triangle_flag==1].reshape(-1)
    triangle_norm_corr = triangle_norm[triangle_flag==1].reshape(-1, 3)
    triangle_pos_corr = triangle_pos[triangle_flag==1].reshape(-1, 3)
    triangle_mat_corr = triangle_mat[triangle_flag==1].reshape(-1, 4)

    # 可视化不同反射物的材质系数
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.heatmap(triangle_mat[:, :, 0])
    plt.show()

    signal = env_once(radar, triangle_pos_corr, triangle_area_corr, triangle_norm_corr, triangle_mat_corr)
    return signal, triangle_pos_corr, triangle_area_corr, triangle_norm_corr, triangle_mat_corr

def combined_sim(radar, persons_loc, radius_per_vertex, triangle_pos_corr, triangle_area_corr, triangle_norm_corr, triangle_mat_corr, person_area, person_norm):
    def env_once(radar, persons_loc, radius_per_vertex, triangle_pos, triangle_area, triangle_norm, triangle_mat, person_area, person_norm):
        theta_0 = 2.0277
        fai_0 = 0.6903
        sigma = 1.11
        pi = np.pi
        # 这里有个假设:在一帧的时间内，多径反射的路径不随chirp变换
        # persons_loc [chirps, points, 3]
        # radius_per_vertex (1, points)
        # triangle pos(64, 64, 3) area(64, 64), norm(64, 64, 3)
        triangle_pos = triangle_pos.reshape(-1, 3)
        triangle_area = triangle_area.reshape(-1, 1)
        triangle_norm = triangle_norm.reshape(-1, 3)
        triangle_mat = triangle_mat.reshape(-1, 4)
        persons_loc = persons_loc.reshape((chirp_num, 1, -1, 3))   # chirps, 1, points, 3
        person_norm = person_norm.reshape((chirp_num, 1, -1, 3))
        person_area = person_area.reshape((chirp_num, 1, -1, 1))
        t1 = time()
        tem1 = triangle_pos.reshape((-1, 1, 3))
        tem_pl = persons_loc[0, :, :, :]
        o_c = ne.evaluate("(tem1 - tem_pl)")
        c = ne.evaluate("sum(o_c**2, axis=2)") - radius_per_vertex.reshape(1, 1, -1)**2
        tx_loc = radar.radar_prop['transmitter'].txchannel_prop['locations'][0] + radar.radar_prop['location']
        rx_loc = radar.radar_prop['receiver'].rxchannel_prop['locations'][0] + radar.radar_prop['location']
        tem1 = ne.evaluate("sum((triangle_pos - tx_loc)**2, axis=1)").reshape((-1, 1))
        r_in_stand = ne.evaluate("(triangle_pos - tx_loc) / sqrt(tem1)")
        tem2 = ne.evaluate("sum((triangle_pos - tx_loc)*triangle_norm, axis=1)").reshape((-1, 1))
        rb = ne.evaluate("r_in_stand - 2*(tem2 / sqrt(tem1))*triangle_norm")
        a = np.ones((rb.shape[0],), dtype=float)
        tem1 = np.repeat(rb.reshape((-1, 1, 3)), o_c.shape[1], axis=1)
        b = ne.evaluate("sum(o_c*tem1, axis=2)") * 2
        tem1 = np.repeat(a.reshape(-1, 1), o_c.shape[1], axis=1)
        delta = ne.evaluate("b*b - 4*tem1*c")
        mask = delta.squeeze() > 0
        triangle_dis = ne.evaluate("sum(triangle_pos**2, axis=1)")
        tem1 = triangle_pos[:, 2]
        tem2 = tx_loc[2]
        theta = ne.evaluate("arcsin((tem2 - tem1) / sqrt(triangle_dis))")
        tem1 = triangle_pos[:, 0]
        tem2 = tx_loc[0]
        fai = ne.evaluate("arcsin((tem2 - tem1) / sqrt(triangle_dis))")
        a_gain = ne.evaluate("10**((96.148*exp(-(theta**2 / 2 / theta_0**2)) - 100)/10)")
        e_gain = ne.evaluate("10**((95.363*exp(-(fai**2 / 2 / fai_0**2)) - 100)/10)")
        G = ne.evaluate("a_gain*e_gain*10**5.6")
        triangle_rev = triangle_mat[:, 3].reshape(-1, 1)
        r_in_stand = rb.reshape(-1, 1, 3)
        tem_pn = person_norm[0, :, :, :]
        tem1 = ne.evaluate("sum(r_in_stand*tem_pn, axis=2)").reshape((r_in_stand.shape[0], -1, 1))
        tem2 = ne.evaluate("sum(r_in_stand**2, axis=2)").reshape((r_in_stand.shape[0], -1, 1))
        rb = ne.evaluate("r_in_stand - tem_pn*2*tem1/sqrt(tem2)")
        tem1 = rx_loc - tem_pl
        tem2 = ne.evaluate("sum(tem1*rb, axis=2)")
        tem3 = ne.evaluate("sum(tem1**2, axis=2)")
        theta_to_strongest = ne.evaluate("arccos(tem2 / sqrt(tem3))")
        N = ne.evaluate("exp(-theta_to_strongest**2 / 2 / sigma**2)")
        A = person_area[0, :, :, 0]
        G = G.reshape(-1, 1)
        rcs_sqrt = ne.evaluate("triangle_rev * G * N * A")[mask]
        sinal_res = np.zeros([cal_frame_num, tx_num, rx_num, chirp_num, sample_num]).astype(np.complex128)
        vir_ant = np.zeros((3, 4, 2, 3), dtype=float)
        for tx_index in range(radar.radar_prop['transmitter'].txchannel_prop['size']):
            for rx_index in range(radar.radar_prop['receiver'].rxchannel_prop['size']):
                vir_ant[tx_index, rx_index, 0] = radar.radar_prop['transmitter'].txchannel_prop['locations'][tx_index] + radar.radar_prop['location']
                vir_ant[tx_index, rx_index, 1] = radar.radar_prop['receiver'].rxchannel_prop['locations'][rx_index] + radar.radar_prop['location']
        vir_ant = vir_ant.reshape(12, 2, -1)
        vir_ant_tx = vir_ant[:, 0, :].reshape(12, 1, 1, 1, -1)
        vir_ant_rx = vir_ant[:, 1, :].reshape(12, 1, 1, 1, -1)
        # (antenna, chirp, env, men, xyz)
        persons_loc = persons_loc.reshape((1, chirp_num, 1, -1, 3))
        person_norm = person_norm.reshape((1, chirp_num, 1, -1, 3))
        person_area = person_area.reshape((1, chirp_num, 1, -1, 1))
        tem_tl = triangle_pos.reshape(1, 1, -1, 1, 3)
        tem1 = ne.evaluate("sum((tem_tl-vir_ant_tx)**2, axis=4)")
        tem2 = ne.evaluate("sum((persons_loc-vir_ant_rx)**2, axis=4)")
        distance_all = ne.evaluate("sqrt(tem1)+sqrt(tem2)")
        filtered_rcs_sqrt = rcs_sqrt.reshape(1, 1, -1)
        filtered_distance = distance_all[:, :, mask]
        delay = ne.evaluate("filtered_distance / light_v")
        Gtx = 10**(radar.radar_prop['transmitter'].txchannel_prop['antenna_gains'][0] / 20)
        Grx = 10**(radar.radar_prop['receiver'].rxchannel_prop['antenna_gains'][0] / 20)
        tem1 = np.sqrt(radar.radar_prop['transmitter'].rf_prop['tx_power'])
        A_1 = ne.evaluate("Gtx * Grx * wavelength * tem1 * filtered_rcs_sqrt / ((4 * pi)**1.5 * filtered_distance**2)")
        tm_shape = list(A_1.shape)
        tm_shape.append(sample_num)
        t_m = np.linspace(0, chirp_ramp_time, sample_num).reshape([1, 1, 1, sample_num])
        t_m = np.broadcast_to(t_m, tm_shape)
        delay = delay.reshape(12, chirp_num, -1, 1)
        A_1 = A_1.reshape(12, chirp_num, -1, 1)
        if_sig_c = ne.evaluate("A_1 * exp(-1j*2*pi* (f_0*delay - frequency_slope*delay**2/2 + frequency_slope*t_m*delay))")
        temp_sis_c = ne.evaluate("sum(if_sig_c, axis=2)").reshape((3, 4, chirp_num, -1))
        if temp_sis_c.shape[3] == 256:
            sinal_res[0, :, :, :, :] = temp_sis_c
        t2 = time()
        print(f"Calculate multipath IF signal cost:{t2 - t1:.6f}秒")
        return sinal_res
    signal = env_once(radar, persons_loc, radius_per_vertex, triangle_pos_corr, triangle_area_corr, triangle_norm_corr, triangle_mat_corr, person_area, person_norm)
    return signal

def combined_sim_2(radar, persons_loc, radius_per_vertex, triangle_pos_corr, triangle_area_corr, triangle_norm_corr, triangle_mat_corr, person_area, person_norm):
    def env_once(radar, persons_loc, radius_per_vertex, triangle_pos, triangle_area, triangle_norm, triangle_mat, person_area, person_norm):
        theta_0 = 2.0277
        fai_0 = 0.6903
        sigma = 1.11
        pi = np.pi
        # persons_loc [chirps, points, 3]
        # radius_per_vertex (1, points)
        # triangle pos(64, 64, 3) area(64, 64), norm(64, 64, 3)
        triangle_pos = triangle_pos.reshape(-1, 3)
        triangle_area = triangle_area.reshape(-1, 1)
        triangle_norm = triangle_norm.reshape(-1, 3)
        triangle_mat = triangle_mat.reshape(-1, 4)
        persons_loc = persons_loc.reshape((chirp_num, 1, -1, 3))    # chirps, 1, points, 3
        person_norm = person_norm.reshape((chirp_num, 1, -1, 3))
        person_area = person_area.reshape((chirp_num, 1, -1, 1))
        t1 = time()
        radius_per_vertex = np.full((triangle_pos.shape[0], 1), 0.02)
        tem1 = triangle_pos.reshape(( -1, 1, 3))
        tem_pl = persons_loc[0, :, :, :]
        o_c = ne.evaluate("(tem1 - tem_pl)")
        c = ne.evaluate("sum(o_c**2, axis=2)") - radius_per_vertex**2
        tx_loc = radar.radar_prop['transmitter'].txchannel_prop['locations'][0] + radar.radar_prop['location']
        rx_loc = radar.radar_prop['receiver'].rxchannel_prop['locations'][0] + radar.radar_prop['location']
        tem_pl = persons_loc[0, 0, :, :]
        tem1 = ne.evaluate("sum((tem_pl - tx_loc)**2, axis=1)").reshape((-1, 1))
        r_in_stand = ne.evaluate("(tem_pl - tx_loc) / sqrt(tem1)")
        tem_pn = person_norm[0, 0, :, :]
        tem2 = ne.evaluate("(sum((tem_pl - tx_loc)*tem_pn, axis=1))").reshape((-1, 1))
        rb = ne.evaluate("r_in_stand - 2 * (tem2 / sqrt(tem1))*tem_pn")
        a = np.ones((rb.shape[0],),dtype=float)
        tem1 = np.repeat(rb.reshape((1, -1, 3)), o_c.shape[0], axis=0)
        b = ne.evaluate("sum(o_c*tem1, axis=2)")*2
        tem1 = np.repeat(a.reshape(1, -1), o_c.shape[0], axis=0)
        delta = ne.evaluate("b*b - 4*tem1*c")
        mask = delta.squeeze() > 0
        person_dis2 = ne.evaluate("sum(tem_pl**2, axis=1)")
        tem1 = tem_pl[:, 2]
        tem2 = tx_loc[2]
        theta = ne.evaluate("arcsin((tem2 - tem1) / sqrt(person_dis2))")
        tem1 = tem_pl[:, 0]
        tem2 = tx_loc[0]
        fai = ne.evaluate("arcsin((tem2 - tem1) / sqrt(person_dis2))")
        a_gain = ne.evaluate("10**((96.148*exp(-(theta**2 / 2 / theta_0**2)) - 100)/10)")
        e_gain = ne.evaluate("10**((95.363*exp(-(fai**2 / 2 / fai_0**2)) - 100)/10)")
        G = ne.evaluate("a_gain*e_gain*10**5.6")
        G = G.reshape(1, -1)
        triangle_rev = triangle_mat[:, 3].reshape(-1, 1)
        r_in_stand = rb.reshape(1, -1, 3)
        tem_tn = triangle_norm.reshape((-1, 1, 3))
        tem1 = ne.evaluate("sum(r_in_stand * tem_tn, axis=2)").reshape((-1, r_in_stand.shape[1], 1))
        tem2 = ne.evaluate("sum(r_in_stand**2, axis=2)").reshape((-1, r_in_stand.shape[1], 1))
        rb = ne.evaluate("r_in_stand - tem_tn*2*tem1/sqrt(tem2)")
        tem1 = (rx_loc - triangle_pos).reshape((-1, 1, 3))
        tem2 = ne.evaluate("sum(tem1*rb, axis=2)")
        tem3 = ne.evaluate("sum(tem1**2, axis=2)")
        theta_to_strongest = ne.evaluate("arccos(tem2 / sqrt(tem3))")
        N = ne.evaluate("exp(-theta_to_strongest**2 / 2 / sigma**2)")
        A = triangle_area
        rcs_sqrt = ne.evaluate("triangle_rev * G * N * A")[mask]
        sinal_res = np.zeros([cal_frame_num, tx_num, rx_num, chirp_num, sample_num]).astype(np.complex128)
        vir_ant = np.zeros((3, 4, 2, 3), dtype=float)
        for tx_index in range(radar.radar_prop['transmitter'].txchannel_prop['size']):
            for rx_index in range(radar.radar_prop['receiver'].rxchannel_prop['size']):
                vir_ant[tx_index, rx_index, 0] = radar.radar_prop['transmitter'].txchannel_prop['locations'][tx_index] + radar.radar_prop['location']
                vir_ant[tx_index, rx_index, 1] = radar.radar_prop['receiver'].rxchannel_prop['locations'][rx_index] + radar.radar_prop['location']
        vir_ant = vir_ant.reshape(12, 2, -1)
        vir_ant_tx = vir_ant[:, 0, :].reshape(12, 1, 1, 1, -1)
        vir_ant_rx = vir_ant[:, 1, :].reshape(12, 1, 1, 1, -1)
        # (antenna, chirp, env, men, xyz)
        persons_loc = persons_loc.reshape((1, chirp_num, 1, -1, 3))
        person_norm = person_norm.reshape((1, chirp_num, 1, -1, 3))
        person_area = person_area.reshape((1, chirp_num, 1, -1, 1))
        tem_tl = triangle_pos.reshape(1, 1, -1, 1, 3)
        tem1 = ne.evaluate("sum((tem_tl-vir_ant_rx)**2, axis=4)")
        tem2 = ne.evaluate("sum((persons_loc-vir_ant_tx)**2, axis=4)")
        distance_all = ne.evaluate("sqrt(tem1)+sqrt(tem2)")
        filtered_rcs_sqrt = rcs_sqrt.reshape(1, 1, -1)
        filtered_distance = distance_all[:, :, mask]
        delay = ne.evaluate("filtered_distance / light_v")
        Gtx = 10**(radar.radar_prop['transmitter'].txchannel_prop['antenna_gains'][0] / 20)
        Grx = 10**(radar.radar_prop['receiver'].rxchannel_prop['antenna_gains'][0] / 20)
        tem1 = np.sqrt(radar.radar_prop['transmitter'].rf_prop['tx_power'])
        A_1 = ne.evaluate("Gtx * Grx * wavelength * tem1 * filtered_rcs_sqrt / ((4 * pi)**1.5 * filtered_distance**2)")
        tm_shape = list(A_1.shape)
        tm_shape.append(sample_num)
        t_m = np.linspace(0, chirp_ramp_time, sample_num).reshape([1,1,1,sample_num])
        t_m = np.broadcast_to(t_m, tm_shape)
        delay = delay.reshape(12, chirp_num, -1, 1)
        A_1 = A_1.reshape(12, chirp_num, -1, 1)
        if_sig_c = ne.evaluate("A_1 * exp(-1j*2*pi*(f_0*delay - frequency_slope*delay**2/2 + frequency_slope*t_m*delay))")
        temp_sis_c = ne.evaluate("sum(if_sig_c, axis=2)").reshape((3, 4, chirp_num, -1))
        if temp_sis_c.shape[3] == 256:
            sinal_res[0, :, :, :, :] = temp_sis_c
        t4 = time()
        print(f"Calculate another multipath IF signal cost:{t4 - t1:.6f}秒")
        return sinal_res
    signal = env_once(radar, persons_loc, radius_per_vertex, triangle_pos_corr, triangle_area_corr, triangle_norm_corr, triangle_mat_corr, person_area, person_norm)
    return signal