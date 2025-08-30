import trimesh
import numpy as np
import pyvista as pv
import open3d as o3d
import pymeshfix as mf
from pymeshfix._meshfix import PyTMesh
from pykinect_azure.utils import Open3dVisualizer

if __name__ == "__main__":
    path = "..\\examples\\scene1.mkv"
    obj_path = "..\\examples\\"

    reader = o3d.io.AzureKinectMKVReader()
    open3dVisualizer = Open3dVisualizer()
    reader.open(path)
    if not reader.is_opened():
        raise RuntimeError("Unable to open file")

    idx = 0
    flag_exit = False
    flag_play = True
    print("->正在加载点云... ")
    while not reader.is_eof() and not flag_exit:
        if flag_play:
            rgbd = reader.next_frame()
            if rgbd is None:
                continue
            color_image = rgbd.color
            depth_image = rgbd.depth
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image, depth_image,
                convert_rgb_to_intensity=False
            )
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
            )
            # 720p的相机内参
            intrinsic.set_intrinsics(1280, 720, 599.795593, 599.633118, 645.792786, 372.238983
            )
            point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
            open3dVisualizer(point_cloud.points)
            idx += 1
            break
    reader.close()
    pcd = point_cloud

    print("->正在转换点云坐标")
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])
    downpcd = pcd

    # 法线估计
    radius1 = 0.1   # 搜索半径
    max_nn = 50     # 邻域内用于估算法线的最大点数
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius1, max_nn))     # 执行法线估计
    o3d.visualization.draw_geometries([downpcd],                                                     # 可视化
                                      window_name="可视化参数设置",
                                      point_show_normal=True)
    # 滚球半径的估计
    distances = downpcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        downpcd,
        o3d.utility.DoubleVector([radius, radius * 2]))
    print("当前设置下每个反射面的面积（单位：平方厘米）：{}".format(mesh.get_surface_area()))
    o3d.visualization.draw_geometries([mesh], window_name='Open3D downSample',
                                    point_show_normal=True, mesh_show_wireframe=True, mesh_show_back_face=True)
    # 从open3d创建具有顶点和面的三角形网格
    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    # tri_mesh格式保存，方便扩展
    tri_mesh.export(obj_path + "temp.ply")
    # 读取mesh
    orig_mesh = pv.read(obj_path + "temp.ply")
    # 计算mesh的孔洞
    meshfix = mf.MeshFix(orig_mesh)
    holes = meshfix.extract_holes()
    # 将mesh与孔洞进行叠加展示
    plotter = pv.Plotter()
    plotter.add_mesh(orig_mesh, color=True)
    plotter.add_mesh(holes, color="r", line_width=5)
    plotter.enable_eye_dome_lighting()  # helps depth perception
    _ = plotter.show()

    # mesh空洞修复，构建PyTMesh修复对象mfix，并加载要修复的mesh
    mfix = PyTMesh(False)
    mfix.load_file(obj_path + "temp.ply")

    # 填充最多具有“nbe”个边界边缘的所有孔,如果“refine”为true，添加内部顶点以重现采样周围环境的密度。返回修补的孔数。
    # “nbe”为0（默认值），则修复所有孔
    mfix.fill_small_boundaries(nbe=0, refine=True)

    # 将PyTMesh对象转换为pyvista mesh
    vert, faces = mfix.return_arrays()
    triangles = np.empty((faces.shape[0], 4), dtype=faces.dtype)
    triangles[:, -3:] = faces
    triangles[:, 0] = 3
    mesh = pv.PolyData(vert, triangles)

    # 进行可视化，并保存mesh
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color=True)
    plotter.add_mesh(holes, color="r", line_width=5)
    plotter.enable_eye_dome_lighting()  # helps depth perception
    _ = plotter.show()
    mesh.save(obj_path + "temp.ply".replace(".ply","_r.ply"))

    # 降采样并对比降采样的效果
    mesh = o3d.io.read_triangle_mesh(obj_path + "temp_r.ply")
    voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 128
    print(f'voxel_size = {voxel_size:e}')
    mesh2 = mesh.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average)
    mesh.translate([5,0,0])
    o3d.visualization.draw_geometries([mesh,mesh2], window_name='Open3D downSample',
                                      point_show_normal=True, mesh_show_wireframe=True, mesh_show_back_face=True,)

    # 将降采样的结果按照毫米波的坐标系保存为最终的环境网格
    mesh2 = mesh2.translate((0, 0, 0))
    R = mesh2.get_rotation_matrix_from_xyz((np.pi/2,0,0))
    mesh2 = mesh2.rotate(R, center=(0, 0, 0))
    o3d.io.write_triangle_mesh(obj_path + "final.obj", mesh2, write_vertex_normals=False)
