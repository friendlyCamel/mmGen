"""
    mmGen(one snapshot is all you need: a generalized method for mmWave signal generation)
    author: Friendly Camel
"""

import numpy as np
from time import time
from tqdm import tqdm
from syn_define import *
from radar_define import *
from syn_functions import *

def human_combined_sys(frame_start, radar, persons, env, env_reflection, triangle_pos_corr, triangle_area_corr, triangle_norm_corr, triangle_mat_corr):
    person_reflection, persons_loc, radius_per_vertex, person_area, person_norm = person_once(radar, persons, frame_start)
    combined_reflection = combined_sim(radar, persons_loc, radius_per_vertex, triangle_pos_corr, triangle_area_corr, triangle_norm_corr, triangle_mat_corr, person_area, person_norm)
    combined_reflection2 = combined_sim_2(radar, persons_loc, radius_per_vertex, triangle_pos_corr, triangle_area_corr, triangle_norm_corr, triangle_mat_corr, person_area, person_norm)
    combined_sig = person_reflection + env_reflection + combined_reflection + combined_reflection2
    return frame_start, combined_sig

def scene(radar, targets, syn_frame_num):
    persons = [target for target in targets if target['type'] == 'person']
    env = [target for target in targets if target['type'] == 'env']
    env_reflection, triangle_pos_corr, triangle_area_corr, triangle_norm_corr, triangle_mat_corr = env_sim(radar, env)
    results = []
    for i in tqdm(range(syn_frame_num)):
        frame_start = i
        _, result = human_combined_sys(frame_start, radar, persons, env, env_reflection, triangle_pos_corr, triangle_area_corr, triangle_norm_corr, triangle_mat_corr)
        results.append(result)
    return results

if __name__ == '__main__':
    '''
    使用mmGen生成信号
    '''
    print("Configuring the radar...")
    tx_channel_1 = dict(location=(-2 * wavelength, 0, 0),
                        azimuth_angle=azimuth_angle,
                        azimuth_pattern=azimuth_pattern,
                        elevation_angle=elevation_angle,
                        elevation_pattern=elevation_pattern,
                        delay=0)
    tx_channel_2 = dict(location=(0, 0, 0),
                        azimuth_angle=azimuth_angle,
                        azimuth_pattern=azimuth_pattern,
                        elevation_angle=elevation_angle,
                        elevation_pattern=elevation_pattern,
                        delay= 520e-6)
    tx_channel_3 = dict(location=(-1 * wavelength, 0, 0.5 * wavelength),
                        azimuth_angle=azimuth_angle,
                        azimuth_pattern=azimuth_pattern,
                        elevation_angle=elevation_angle,
                        elevation_pattern=elevation_pattern,
                        delay= 2 * 520e-6)

    tx = Transmitter(f=[f_0, f_0 + frequency_slope*chirp_ramp_time],
                     t=chirp_ramp_time,
                     tx_power=158.5,  # 12dBm ->158.5mW
                     prp=chirp_period,
                     pulses=chirp_num,
                     channels=[tx_channel_1, tx_channel_2, tx_channel_3])

    channels = []
    for idx in range(0, rx_num):
        channels.append(
            dict(
                location=(wavelength / 2 * idx, 0, 0),
                azimuth_angle=azimuth_angle,
                azimuth_pattern=azimuth_pattern,
                elevation_angle=elevation_angle,
                elevation_pattern=elevation_pattern,
            ))

    # 真实毫米波雷达的采样频率原为5600e3 即接收天线后面ADC采样器的采样频率
    # 但模拟生成信号没有采样这个过程,所以这里的采样频率为sample_num/chirp_ramp_time
    # 也就是说，要在一个chirp ramp时间内把256个点采出来就行 （9142e3）
    sample_rate = sample_num/chirp_ramp_time
    rx = Receiver(fs=sample_rate,
                  noise_figure=0,
                  rf_gain=48, # dB
                  baseband_gain=20,
                  load_resistor=500,
                  channels=channels)

    # 场景mesh路径 ../examples/final.obj
    target_1 = {
        'model': '../examples/final.obj',
        # 深度相机和毫米波雷达摆放时的位置矫正
        'location': (0.1, 0, 0.25),
        'speed': (0, 0, 0),
        'rotation': (0, 0, 0),
        'rcs': 10,
        'type': 'env'
    }
    # 人体mesh路径 ../examples/raw_mmWave_human/captured signal/out
    #             ../examples/raw_mmWave_human/captured signal/out/frame0.obj
    #             ../examples/raw_mmWave_human/captured signal/skeleton_LEFT_KNEE.npy
    target_2 = {
        # 传入的人物模型文件夹路径
        'model_dir': '../examples/raw_mmWave_human/captured signal/out',
        'model': '../examples/raw_mmWave_human/captured signal/out/frame0.obj',
        'loc_path': '../examples/raw_mmWave_human/captured signal/skeleton_LEFT_KNEE.npy',
        'speed': (0, 0, 0),
        # 由深度相机坐标和毫米波坐标匹配时产生的偏差
        'rotation': (-np.pi/2-10*np.pi/180, 0, 0),
        'rcs': 20,
        'type': 'person'
    }
    radar_location = (0, 0, 0)
    targets = [target_1, target_2]
    radar = Radar(transmitter=tx, receiver=rx, location=radar_location)
    print("Radar configuration done.")

    if vis:
        vis_scene(radar_location, targets)

    tic = time()
    data = scene(radar, targets, syn_frame_num)
    toc = time()
    print(f"Total time:{(toc - tic) / 60:.6f}分钟")
    data = np.array(data)
    np.save("../examples/raw_mmWave_human/captured signal/sys_sig0_1.npy", data)
    print("DONE!")