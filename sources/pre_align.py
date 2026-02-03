import os
import cv2
import numpy as np
import os.path as opt
from datetime import datetime
from datetime import timedelta
from collections import Counter
import pykinect_azure as pykinect


init_dir = "../examples/raw_mmWave_human"
save_dir = "../examples/raw_mmWave_human"
aligned_index = []
is_read_rgb = True
is_read_depth = False
is_read_skeleton = True

def align_time(dir):
    global init_dir
    global save_dir
    global aligned_index
    
    mm_frame = 45
    period_second = 3
    
    # 读取雷达的开始时间
    log_file = open(opt.join(init_dir,dir,'LogFile.txt'),'r')
    aa =log_file.readlines()
    mm_start_time = (datetime.strptime(aa[1].strip(),'%Y %m %d %H %M %S %f')-datetime.strptime(aa[0].strip(),'%Y %m %d %H %M %S %f'))/2+datetime.strptime(aa[0].strip(),'%Y %m %d %H %M %S %f')
    time_period = timedelta(seconds=period_second)
    # 读取kinect每一帧的记录时间
    kinect_time_array = []
    with open(opt.join(init_dir, dir, 'kinect', 'time_log.txt'), 'r') as f:
        for line in f.readlines():
            kinect_time=datetime.strptime(line.strip(),'%Y %m %d %H %M %S %f')
            kinect_time_array.append(kinect_time)
            
    mm_time_array = []
    aligned_index = []
    for i in range(mm_frame):
        mm_time = mm_start_time + time_period/mm_frame*(i)
        mm_time_array.append(mm_time)
        index = np.searchsorted(kinect_time_array, mm_time)
        #print(index)
        aligned_index.append(index)
    # 判断kinect记录时间是否完全包含雷达时间
    if (aligned_index[0]==0 or aligned_index[int(mm_frame-1)]>len(kinect_time_array)-10):
        print("ERROR! [{}] can't be used! index[0] is {}, index[-1] is {} ".format(dir,aligned_index[0],aligned_index[mm_frame-1]))
        return False
        
    with open(opt.join(save_dir,dir,'aligned_index.txt'),'w') as f:
        for line in aligned_index:
            f.write(str(line))
            f.write('\n')
    print("{} is aligned! index[0] is {}, index[-1] is {} ".format(dir,aligned_index[0],aligned_index[mm_frame-1]))
    return True

    
def get_kinect_map(dir):
    global init_dir
    global save_dir
    global aligned_index
    
    mm_frame = 45
    if (len(aligned_index)!=mm_frame):
        print('ERROR! aligned_num!={}'.format(mm_frame))
        return
    print('processing {}'.format(dir))
    aligned_index_counter = Counter(aligned_index)
            
    if not os.path.exists(opt.join(save_dir,dir,'rgb')):
        os.makedirs(opt.join(save_dir,dir,'rgb'))
        
    video_filename = opt.join(init_dir,dir,'kinect','recoder.mkv')
    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries(track_body=True)
    # Start playback
    playback = pykinect.start_playback(video_filename)
    playback_config = playback.get_record_configuration()
    # print(playback_config)
    playback_calibration = playback.get_calibration()
    # Start body tracker
    bodyTracker = pykinect.start_body_tracker(calibration=playback_calibration)

    mmwave_index = -1
    kinect_index = -1
    # 帧数 人数 32个关节 关节xyz坐标
    skeleton_3d = np.zeros([mm_frame,1,32,3])

    while True:
        # Get camera capture
        ret, capture = playback.update()
        if not ret:
            break
        
        kinect_index +=1
        if kinect_index not in aligned_index:
            continue
        
        # 有可能两个雷达帧映射至同一个kinect_index 为了避免这个问题 相机帧率最好等于毫米波帧率 设置为15fps
        for i in range(aligned_index_counter[kinect_index]):
            # kinect的当前帧与雷达对齐
            mmwave_index += 1
            # Get body tracker frame
            try:
                body_frame = bodyTracker.update(capture=capture)
            except:
                ret, capture = playback.update()
                kinect_index +=1
                body_frame = bodyTracker.update(capture=capture)
            # Get color image
            ret_color, color_image = capture.get_color_image()
            # Get the colored depth
            ret_depth, depth_image = capture.get_depth_image()
            # Get the colored depth
            ret_transformed_depth, transformed_depth_image = capture.get_transformed_depth_image()
            normed_transformed_depth_image = capture.color_depth_image(transformed_depth_image)
            cv2.imwrite(opt.join(save_dir, dir, 'rgb', '{}.jpg'.format(mmwave_index)), color_image)
            try:
                temp = body_frame.get_body(0).numpy()[:, :3] / 1000.
            except:
                temp = skeleton_3d[mmwave_index-1, 0, :, :]
            skeleton_3d[mmwave_index, 0, :, :] = temp

    check = np.matmul(skeleton_3d[:, 0, pykinect.K4ABT_JOINT_KNEE_LEFT, :], ([1, 0, 0], [0, 0, -1], [0, 1, 0]))

    if mmwave_index+1 != mm_frame:
        print('ERROR! processed_aligned_num is {}!={}'.format(mmwave_index, mm_frame))
    with open(opt.join(save_dir, dir, 'skeleton_LEFT_KNEE.npy'.format(mmwave_index)), "wb") as skelon_write:
        np.save(skelon_write, check)
    with open(opt.join(save_dir, dir, 'is_done.txt'), 'w') as f:
        f.write('done')
        
if __name__ == "__main__":
    dirs = os.listdir(init_dir)
    dirs.sort()
    for dir in dirs:
        if os.path.isdir(init_dir+'/'+dir):
            if not os.path.exists(opt.join(save_dir, dir)):
                os.makedirs(opt.join(save_dir, dir))
            if not opt.exists(opt.join(save_dir, dir, 'is_done.txt')):
                if (align_time(dir)):
                    get_kinect_map(dir)
        