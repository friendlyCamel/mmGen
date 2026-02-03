from radar_define import *
# 一些全局变量
syn_frame_num = 44
vis = False
# 以下变量用于插值
init_points_num = 10475             # 一个mesh包含的点数
mesh_scale = 0.1                   # 用于生成信号的mesh点的比例，可根据设备硬件能力适度提高，在原实验中硬件：intel i5-8500 + 32G
triangles_num = 20908
downsample_triangles_num = int(triangles_num * mesh_scale)
mesh_period = 0.0666667             # 生成2帧mesh的间隔时间（即内插满255个chirp的时间）s
mesh_frame_num = int(mesh_period / chirp_period / chirp_num)
cal_frame_num = 1