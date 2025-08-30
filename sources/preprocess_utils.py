import os
import sys
import time
import glob
import numpy as np
import sympy as sp
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fftpack import fft, ifft
from mpl_toolkits.mplot3d import Axes3D
from config import PointCloudConfig as cfg


this_path = os.path.dirname(__file__)
sys.path.append(os.path.join(this_path, '../'))


att_phase_shift_6843 = np.array([
    [0, -0.2105, 0.0019, 0.0514],
    [0.7710, 0.5740, 0.8460, 1.0143],
    [0.2300, -0.0346, 0.2101, 0.2928]
])


'''============= Read One Frame ============='''
class BufferReader:
    def __init__(self, path):
        self.path = path
        self.bin_fid = open(path, 'rb')

    def getNextBuffer(self):
        buffer = np.frombuffer(self.bin_fid.read(cfg.FRAME_SIZE * 2 * 2), dtype=np.int16)
        return buffer

    def close(self):
        self.bin_fid.close()

    def __del__(self):
        self.close()

def attPhaseComps(frame):
    arr_comps = np.reshape(att_phase_shift_6843, (3, 4, 1, 1))
    arr_comps = np.exp(-1 * 1j * arr_comps, dtype = np.complex128)
    frame = arr_comps * frame
    return frame
# return frame(tx rx chirpNum sampleNum到频域的值)
def rangeFFT(frame):
    # Black加窗
    r_win = np.blackman(cfg.SAMPLE_NUM)
    frame = np.fft.fft(frame * r_win, axis = -1)
    return frame

# return frame(tx rx chirpNum sampleNum频域的值)
def rangeCut(frame):
    frame[:, :, :, cfg.MAX_R_I:cfg.SAMPLE_NUM] = 1e-16
    frame[:, :, :, 0:10] = 1e-18
    return frame

def clutterRemoval(frame):
    frame = frame.transpose(2, 1, 0, 3)
    mean = frame.mean(0)
    frame = frame - mean
    return frame.transpose(2, 1, 0, 3)
def dopplerFFT(frame):
    # Hanning窗
    v_win = np.reshape(np.hanning(cfg.CHIRP_NUM), (1, 1, -1, 1))
    frame = np.fft.fft(frame * v_win, axis=2)
    frame = np.fft.fftshift(frame, axes=2)
    return frame


# return frame(tx, rx, chirp, sample)
def buffer2Frame(buffer):
    raw_frame = np.reshape(buffer, (-1, 2, 2))
    cpx_data = raw_frame[:, 0, :] + 1j * raw_frame[:, 1, :]
    cpx_data = np.reshape(cpx_data, (-1, 1))
    cpx_data = np.reshape(cpx_data, (cfg.CHIRP_NUM, cfg.TX_NUM, cfg.RX_NUM, cfg.SAMPLE_NUM))
    cpx_data = cpx_data.transpose(1, 2, 0, 3)
    # print(type(cpx_data[0, 0, 0, 0]))
    return cpx_data


def tdmPhaseComps(frame):
    for m in range(cfg.CHIRP_NUM):
        for n in range(1, cfg.TX_NUM):
            frame[n, :, m, :] *= np.exp(-1j * n * 2 * np.pi / cfg.TX_NUM * (m - cfg.CHIRP_NUM // 2) / cfg.CHIRP_NUM)
    return frame

def angleFFT(frame, frame_filter):
    f_frame = frame[:, :, frame_filter == True]
    f_frame = f_frame.reshape(12, -1)
    azimuth_frame = f_frame[: 2 * cfg.RX_NUM, :]
    elevation_frame = f_frame[2 * cfg.RX_NUM :, :]
    azimuth_frame = np.fft.fft(azimuth_frame, cfg.ANGLE_PADDED_NUM, axis = 0)
    azimuth_frame = np.fft.fftshift(azimuth_frame, axes = 0)
    elevation_frame = np.fft.fft(elevation_frame, cfg.ANGLE_PADDED_NUM, axis = 0)
    elevation_frame = np.fft.fftshift(elevation_frame, axes = 0)
    return azimuth_frame, elevation_frame

def frameHPF(frame):
    # temp_num = cfg.CHIRP_NUM * cfg.MAX_R_I - cfg.HPF_NUM
    temp_num = 255*256 - 4096
    doppler_db = frame
    rss_thr = np.partition(doppler_db.ravel(), temp_num - 1)[temp_num - 1]
    frame_filter =  doppler_db > rss_thr
    filter_indices = np.argwhere(frame_filter == 1)
    return doppler_db, frame_filter, filter_indices

def getWxWz(azimuth_fft, elevation_fft, astd):
    # 设置寻找峰值的相关参数，例如阈值、最小峰间距等
    # 自定义阈值astd，信号标准差
    min_distance = 5  # 最小峰宽
    # 寻找峰值
    a_i_list, _ = find_peaks(abs(azimuth_fft), height=2*astd, width=min_distance)
    if len(a_i_list) != 0:
        # peak_values = azimuth_fft[peaks]  # 这是峰值对应的幅度值
        # peak_indices = peaks  # 这是峰值对应的频率索引（在FFT结果中的位置）
        wx_l = []
        wz_l = []
        for a_i in a_i_list:
            e_i = a_i
            wx = (a_i - 64 // 2) / 64 * 2 * np.pi
            wz = np.angle(azimuth_fft[a_i] * elevation_fft[e_i].conj() * np.exp(2j * wx))
            wx_l.append(wx)
            wz_l.append(wz)
        return wx_l, wz_l
    else:
        return [], []

def getRawPC(doppler_db, frame_filter, filter_indices, azimuth_frame, elevation_frame):
    param_R = filter_indices[:, 1] * cfg.RANGE_RES
    # param_V = (filter_indices[:, 0] - 120 // 2) * 0.025
    list_wx = []
    list_wz = []
    list_wy = []
    azimuth_frame = azimuth_frame.transpose(1, 0)
    elevation_frame = elevation_frame.transpose(1, 0)
    astd = abs(azimuth_frame).std()
    for R_i, (a_f, e_f) in enumerate(zip(azimuth_frame, elevation_frame)):
        wx, wz = getWxWz(a_f, e_f, astd)
        if len(wx) > 0:
            list_wx.append(np.array(wx) * param_R[R_i] / np.pi)
            list_wz.append(np.array(wz) * param_R[R_i] / np.pi)
            wy = param_R[R_i] ** 2 - list_wz[-1]**2 - list_wx[-1]**2
            if np.all(wy > 0):
                list_wy.append(wy)
            else:
                list_wx.pop()
                list_wz.pop()
    param_X = np.concatenate(list_wx)
    param_Z = np.concatenate(list_wz)
    param_Y = np.concatenate(list_wy)
    param_Y = np.sqrt(param_Y)
    # param_RSS = np.log10(doppler_db[frame_filter])
    point_cloud = np.concatenate((param_X, param_Y, param_Z))
    return point_cloud.reshape(3, -1)

