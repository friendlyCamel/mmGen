import os
import cv2
import math
import scipy
import numpy as np
import sympy as sp
from tqdm import tqdm
from PIL import Image
import seaborn as sns
from preprocess_utils import *
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt2d, hilbert2, find_peaks

def compute_point_cloud_from_frame(frame):
    try:
        # Range FFT
        frame = rangeFFT(frame)
        frame = np.flip(frame)
        frame = clutterRemoval(frame)
        frame = rangeCut(frame)
        frame_dop = dopplerFFT(frame)
        doppler = np.abs(frame_dop).sum(axis=(0, 1))
        doppler_db, frame_filter, filter_indices = frameHPF(doppler)
        azimuth_frame, elevation_frame = angleFFT(frame_dop[:, :, :, :], frame_filter)
        point_cloud = getRawPC(doppler_db, frame_filter, filter_indices, azimuth_frame, elevation_frame)
        return point_cloud
    except Exception as e:
        print(f"Error computing point cloud: {e}")
        return None

def compute_all_point_clouds(sys_frames):
    point_clouds = []
    for k in tqdm(range(len(sys_frames)), desc='Computing point clouds'):
        frame = sys_frames[k].squeeze()
        point_cloud = compute_point_cloud_from_frame(frame)
        point_clouds.append(point_cloud)
    return point_clouds

def vis_point_cloud(mmw_path, syn_path, rgb_path):
    sys_frames = np.load(syn_path)
    rgb_files = [f for f in os.listdir(rgb_path) if f.lower().endswith('.jpg')]
    rgb_files = sorted(rgb_files, key=lambda fn: int(os.path.splitext(fn)[0]))
    rgb_files = [os.path.join(rgb_path, f) for f in rgb_files]
    syn_point_clouds = compute_all_point_clouds(sys_frames)
    
    plt.ion()
    fig = plt.figure(figsize=(15, 6))
    ax_pc = fig.add_subplot(121, projection='3d')
    ax_img = fig.add_subplot(122)

    ax_pc.set_title('Synthesized Point Cloud')
    ax_img.set_title('RGB Image')
    ax_img.axis('off')
    max_len = max(45, len(sys_frames), len(rgb_files))
    for k in range(max_len):
        t0 = time.time()
        # Get synthesized point cloud
        syn_pc = None
        if k < len(syn_point_clouds):
            syn_pc = syn_point_clouds[k]
        # Get RGB image
        img = None
        if k < len(rgb_files):
            try:
                img = np.array(Image.open(rgb_files[k]).convert('RGB'))
            except Exception as e:
                img = None
        # Clear plots
        ax_pc.cla()
        ax_img.cla()
        # Plot point cloud
        ax_pc.set_title(f'Point Cloud (Frame {k})')
        ax_pc.set_xlabel('X')
        ax_pc.set_ylabel('Y')
        ax_pc.set_zlabel('Z')
        ax_pc.view_init(azim=-0, elev=90)
        ax_pc.set_xlim(-3, 3)
        ax_pc.set_ylim(0, 5)
        ax_pc.set_zlim(-3, 3)
        if syn_pc is not None and syn_pc.shape[1] > 0:
            ax_pc.scatter(syn_pc[0, :], syn_pc[1, :], syn_pc[2, :],
                         c='red', alpha=0.6, s=10, label='Synthesized')
        ax_pc.legend()
        
        # Plot RGB image
        ax_img.set_title('RGB Image')
        ax_img.axis('off')
        if img is not None:
            ax_img.imshow(img)
        else:
            ax_img.text(0.5, 0.5, 'No image', transform=ax_img.transAxes, ha='center')
        # Refresh
        fig.canvas.draw_idle()
        plt.pause(0.1)
    plt.ioff()
    plt.show()

def vis_range_compare(mmw_path, syn_path, rgb_path):
    f_reader = BufferReader(mmw_path)
    sys_frames = np.load(syn_path)
    rgb_files = [f for f in os.listdir(rgb_path) if f.lower().endswith('.jpg')]
    rgb_files = sorted(rgb_files, key=lambda fn: int(os.path.splitext(fn)[0]))
    rgb_files = [os.path.join(rgb_path, f) for f in rgb_files]
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    ax_bin, ax_syn, ax_img = axes

    ax_bin.set_title('Real mmWave')
    ax_syn.set_title('Syn mmWave')
    ax_img.set_title('RGB image')
    ax_bin.set_ylim((0, 1))
    ax_syn.set_ylim((0, 1))
    ax_img.axis('off')

    max_len = max(45, len(sys_frames), len(rgb_files))
    for k in range(max_len):
        t0 = time.time()
        frame_bin = None
        try:
            buffer = f_reader.getNextBuffer()
            if buffer is not None:
                frame_bin = buffer2Frame(buffer)
                frame_bin = attPhaseComps(frame_bin)
                frame_bin = rangeFFT(frame_bin)
                frame_bin = rangeCut(frame_bin)
        except Exception as e:
            frame_bin = None
        frame_syn = None
        if k < len(sys_frames):
            try:
                frame_syn = sys_frames[k].squeeze()
                frame_syn = rangeFFT(frame_syn)
                frame_syn = np.flip(frame_syn)
                frame_syn = rangeCut(frame_syn)
            except Exception as e:
                frame_syn = None

        img = None
        if k < len(rgb_files):
            try:
                img = np.array(Image.open(rgb_files[k]).convert('RGB'))
            except Exception as e:
                img = None
        ax_bin.cla()
        ax_syn.cla()
        ax_img.cla()
        ax_bin.set_title('Real mmWave')
        ax_bin.set_ylim((0, 1))
        if frame_bin is not None:
            try:
                y = np.abs(frame_bin[0, 0, 0])
                y_max = max(y)
                x = np.arange(0, y.size * cfg.RANGE_RES, cfg.RANGE_RES)
                ax_bin.plot(x, y / y_max)
            except Exception:
                y = np.abs(frame_bin.flatten())
                x = np.arange(0, y.size * cfg.RANGE_RES, cfg.RANGE_RES)
                ax_bin.plot(x, y)
        else:
            ax_bin.text(0.5, 0.5, 'no mmwave frame', transform=ax_bin.transAxes, ha='center')

        ax_syn.set_title('Syn mmWave')
        ax_syn.set_ylim((0, 1))
        if frame_syn is not None:
            try:
                y2 = np.abs(frame_syn[0, 0, 0])
                y2_max = max(y2)
                x2 = np.arange(0, y2.size * cfg.RANGE_RES, cfg.RANGE_RES)
                ax_syn.plot(x2, y2 / y2_max)
            except Exception:
                y2 = np.abs(frame_syn.flatten())
                x2 = np.arange(0, y2.size * cfg.RANGE_RES, cfg.RANGE_RES)
                ax_syn.plot(x2, y2)
        else:
            ax_syn.text(0.5, 0.5, 'no synthesized frame', transform=ax_syn.transAxes, ha='center')

        ax_img.set_title('RGB image')
        ax_img.axis('off')
        if img is not None:
            ax_img.imshow(img)
        else:
            # 空占位
            ax_img.text(0.5, 0.5, 'no image', transform=ax_img.transAxes, ha='center')

        # 刷新
        fig.canvas.draw_idle()
        plt.pause(0.05)
        plt.ioff()

def compute_md_from_sys(sys_frames):
    micro_doppler = []
    for k in tqdm(range(len(sys_frames)), desc='read sys mmWave'):
        frame = sys_frames[k].squeeze()
        frame = rangeFFT(frame)
        frame = np.flip(frame)
        frame = clutterRemoval(frame)
        frame = rangeCut(frame)
        frame_dop = dopplerFFT(frame)
        doppler = np.abs(frame_dop).sum(axis=(0, 1))
        micro_doppler.append(np.sum(doppler, axis=-1))
    return np.array(micro_doppler)

def transform_md(md, method='global_max', eps=1e-12):
    """
    method: 'none' | 'per_matrix' | 'db'
    """
    md = md.astype(np.float64)
    if method == 'none':
        out = md
    elif method == 'per_matrix':
        out = md / (md.max() + eps)
    elif method == 'db':
        out = 20 * np.log10(md + eps)
        out = np.clip(out, out.max() - 80, out.max())
        out = (out - out.min()) / (out.max() - out.min() + eps)
    else:
        out = md
    return out
def vis_micro_doppler(mmw_path, syn_path):
    sys_frames = np.load(syn_path)
    sys_md = compute_md_from_sys(sys_frames)
    method = 'db'  # 'per_matrix', 'db', 'none'
    apply_smoothing = True
    sigma = 1.0
    sys_md_t = transform_md(sys_md, method=method)
    if apply_smoothing:
        sys_md_t = gaussian_filter(sys_md, sigma=(1, sigma))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    vmin = sys_md_t.min()
    vmax = sys_md_t.max()
    im1 = ax.imshow(sys_md_t.T, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)

    ax.set_title('syn mmWave', fontsize=18, fontweight='bold', pad=12)
    ax.set_xlabel('time', fontsize=14, labelpad=8)
    ax.set_ylabel('velocity', fontsize=14, labelpad=8)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    plt.show()


def vis_RA(mmw_path, syn_path, rgb_path):
    sys_frames = np.load(syn_path)
    rgb_files = [f for f in os.listdir(rgb_path) if f.lower().endswith(('.jpg', '.png'))]
    rgb_files = sorted(rgb_files, key=lambda fn: int(os.path.splitext(fn)[0]))
    rgb_files = [os.path.join(rgb_path, f) for f in rgb_files]

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax_syn, ax_img = axes
    ax_syn.set_title('Syn RA')
    ax_img.set_title('RGB image')
    ax_img.axis('off')
    max_len = max(45, len(sys_frames), len(rgb_files))
    for k in range(max_len):
        t0 = time.time()
        RA_syn = None
        if k < len(sys_frames):
            try:
                frame = sys_frames[k].squeeze()
                frame = rangeFFT(frame)
                frame = np.flip(frame)
                frame = rangeCut(frame)
                azimuth_frame = frame.reshape(12, cfg.CHIRP_NUM, -1)[:8]
                azimuth_frame = np.fft.fft(azimuth_frame, 64, axis=0)
                azimuth_frame = np.fft.fftshift(azimuth_frame, axes=0)
                RA_figure = np.sum(abs(azimuth_frame), axis=1)[:, :90]
                RA_syn = RA_figure / np.max(RA_figure, axis=(0, 1))
            except Exception:
                RA_syn = None
        img = None
        if k < len(rgb_files):
            try:
                img = np.array(Image.open(rgb_files[k]).convert('RGB'))
            except Exception:
                img = None
        ax_syn.cla()
        ax_img.cla()
        ax_syn.set_title('Syn RA', fontsize=18, fontweight='bold', pad=12)
        if RA_syn is not None:
            sns.heatmap(RA_syn, ax=ax_syn, cbar=False)
            ax_syn.invert_yaxis()
            ax_syn.set_xlabel('range bin', fontsize=14)
            ax_syn.set_ylabel('azimuth bin', fontsize=14)
            ax_syn.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                           labelbottom=False, labelleft=False)
        else:
            ax_syn.text(0.5, 0.5, 'no synthesized frame', transform=ax_syn.transAxes, ha='center')
        ax_img.set_title('RGB image')
        ax_img.axis('off')
        if img is not None:
            ax_img.imshow(img)
        else:
            ax_img.text(0.5, 0.5, 'no image', transform=ax_img.transAxes, ha='center')
        fig.canvas.draw_idle()
        plt.pause(0.05)
    plt.ioff()
    plt.show()
if __name__ == "__main__":
    # path
    mmw_path = r'../examples/raw_mmWave_human/captured signal/adc.bin'
    syn_path = r'../examples/raw_mmWave_human/captured signal/sys_sig0_1.npy'
    rgb_path = r'../examples/raw_mmWave_human/captured signal/rgb'

    # vis_range_compare(mmw_path, syn_path, rgb_path)
    vis_micro_doppler(mmw_path, syn_path)
    # vis_RA(mmw_path, syn_path, rgb_path)
    # vis_point_cloud(mmw_path, syn_path, rgb_path)



