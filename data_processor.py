import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# ====== 配置参数 ======
file_path = 'data/marker_test.txt'
data_start = 0             # 数据起始 index
data_end = 2236           # 数据结束 index
# 滤波和检测参数
gaussian_sigma = 20       # 高斯滤波强度
valley_distance = 50       # 谷底最小间隔，用于初步检测
condense_threshold = 0.3   # 相邻谷底之间距离阈值，小于此则剔除后者
# 距离计算窗口大小
average_window = 50        # 用于计算平均距离的后续点个数
# 动态端点检测参数（按 X, Y, Z 分别设置）
tolerance_x = 1.0         # X 方向误差容忍
tolerance_y = 1.0         # Y 方向误差容忍
tolerance_z = 1.0         # Z 方向误差容忍
end_min_distance = 5000    # endpoint 之间的最小间隔

# ====== 读取数据 ======
def read_data(file_path, start=None, end=None):
    data = pd.read_csv(file_path, delim_whitespace=True)
    if start is not None or end is not None:
        data = data.iloc[start:end]
    return data[['X', 'Y', 'Z']].values

# ====== 计算窗口平均距离 ======
def compute_windowed_average_distances(xyz, window=10):
    N = len(xyz)
    avg_dists = np.zeros(N-1)
    for i in range(N-1):
        end_idx = min(i + window, N - 1)
        ds = [np.linalg.norm(xyz[j] - xyz[i]) for j in range(i+1, end_idx+1)]
        avg_dists[i] = np.mean(ds) if ds else 0
    return avg_dists

# ====== 计算相邻点距离 ======
def compute_adjacent_distances(xyz):
    return np.array([np.linalg.norm(xyz[i+1] - xyz[i]) for i in range(len(xyz)-1)])

# ====== 高斯平滑 ======
def smooth_distances(distances, sigma):
    return gaussian_filter1d(distances, sigma=sigma)

# ====== 谷底检测 ======
def detect_valleys(smoothed_distances, min_distance):
    inverted = -smoothed_distances
    peaks, _ = find_peaks(inverted, distance=min_distance)
    return peaks

# ====== 主流程 ======
xyz = read_data(file_path, data_start, data_end)

# 计算窗口平均距离
distances = compute_windowed_average_distances(xyz, window=average_window)
# 平滑处理
a_smoothed = smooth_distances(distances, gaussian_sigma)
# 初步检测谷底
valleys = detect_valleys(a_smoothed, valley_distance)
valley_frames = (valleys + 1).astype(int)

# 计算谷底 XYZ 与相邻谷底间距
valley_xyz = xyz[valley_frames - 1]
valley_distances = compute_adjacent_distances(valley_xyz)

# 根据阈值剔除过近谷底
condensed_frames = []
if len(valley_frames) > 0:
    condensed_frames.append(valley_frames[0])
    for i in range(1, len(valley_frames)):
        if valley_distances[i-1] >= condense_threshold:
            condensed_frames.append(valley_frames[i])
condensed_frames = np.array(condensed_frames, dtype=int)

# ====== 动态端点检测 ======
prev_endpoint = xyz[0]
dynamic_ends = []
for f in np.sort(condensed_frames):
    pos = xyz[f - 1]
    dx = abs(pos[0] - prev_endpoint[0])
    dy = abs(pos[1] - prev_endpoint[1])
    dz = abs(pos[2] - prev_endpoint[2])
    if dx < tolerance_x and dy < tolerance_y and dz < tolerance_z:
        dynamic_ends.append(int(f))
        prev_endpoint = pos

# ====== endpoint 最小间隔过滤 ======
valid_ends = []
last_end = -np.inf
for e in dynamic_ends:
    if e - last_end >= end_min_distance:
        valid_ends.append(e)
        last_end = e
valid_ends = np.array(valid_ends, dtype=int)

# ====== 按区间划分 trajectories ======
trajectories = []
sorted_ends = np.sort(valid_ends)
prev_end = 0
for end in sorted_ends:
    seg = [f for f in condensed_frames if prev_end < f <= end]
    if seg and seg[-1] != end:
        seg.append(end)
    trajectories.append(seg)
    prev_end = end

# ====== 导出特定长度的轨迹点到文件（不包括终点 XYZ） ======
selected = []
count_52 = 0
for idx, seg in enumerate(trajectories, start=1):
    if len(seg) == 52:
        count_52 += 1
        # 排除最后一个端点帧
        for f in seg[:-1]:
            x, y, z = xyz[f - 1]
            selected.append({'traj': idx, 'X': x, 'Y': y, 'Z': z})

# 如果没有任何52-step轨迹并且也没有endpoint被检测到，则把所有filtered valleys当成一个轨迹
if count_52 == 0 and len(valid_ends) == 0:
    print("No 52-step trajectories or endpoints detected → saving all filtered valleys as one trajectory.")
    selected = []
    for f in condensed_frames:
        x, y, z = xyz[f - 1]
        selected.append({'traj': 1, 'X': x, 'Y': y, 'Z': z})

# 写入 CSV
if selected:
    df_clean = pd.DataFrame(selected, columns=['traj', 'X', 'Y', 'Z'])
    df_clean.to_csv('data/cleaned_xyz_test.csv', index=False)
# 打印统计
print(f"Number of trajectories with 52 steps: {count_52}")

# ====== 绘制距离与谷底 ======
plt.figure(figsize=(12, 6))
raw_plot = np.full(len(xyz), np.nan)
raw_plot[1:] = distances
smooth_plot = np.full(len(xyz), np.nan)
smooth_plot[1:] = a_smoothed
plt.plot(np.arange(1, len(xyz)+1), raw_plot, 'o-', alpha=0.2, label='Raw Distance')
plt.plot(np.arange(1, len(xyz)+1), smooth_plot, 'r-', label='Smoothed Distance')
plt.plot(valley_frames, a_smoothed[valleys], 'bv', label='All Valleys')
plt.plot(condensed_frames, a_smoothed[condensed_frames-1], 'cx', label='Condensed Valleys')
plt.plot(sorted_ends, a_smoothed[sorted_ends-1], 'go', label='Dynamic Ends')
plt.xlabel('Frame Index')
plt.ylabel('Distance')
plt.title('Valleys and Trajectory Segments')
plt.legend()
plt.grid(True)
plt.show()

# ====== 绘制谷底间距 ======
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, len(valley_distances)+1), valley_distances, 'm.-', label='Valley Distances')
plt.xlabel('Valley Index')
plt.ylabel('Distance between Consecutive Valleys')
y_max = valley_distances.max()
plt.yticks(np.arange(0, y_max + 0.5, 0.5))
plt.title('Distances Between Consecutive Valley Points')
plt.grid(True)
plt.legend()
plt.show()

# ====== 输出统计 ======
print(f"初始检测到谷底数: {len(valleys)}")
print(f"剔除后谷底数: {len(condensed_frames)}")
print(f"动态检测的结束帧数: {len(sorted_ends)}")
for i, seg in enumerate(trajectories, start=1):
    if not seg:
        continue
    print(f"Trajectory {i}: steps={len(seg)}, frames {seg[0]}–{seg[-1]}")
