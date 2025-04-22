import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# ====== 配置参数 ======
file_path = 'marker_1__.txt'
data_start = 0             # 数据起始 index
data_end = 250000         # 数据结束 index
# 滤波和检测参数
gaussian_sigma = 19       # 高斯滤波强度
valley_distance = 55       # 谷底最小间隔，用于初步检测
condense_threshold = 0.3   # 相邻谷底之间距离阈值，小于此则剔除后者
known_end_position = np.array([65.0, 62.7, 131.5])  # 已知 trajectory 结束位置
position_tolerance = 1.5   # 结束位置误差容忍
end_min_distance = 5000    # endpoint 之间的最小间隔

# ====== 读取数据 ======
def read_data(file_path, start=None, end=None):
    data = pd.read_csv(file_path, delim_whitespace=True)
    if start is not None or end is not None:
        data = data.iloc[start:end]
    return data[['X', 'Y', 'Z']].values

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
# 计算相邻点距离
distances = compute_adjacent_distances(xyz)
# 平滑处理
a_smoothed = smooth_distances(distances, gaussian_sigma)
# 初步检测谷底
valleys = detect_valleys(a_smoothed, valley_distance)
# 将谷底索引映射到数据帧号
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

# 筛选确认的结束帧候选
end_positions = xyz[condensed_frames - 1]
dists_to_known = np.linalg.norm(end_positions - known_end_position, axis=1)
candidates = condensed_frames[dists_to_known < position_tolerance]

# endpoint 最小间隔过滤
valid_ends = []
last_end = -np.inf
for e in np.sort(candidates):
    if e - last_end >= end_min_distance:
        valid_ends.append(int(e))
        last_end = e
valid_ends = np.array(valid_ends, dtype=int)

# 按区间划分 trajectories
trajectories = []
sorted_ends = np.sort(valid_ends)
prev_end = 0
for end in sorted_ends:
    seg = [f for f in condensed_frames if prev_end < f <= end]
    if seg and seg[-1] != end:
        seg.append(end)
    trajectories.append(seg)
    prev_end = end

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
plt.plot(sorted_ends, a_smoothed[sorted_ends-1], 'go', label='Confirmed Ends')
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
# 保持完整范围，设置 y 轴刻度间隔为 0.5
y_max = valley_distances.max()
plt.yticks(np.arange(0, y_max + 0.5, 0.5))
plt.title('Distances Between Consecutive Valley Points')
plt.grid(True)
plt.legend()
plt.show()

# ====== 输出统计 ======
print(f"初始检测到谷底数: {len(valleys)}")
print(f"剔除后谷底数: {len(condensed_frames)}")
print(f"确认的结束帧数: {len(sorted_ends)}")
for i, seg in enumerate(trajectories, start=1):
    if not seg:
        continue
    print(f"Trajectory {i}: steps={len(seg)}, frames {seg[0]}–{seg[-1]}")
