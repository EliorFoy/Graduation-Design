import mne
import numpy as np
import matplotlib.pyplot as plt

# 设置Matplotlib支持中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 读取 GDF 文件
print("正在加载数据集...")
raw = mne.io.read_raw_gdf(".\BCICIV_2a_gdf\A01T.gdf", preload=True)
print("数据集加载完成！")

# 1. 通道类型自动分类
print("\n=== 1. 通道类型自动分类 ===")
# 基于通道名称手动选择通道
eeg_channel_names = [ch for ch in raw.info['ch_names'] if 'EOG' not in ch]
eog_channel_names = [ch for ch in raw.info['ch_names'] if 'EOG' in ch]
print("通道类型分类完成：")
print(f"EEG通道: {eeg_channel_names}")
print(f"EOG通道: {eog_channel_names}")

# 2. 通道类型统计
print("\n=== 2. 通道类型统计 ===")
print(f"EEG通道数: {len(eeg_channel_names)}")
print(f"EOG通道数: {len(eog_channel_names)}")

# 3. 分别可视化分析
print("\n=== 3. 分别可视化分析 ===")

# 3.1 原始EEG信号可视化
print("绘制原始EEG信号...")
raw_eeg = raw.copy().pick(picks=eeg_channel_names)
fig_eeg = raw_eeg.plot(duration=10, n_channels=8, scalings='auto', title='原始EEG信号')

# 3.2 原始EOG信号可视化
print("绘制原始EOG信号...")
raw_eog = raw.copy().pick(picks=eog_channel_names)
fig_eog = raw_eog.plot(duration=10, scalings='auto', title='原始EOG信号')

# 3.3 EEG功率谱密度
print("绘制EEG功率谱密度...")
psd_eeg = raw_eeg.compute_psd(fmax=50, average='mean')
psd_eeg.plot()
plt.title('EEG信号功率谱密度（0-50Hz）')

# 3.4 EOG功率谱密度
print("绘制EOG功率谱密度...")
psd_eog = raw_eog.compute_psd(fmax=50, average='mean')
psd_eog.plot()
plt.title('EOG信号功率谱密度（0-50Hz）')

# 3.5 通道相关性分析
print("分析通道相关性...")
# 提取EEG数据
eeg_data = raw_eeg.get_data()
# 计算EEG通道间相关性
corr_matrix = np.corrcoef(eeg_data)

# 绘制相关性矩阵热力图
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(np.arange(len(eeg_channel_names)))
ax.set_yticks(np.arange(len(eeg_channel_names)))
ax.set_xticklabels([ch.split('-')[-1] if '-' in ch else ch for ch in eeg_channel_names], rotation=45, ha='right')
ax.set_yticklabels([ch.split('-')[-1] if '-' in ch else ch for ch in eeg_channel_names])
ax.set_title('EEG通道间相关性矩阵')
plt.colorbar(im, ax=ax, label='相关性系数')
plt.tight_layout()

# 4. 伪迹分析
print("\n=== 4. 伪迹分析 ===")

# 4.1 EOG伪迹检测
print("检测EOG伪迹...")
eog_data = raw_eog.get_data()
threshold = 3 * np.std(eog_data)
eog_artifacts = np.any(np.abs(eog_data) > threshold, axis=0)
artifact_ratio = np.sum(eog_artifacts) / len(eog_artifacts)
print(f"EOG伪迹比例: {artifact_ratio:.2%}")

# 4.2 伪迹时间分布
print("绘制伪迹时间分布...")
time_axis = np.arange(len(eog_artifacts)) / raw.info['sfreq']
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(time_axis, eog_artifacts.astype(int), 'r-', alpha=0.6)
ax.set_xlabel('时间 (s)')
ax.set_ylabel('伪迹检测')
ax.set_title('EOG伪迹时间分布')
ax.set_ylim(-0.1, 1.1)
ax.set_yticks([0, 1])
ax.set_yticklabels(['无伪迹', '有伪迹'])
ax.grid(True, alpha=0.3)

print("\n所有分析完成！请查看弹出的图形窗口。")
plt.tight_layout()
plt.show()