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

# 4.3 EMG伪迹检测
print("\n=== 4.3 EMG伪迹分析 ===")
print("正在进行肌电(EMG)伪迹分析...")

# 1. 频域分析：计算高频能量 (30-100Hz)
# EMG在频谱上通常表现为30Hz以上的宽带噪声
sfreq = raw.info['sfreq']
fmax_emg = min(100, sfreq / 2.0 - 1)
print(f"分析频段: 30-{fmax_emg:.1f}Hz")

# 计算高频段PSD
# 注意：compute_psd 返回的是 Spectrum 对象
spectrum_emg = raw_eeg.compute_psd(fmin=30, fmax=fmax_emg)
# 获取数据 (n_channels, n_freqs) 并计算平均功率
emg_power = spectrum_emg.get_data().mean(axis=1)

# 找出受肌电影响最大的通道
max_emg_idx = np.argmax(emg_power)
max_emg_ch = eeg_channel_names[max_emg_idx]
print(f"高频能量最强的通道: {max_emg_ch} (可能受EMG污染最重)")

# 绘制各通道肌电能量水平
fig_emg_power, ax_emg_p = plt.subplots(figsize=(12, 6))
# 使用归一化颜色
norm_power = emg_power / np.max(emg_power)
bars = ax_emg_p.bar(range(len(eeg_channel_names)), emg_power, color=plt.cm.coolwarm(norm_power))
ax_emg_p.set_xticks(range(len(eeg_channel_names)))
short_labels = [ch.split('-')[-1] if '-' in ch else ch for ch in eeg_channel_names]
ax_emg_p.set_xticklabels(short_labels, rotation=45, ha='right')
ax_emg_p.set_title('各EEG通道高频能量分布 (30-100Hz EMG指示)')
ax_emg_p.set_ylabel('平均功率密度')
# 添加颜色条
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=np.max(emg_power)))
sm.set_array([])
plt.colorbar(sm, ax=ax_emg_p, label='能量强度 (W/Hz)')

# 2. 时域分析：检测肌电爆发 (基于大幅度阈值)
eeg_data_all = raw_eeg.get_data()
std_global = np.std(eeg_data_all)
emg_threshold = 4 * std_global  # 4倍标准差
print(f"EMG检测幅度阈值: {emg_threshold:.2f} uV")

# 检测任意通道超过阈值的时刻
emg_mask = np.any(np.abs(eeg_data_all) > emg_threshold, axis=0)
emg_ratio = np.mean(emg_mask)
print(f"检测到的EMG伪迹时间比例: {emg_ratio:.2%}")

# 绘制EMG伪迹时间分布
fig_emg_time, ax_emg_t = plt.subplots(figsize=(12, 4))
# 绘制检测结果
ax_emg_t.fill_between(time_axis, 0, emg_mask.astype(int), color='orange', alpha=0.5, label='EMG Artifact Regions')
# 叠加一个参考EEG通道 (归一化到0-1之间以便展示)
ref_data = eeg_data_all[max_emg_idx] # 使用受影响最大的通道作为参考
ref_data_norm = (ref_data - np.mean(ref_data)) / (np.std(ref_data) * 10) + 0.5 
ax_emg_t.plot(time_axis, ref_data_norm, color='black', alpha=0.3, linewidth=0.5, label=f'EEG Ref ({max_emg_ch})')

ax_emg_t.set_title(f'EMG伪迹时间分布 (检测阈值: >4*STD)')
ax_emg_t.set_xlabel('时间 (s)')
ax_emg_t.set_yticks([0, 1])
ax_emg_t.set_yticklabels(['Low', 'High'])
ax_emg_t.legend(loc='upper right')
ax_emg_t.set_xlim(time_axis[0], time_axis[-1])
ax_emg_t.grid(True, alpha=0.3)

print("\n所有分析完成！请查看弹出的图形窗口。")
plt.tight_layout()
plt.show()