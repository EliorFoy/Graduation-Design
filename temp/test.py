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

# 1. 基本信息浏览
print("\n=== 1. 基本信息 ===")
print(raw.info)

# 2. 通道信息
print("\n=== 2. 通道信息 ===")
print(f"通道数量: {raw.info['nchan']}")
print(f"通道名称: {raw.info['ch_names'][:10]}...")  # 显示前10个通道

# 3. 事件标记
print("\n=== 3. 事件标记 ===")
events, event_id = mne.events_from_annotations(raw)
print(f"事件数量: {len(events)}")
print(f"事件类型: {event_id}")

# 4. 基本统计信息
print("\n=== 4. 基本统计信息 ===")
data, times = raw[:]
print(f"数据形状: {data.shape} (通道数, 时间点)")
print(f"时间范围: {times[0]:.2f}s - {times[-1]:.2f}s")
print(f"数据均值: {np.mean(data):.4f}")
print(f"数据标准差: {np.std(data):.4f}")
print(f"数据最小值: {np.min(data):.4f}")
print(f"数据最大值: {np.max(data):.4f}")

# 5. 数据可视化
print("\n=== 5. 数据可视化 ===")

# 5.1 绘制原始EEG信号
print("绘制原始EEG信号...")
raw.plot(duration=10, n_channels=8, scalings='auto', title='原始EEG信号（前8个通道，10秒数据）')

# 5.2 绘制功率谱密度(PSD)
print("绘制功率谱密度...")
# 使用新的API方式
psd = raw.compute_psd(fmax=50, average='mean')
psd.plot()
plt.title('EEG信号功率谱密度（0-50Hz）')

# 5.3 绘制事件标记
print("绘制事件标记...")
fig3, ax = plt.subplots(figsize=(12, 4))
ax.plot(events[:, 0] / raw.info['sfreq'], events[:, 2], 'o', markersize=4)
ax.set_xlabel('时间 (s)')
ax.set_ylabel('事件类型')
ax.set_title('事件标记时间分布')
ax.grid(True, alpha=0.3)

# 5.4 提取并绘制事件相关电位(ERP)
print("提取并绘制事件相关电位...")
# 选择一个关键通道（如C3）
channel_index = raw.info['ch_names'].index('EEG-C3') if 'EEG-C3' in raw.info['ch_names'] else 0

# 为每种事件类型提取ERP
plt.figure(figsize=(12, 6))
for event_name, event_code in event_id.items():
    # 找到该事件类型的所有事件
    event_indices = np.where(events[:, 2] == event_code)[0]
    if len(event_indices) > 0:
        # 提取事件前后的信号
        epochs = []
        for event_idx in event_indices[:10]:  # 只使用前10个事件以提高速度
            start = events[event_idx, 0] - int(0.5 * raw.info['sfreq'])  # 事件前0.5秒
            end = events[event_idx, 0] + int(2.0 * raw.info['sfreq'])    # 事件后2秒
            if start >= 0 and end < len(raw.times):
                epoch_data = raw[channel_index, start:end][0][0]
                epochs.append(epoch_data)
        
        if epochs:
            epochs = np.array(epochs)
            erp = np.mean(epochs, axis=0)
            time_points = np.linspace(-0.5, 2.0, len(erp))
            plt.plot(time_points, erp, label=event_name)

plt.xlabel('时间 (s)')
plt.ylabel('电位 (μV)')
plt.title(f'事件相关电位 (ERP) - 通道: {raw.info["ch_names"][channel_index]}')
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='事件发生')
plt.legend()
plt.grid(True, alpha=0.3)

print("所有可视化完成！请查看弹出的图形窗口。")
plt.tight_layout()
plt.show()
