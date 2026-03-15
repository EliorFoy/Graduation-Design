import mne
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path



def get_raw_data(subject='A01T') -> mne.io.Raw:
    """
    加载指定被试的原始数据
    
    Args:
        subject: 被试标识，如 'A01T'（训练集）或 'A01E'（测试集）
    
    Returns:
        raw: MNE Raw 对象
    """
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    print("加载数据集")
    data_path = project_root / "BCICIV_2a_gdf" / f"{subject}.gdf"
    print(f"数据文件路径：{data_path}")
    
    # 检查文件是否存在
    if not data_path.exists():
        print(f"❌ 错误：数据文件不存在！")
        print(f"   期望路径：{data_path}")
        raise FileNotFoundError(f"数据文件不存在：{data_path}")
    
    raw = mne.io.read_raw_gdf(str(data_path), preload=True)
    return raw

def modify_channel_name_and_type(raw)->mne.io.Raw:
    print("\n原始通道名称:")
    print(raw.ch_names)

    # 1. 重命名通道为标准名称
    # BCIC IV-2a 数据集的标准 22 个 EEG 通道
    standard_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 
        'P1', 'Pz', 'P2', 'POz'
    ]

    # 获取 EEG 通道（排除 EOG）
    eog_indices = [i for i, name in enumerate(raw.ch_names) if 'EOG' in name]
    eeg_indices = [i for i, name in enumerate(raw.ch_names) if 'EOG' not in name]

    print(f"\nEEG 通道数：{len(eeg_indices)}")
    print(f"EOG 通道数：{len(eog_indices)}")

    # 创建重命名映射
    rename_map = {}
    for i, idx in enumerate(eeg_indices):
        if i < len(standard_names):
            rename_map[raw.ch_names[idx]] = standard_names[i]

    print(f"\n重命名映射：{rename_map}")
    raw.rename_channels(rename_map)

    print("\n重命名后的通道名称:")
    print(raw.ch_names)

    # 2. 设置通道类型
    # 创建通道类型映射字典
    ch_types_map = {}
    for ch_name in raw.ch_names:
        if 'EOG' in ch_name:
            ch_types_map[ch_name] = 'eog'
        else:
            ch_types_map[ch_name] = 'eeg'

    print(f"\n设置通道类型：{ch_types_map}")
    raw.set_channel_types(ch_types_map)

    return raw

def set_electrode_and_show(raw):
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 3. 根据电极设置国际通用 10-20 电极位置（基本匹配可能略有差异）
    print("\n设置标准 10-20 电极位置...")
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')

    print("电极位置设置完成！")
    print(f"可用通道：{raw.ch_names}")

    # 4. 可视化电极位置
    raw.plot_sensors(show_names=True, title='电极位置分布', block=False)
    plt.show()
    return raw

def get_modified_raw_data(subject='A01T') -> mne.io.Raw:
    """
    加载并处理指定被试的原始数据
    
    Args:
        subject: 被试标识，如 'A01T'（训练集）或 'A01E'（测试集）
    
    Returns:
        modified_raw: 处理后的 MNE Raw 对象
    """
    raw = get_raw_data(subject)
    modified_raw = modify_channel_name_and_type(raw)
    return modified_raw

if __name__ == "__main__":
    modified_raw = get_modified_raw_data()
    set_electrode_and_show(modified_raw)
