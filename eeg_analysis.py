import mne
import numpy as np 
import matplotlib.pyplot as plt 

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def read_eeg_data(file_path= ".\BCICIV_2a_gdf\A01T.gdf"):
    raw = mne.io.read_raw_gdf(file_path, preload=True) # 文件不大，直接加载，并方便后续滤波
    print("=====数据集加载完成！=====")
    print("=====数据集信息：=====")
    print(raw.info)
    return raw

def channel_classification(raw):
    print("=====由于数据集只提供了通道名，这里进行分类=====")
    eeg_channel_names = [ch for ch in raw.info['ch_names'] if 'EOG' not in ch]
    eog_channel_names = [ch for ch in raw.info['ch_names'] if 'EOG' in ch]
    print("通道类型分类完成：")
    print(f"EEG通道: {eeg_channel_names}")
    print(f"EOG通道: {eog_channel_names}")

if __name__ == "__main__":
    raw = read_eeg_data()
    channel_classification(raw)
