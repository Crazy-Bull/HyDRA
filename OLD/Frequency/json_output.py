import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from scipy.fft import fftfreq
import json

def load_data(path):
    """加载SingleDay数据集
    输入: 
        path - pickle文件路径
    输出: 
        dict包含:
        - tx_list: 发射设备列表
        - rx_list: 接收设备列表
        - data: 三维数据矩阵[tx][rx][date][sample]
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def compute_psd(signal, fs=25e6):
    """计算功率谱密度
    输入:
        signal - 时域IQ信号(复数数组)
        fs - 采样率(默认25MHz)
    输出:
        f - 完整频率数组(-12.5MHz ~ +12.5MHz)
        Pxx - 双边功率谱密度
    """
    f, Pxx = periodogram(signal, fs=fs, nfft=256, return_onesided=False)
    return f, Pxx

def analyze_frequencies(dataset, fs=25e6):
    """频谱分析
    输入:
        dataset - 加载的原始数据集
        fs - 采样率
    输出:
        full_freqs - 完整频率数组(256点)
        device_psds - 各设备组的平均PSD列表
        device_info_list - 设备信息字典列表
    """
    # 生成完整频率轴
    full_freqs = fftfreq(256, 1/fs)
    
    device_psds = []
    device_info_list = []
    
    for tx_idx in range(len(dataset['tx_list'])):
        for rx_idx in range(len(dataset['rx_list'])):
            date_data = dataset['data'][tx_idx][rx_idx][0][0]
            if date_data.size == 0:
                continue

            psd_list = []
            for sample in date_data:
                iq_signal = sample[:, 0] + 1j * sample[:, 1]
                _, psd = compute_psd(iq_signal, fs)
                psd_list.append(psd)  # 保留完整256点
            
            avg_psd = np.mean(psd_list, axis=0)
            device_psds.append(avg_psd)
            
            device_info_list.append({
                "tx": dataset['tx_list'][tx_idx],
                "rx": dataset['rx_list'][rx_idx],
                "psd": avg_psd.tolist()  # 完整256点PSD
            })
    
    return full_freqs, device_psds, device_info_list

def export_full_spectrum_json(freqs, device_info_list, output_path):
    """导出谱数据到JSON
    输入:
        freqs - 完整频率数组
        device_info_list - 设备信息列表
        output_path - 输出文件路径
    生成JSON结构:
    {
        "frequencies": [-12500000.0, ..., +12500000.0], 
        "devices": [
            {
                "tx": "设备A",
                "rx": "设备B",
                "psd": [1.2e-15, ..., 3.4e-15]  # 256个点
            },
            // 其他设备对...
        ]
    }
    """
    output_data = {
        "frequencies": np.fft.fftshift(freqs).tolist(),  # 将零频移至中心
        "devices": [
            {
                "tx": dev["tx"],
                "rx": dev["rx"],
                "psd": np.fft.fftshift(dev["psd"]).tolist()  # PSD对齐频率
            } for dev in device_info_list
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    dataset = load_data(".../SingleDay.pkl")
    freqs, psd_list, device_info = analyze_frequencies(dataset)
    export_full_spectrum_json(freqs, device_info, "full_spectrum.json")