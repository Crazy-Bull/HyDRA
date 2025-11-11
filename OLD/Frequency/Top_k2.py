import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from scipy.fft import fftfreq

def load_data(path):
    """加载SingleDay数据集"""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def compute_psd(signal, fs=25e6):
    """计算功率谱密度"""
    f, Pxx = periodogram(signal, fs=fs, nfft=256, return_onesided=False)
    return f, Pxx

def analyze_topk_frequencies(dataset, k=5, fs=25e6):
    """修正后的Top-K分析"""
    # 计算频率轴并创建掩码
    freqs = fftfreq(256, 1/fs)
    pos_mask = freqs >= 0
    pos_freqs = freqs[pos_mask]  # 维度 (128,)
    
    # 收集设备PSD（仅正频率）
    device_psds = []
    all_psd_pos = []
    
    for tx_idx in range(len(dataset['tx_list'])):
        for rx_idx in range(len(dataset['rx_list'])):
            date_data = dataset['data'][tx_idx][rx_idx][0][0]
            if date_data.size == 0:
                continue
                
            # 计算设备PSD（正频率）
            psd_list = []
            for sample in date_data:
                iq_signal = sample[:, 0] + 1j * sample[:, 1]
                _, psd = compute_psd(iq_signal, fs)
                psd_pos = psd[pos_mask]  # 关键修正点
                psd_list.append(psd_pos)
            
            avg_psd = np.mean(psd_list, axis=0)  # 维度 (128,)
            device_psds.append(avg_psd)
            all_psd_pos.append(avg_psd)
    
    # 计算全局平均PSD
    global_psd = np.mean(all_psd_pos, axis=0)  # 维度 (128,)
    
    # 计算Top-K索引
    topk_indices = []
    for psd in device_psds:
        deviation = np.sqrt(np.abs(psd - global_psd))
        topk_idx = np.argsort(deviation)[-k:][::-1]
        topk_indices.append(topk_idx)
    
    return pos_freqs, device_psds, topk_indices
    
    # 计算各设备Top-K
    topk_indices = []
    for psd in device_psds:
        psd_pos = psd[pos_mask]
        deviation = np.sqrt(np.abs(psd_pos - global_psd))
        topk_idx = np.argsort(deviation)[-k:][::-1]
        topk_indices.append(topk_idx)
    
    return pos_freqs, device_psds, topk_indices

def plot_combined_spectrum(freqs, psd_list, topk_indices, max_devices=28):
    """综合频谱对比图（修正版）"""
    plt.figure(figsize=(16, 8))
    ax = plt.gca()  # 获取当前Axes
    
    # 创建颜色映射
    n_devices = min(len(psd_list), max_devices)
    colors = plt.cm.jet(np.linspace(0, 1, n_devices))
    
    # 绘制频谱曲线和标记点
    for i in range(n_devices):
        psd = psd_list[i]
        indices = topk_indices[i]
        
        # 绘制频谱曲线
        line = ax.semilogy(freqs/1e6, psd, 
                          color=colors[i], 
                          alpha=0.6,
                          label=f'Dev{i+1}' if i < 15 else "")[0]
        
        # 标记Top-K点
        scatter = ax.scatter(freqs[indices]/1e6, psd[indices],
                            color=colors[i],
                            s=50, 
                            edgecolor='k',
                            alpha=0.8)
    
    # 创建颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, 
                             norm=plt.Normalize(vmin=1, vmax=n_devices))
    sm.set_array([])  # 必须设置空数组
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Device Index', rotation=270, labelpad=15)
    cbar.set_ticks(np.linspace(1, n_devices, 5))
    
    # 设置坐标轴
    ax.set_xlim(0, 12.5)
    ax.set_ylim(1e-15, 1e-5)  # 根据实际数据调整
    ax.set_title(f'Combined Spectrum Analysis (First {n_devices} Devices)')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Power Spectral Density (dB/Hz)')
    ax.grid(True, which='both', alpha=0.4)
    
    # 添加图例
    if n_devices <= 15:
        ax.legend(ncol=3, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    dataset = load_data("E:/真正的桌面/学习/大学内容/enjoy/G1/Week5-6/SingleDay.pkl/SingleDay.pkl")
    freqs, psd_list, topk_idx = analyze_topk_frequencies(dataset, k=5)
    plot_combined_spectrum(freqs, psd_list, topk_idx)