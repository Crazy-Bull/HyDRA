import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pickle

# 配置参数
DATA_PATH = ".../SingleDay.pkl"     # 数据集路径
SAMPLE_RATE = 20e6              # 实际采样率(需确认)
N_FFT = 256                     # FFT点数
NUM_PEAKS = 3                   # 提取主频数量
EQ_MODE = 0                     # 均衡模式(0=原始信号)

def load_dataset():
    """加载并解析数据集"""
    with open(DATA_PATH, 'rb') as f:
        dataset = pickle.load(f)
        
    # 获取索引
    rx_idx = dataset['rx_list'].index(RX_NODE)
    date_idx = dataset['capture_date_list'].index('2021_03_23')
    eq_idx = dataset['equalized_list'].index(EQ_MODE)
    
    # 构建标签数据字典
    label_data = {}
    for tx_idx, tx_name in enumerate(dataset['tx_list']):
        # 提取指定接收节点的数据 [tx][rx][date][eq]
        data = dataset['data'][tx_idx][rx_idx][date_idx][eq_idx]
        if data.size > 0:
            label_data[tx_name] = data
            
    return label_data

def compute_main_frequencies(signals, top_k=3):
    """返回：(完整频率轴, 功率谱, 主频列表)"""
    complex_sig = signals[...,0] + 1j*signals[...,1]
    
    # 计算平均功率谱
    spectrum = np.mean([
        np.abs(np.fft.fft(sig * np.hamming(N_FFT))[:N_FFT//2])**2 
        for sig in complex_sig
    ], axis=0)
    
    # 频率轴
    freqs = np.fft.fftfreq(N_FFT, 1/SAMPLE_RATE)[:N_FFT//2]
    
    # 峰值检测
    peaks, _ = find_peaks(spectrum, 
                        height=np.percentile(spectrum, 75),  # 高于75%分位数
                        distance=N_FFT//20)
    if len(peaks) == 0:
        return freqs, spectrum, np.array([])
    
    # 取最高top_k个峰值
    top_peaks = peaks[np.argsort(spectrum[peaks])[-top_k:]]
    
    return freqs, spectrum, freqs[top_peaks]

def main():
    # 1. 加载数据
    label_data = load_dataset()
    print(f"Find out {len(label_data)}valid tx data")
    
    # 2. 主频分析
    main_freqs_dict = {}
    plt.figure(figsize=(14, 8))
    
    for label, signals in label_data.items():
            if len(signals) > 100:
                signals = signals[np.random.choice(len(signals), 100)]
                
            # 获取频谱数据
            freqs, spectrum, main_freqs = compute_main_frequencies(signals, NUM_PEAKS)
            main_freqs_dict[label] = main_freqs
            
            # 绘制频谱曲线
            plt.plot(freqs/1e6, 10*np.log10(spectrum), 
                    alpha=0.6, label=f'TX {label}')
            
            # 标记主频点
            if len(main_freqs) > 0:
                plt.scatter(main_freqs/1e6, 10*np.log10(spectrum[np.isin(freqs, main_freqs)]),
                        marker='*', color='red', zorder=5)

    # 3. 可视化设置
    plt.title(f'RF Spectrum (RX {RX_NODE}, Top {NUM_PEAKS} Peaks)')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    # 4. 输出网络配置参数
    print("\nSuggestion:")
    for label, freqs in main_freqs_dict.items():
        print(f"{label}: {freqs/1e6} MHz")
    
    plt.savefig('spectrum_analysis.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()