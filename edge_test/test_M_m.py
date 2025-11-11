import torch
import torch.nn as nn
import time
from thop import profile  # 用于计算 FLOPs
import pickle
from HyDRA_M_m import ResConv1d, CNNFeatureExtractor, MambaBlock, MambaEncoder, CNNMambaModel, ModelArgs

# 如果你已经有一个单独的 .py 文件包含这些类，可以改为：
# from your_model_file import CNNTransformerModel

def analyze_model(model_path, input_channels=2, num_classes=10, batch_size=1, seq_length=256):
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    model = model.to(device)
    model.eval()

    # 创建全一输入张量
    input_tensor = torch.ones(batch_size, seq_length, input_channels).to(device)
    print(f"Input tensor shape: {input_tensor.shape}")

    # 1. 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Total parameters (M): {total_params / 1e6:.2f}M")

    # 2. 计算 FLOPs
    try:
        flops = profile(model, inputs=(input_tensor,), verbose=False)[0]
        print(f"Total FLOPs: {flops:,}")
        print(f"Total FLOPs (G): {flops / 1e9:.2f}G")
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")

    # 3. 测试推理时间
    num_runs = 1000
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # 实际计时
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            output = model(input_tensor)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs * 1000  # 转换为毫秒
    print(f"Average inference time: {avg_time:.4f} ms")
    print(f"Throughput: {1000 / avg_time:.2f} samples/second")

    # 检查输出形状
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    # 设置参数（根据你的模型调整这些值）
    model_path = r"/home/qiu/桌面/RF-fingerprint-classification/edge_test/model_lossless_iq_3_SingleDay_M-m.pkl"
    input_channels = 6  # 根据你的模型输入通道数调整
    num_classes = 28    # 根据你的分类数调整
    batch_size = 1
    seq_length = 256    # 输入序列长度

    # 运行分析
    analyze_model(
        model_path=model_path,
        input_channels=input_channels,
        num_classes=num_classes,
        batch_size=batch_size,
        seq_length=seq_length
    )
