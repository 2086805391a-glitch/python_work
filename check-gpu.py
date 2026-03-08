import torch

# 1. 基础检查
print(f"PyTorch 版本: {torch.__version__}")

# 2. 核心检查：CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA (GPU 加速) 是否可用: {cuda_available}")

if cuda_available:
    # 获取显卡信息
    device_name = torch.cuda.get_device_name(0)
    print(f"检测到的显卡设备: {device_name}")
    
    # 做一个简单的矩阵运算测试
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("GPU 矩阵运算测试成功！")
else:
    print("目前只能使用 CPU 运行。如果你有 NVIDIA 显卡但显示不可用，可能需要安装 CUDA Toolkit 或更新驱动。")