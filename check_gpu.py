import torch
import sys

def check_environment():
    print("="*30)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print("="*30)
    if torch.cuda.is_available():
        print("CUDA is available! (GPU 准备就绪)")
        
        # 检查 PyTorch 内置的 CUDA 版本，不需要和系统的 CUDA 完全一致，只要低于或等于系统驱动支持的版本即可
        print(f"   PyTorch compiled with CUDA: {torch.version.cuda}")
        
        device_count = torch.cuda.device_count()
        print(f"   GPU Count: {device_count}")
        for i in range(device_count):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        
        try:
            x = torch.tensor([1.0, 2.0]).cuda()
            print("\nTensor operation on GPU successful.")
        except Exception as e:
            print(f"\n❌ Error: GPU detected but operation failed. \n{e}")
            
    else:
        print("❌ CUDA is NOT available. Using CPU.")
        print("   可能的解决方案: 使用 uv add torch --index-url https://download.pytorch.org/whl/cu124 重新安装")

if __name__ == "__main__":
    check_environment()