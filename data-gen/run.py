import os
import subprocess
import time

FEMFFUSION_EXE = os.path.abspath("./FEMFFUSION/femffusion.exe")

# 数据文件夹
DATA_DIR = "dataset_raw"
# ===========================================

def run_simulations():
    all_files = os.listdir(DATA_DIR)
    prm_files = [f for f in all_files if f.endswith(".prm")]
    
    prm_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    total = len(prm_files)
    print(f"在 {DATA_DIR} 中找到了 {total} 个任务，准备开始计算...")

    for index, prm_file in enumerate(prm_files):
        # 构造完整路径：dataset_raw/sample_0.prm
        prm_path = os.path.join(DATA_DIR, prm_file)
        
        print(f"[{index+1}/{total}] 正在运行: {prm_file} ...", end="", flush=True)
        
        start_time = time.time()
        try:
            subprocess.run(
                [FEMFFUSION_EXE, "-f", prm_path],
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.PIPE,    
                check=True
            )
            elapsed = time.time() - start_time
            print(f" 完成 ({elapsed:.2f}s)")
            
        except subprocess.CalledProcessError as e:
            print("\n!!! 运行失败 !!!")
            print(e.stderr.decode('utf-8', errors='ignore'))
        except Exception as e:
            print(f"\n发生未知错误: {e}")

    print("\n所有计算结束。")

if __name__ == "__main__":
    run_simulations()