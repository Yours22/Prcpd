import os
import time
import csv
import subprocess
import glob
import re  # 👈 新增：引入正则表达式模块，用于精准提取数字

EXECUTABLE = "FEMFFUSION/femffusion.exe"  
PRM_DIR = "data-gen/prm_cases"
TIMING_CSV = "data-raw/execution_times.csv"

# =====================================================================

def run_batch():
    # 检查执行文件是否存在
    if not os.path.exists(EXECUTABLE):
        print(f"警告: 找不到可执行文件 {EXECUTABLE}，请确认路径。")

    # 👈 核心修复部分：获取所有 prm 文件，并按文件名中的数字大小进行自然排序
    prm_files = glob.glob(os.path.join(PRM_DIR, "*.prm"))
    # 使用正则表达式 \d+ 提取文件名中的数字，转化为整数后作为排序依据
    prm_files = sorted(prm_files, key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
    
    total_cases = len(prm_files)
    
    if total_cases == 0:
        print(f"错误: 在 {PRM_DIR} 中没有找到任何 .prm 文件。")
        return

    print(f"发现 {total_cases} 个算例，准备开始批量计算...")
    os.makedirs(os.path.dirname(TIMING_CSV), exist_ok=True)

    # 准备写入时间记录表
    with open(TIMING_CSV, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['case_id', 'execution_time_seconds', 'status'])

        for i, prm_path in enumerate(prm_files, 1):
            # 提取文件名作为 case_id (例如: case_200)
            case_id = os.path.splitext(os.path.basename(prm_path))[0]
            
            # 构建终端执行指令
            command = [EXECUTABLE, "-f", prm_path]
            print(f"[{i}/{total_cases}] 正在计算 {case_id} ...", end="", flush=True)

            # 记录开始时间
            start_time = time.time()
            
            try:
                # 执行指令，屏蔽标准输出
                result = subprocess.run(
                    command, 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.PIPE, 
                    text=True,
                    check=True
                )
                
                # 记录结束时间并计算耗时
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                print(f" 完成! 耗时: {elapsed_time:.2f} 秒")
                csv_writer.writerow([case_id, f"{elapsed_time:.3f}", "Success"])
                
            except subprocess.CalledProcessError as e:
                # 捕获运行崩溃
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                print(f" 失败! 耗时: {elapsed_time:.2f} 秒")
                print(f" 报错信息: {e.stderr.strip()}")
            
                csv_writer.writerow([case_id, f"{elapsed_time:.3f}", "Failed"])
                
            # 强制将内存中的数据刷入硬盘
            csvfile.flush() 

    print(f"\n 所有算例计算完毕！运行时间日志已保存至 '{TIMING_CSV}'。")

if __name__ == "__main__":
    run_batch()