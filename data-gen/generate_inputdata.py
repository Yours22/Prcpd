import os
import random
import re

TEMPLATE_PRM = "./data-gen/2D_TWIGL/twigl_diff_ramp_quarter.prm"
TEMPLATE_XSEC = "./data-gen/2D_TWIGL/twigl_quarter.xsec"

NUM_SAMPLES = 200

# 扰动幅度
PERTURBATION_LEVEL = 0.005

# 输出目录
OUTPUT_DIR = "dataset_raw"

def perturb_value(match):
    """ 回调函数：随机扰动 """
    val_str = match.group(0)
    original_value = float(val_str)
    factor = 1.0 + random.uniform(-PERTURBATION_LEVEL, PERTURBATION_LEVEL)
    new_value = original_value * factor
    return f"{new_value:.6e}"

def generate_perturbed_xsec(content):
    """ 智能解析函数：只扰动 XSecs 部分 """
    lines = content.splitlines()
    new_lines = []
    in_xsecs_block = False
    
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.lower().startswith("xsecs"):
            in_xsecs_block = True
            new_lines.append(line)
            continue
        if stripped_line.startswith("Precursors") or stripped_line.startswith("Velocity"):
            in_xsecs_block = False
            
        if in_xsecs_block:
            new_line = re.sub(r'\d+\.\d+([eE][+-]?\d+)?', perturb_value, line)
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    return "\n".join(new_lines)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建文件夹: {OUTPUT_DIR}")

    print(f"正在生成 {NUM_SAMPLES} 组输入文件...")

    with open(TEMPLATE_XSEC, 'r') as f:
        xsec_content_raw = f.read()
    with open(TEMPLATE_PRM, 'r') as f:
        prm_content_raw = f.read()

    for i in range(NUM_SAMPLES):
        sample_id = f"sample_{i}"
        
        # 1. 保存截面文件
        new_xsec_content = generate_perturbed_xsec(xsec_content_raw)
        xsec_filename = f"{sample_id}.xsec"
        with open(os.path.join(OUTPUT_DIR, xsec_filename), 'w') as f:
            f.write(new_xsec_content)

        # 2. 修改并保存输入卡
        # 注意：因为我们要在根目录运行程序，所以这里写入文件内部的路径要带上 "dataset_raw/"
        rel_xsec_path = f"{OUTPUT_DIR}/{xsec_filename}"
        out_filename = f"{OUTPUT_DIR}/outputs/{sample_id}.out"

        new_prm_content = re.sub(
            r'set\s+XSECS_Filename\s*=.*', 
            f'set XSECS_Filename = {rel_xsec_path}', 
            prm_content_raw,
            flags=re.IGNORECASE
        )
        new_prm_content = re.sub(
            r'set\s+Output_Filename\s*=.*', 
            f'set Output_Filename = {out_filename}', 
            new_prm_content,
            flags=re.IGNORECASE
        )

        prm_filename = f"{sample_id}.prm"
        with open(os.path.join(OUTPUT_DIR, prm_filename), 'w') as f:
            f.write(new_prm_content)

    print(f"生成完成！请打开 {OUTPUT_DIR} 文件夹检查一下文件内容是否正确。")

if __name__ == "__main__":
    main()