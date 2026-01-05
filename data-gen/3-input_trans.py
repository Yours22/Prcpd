import os
import numpy as np
import re

def parse_single_xsec(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # 1. 定位数据块：找到 XSecs 和 Precursors 之间的内容
    # 使用正则表达式提取中间部分，re.DOTALL 让 . 匹配换行符
    match = re.search(r'XSecs(.*?)(?:Precursors|$)', content, re.DOTALL)
    if not match:
        raise ValueError(f"无法在 {filepath} 中找到 XSecs 数据块")
    
    block_content = match.group(1)
    
    # 2. 清理注释：去除 # 及其后面的内容
    block_content = re.sub(r'#.*', '', block_content)
    
    # 3. 提取所有数字 token
    tokens = block_content.split()
    
    # 4. 解析数值
    # 第一个 token 是材料序号
    try:
        num_materials = int(tokens[0])
    except ValueError:
        raise ValueError(f"文件格式错误: {filepath} XSecs 后应该跟材料数量")

    data_tokens = tokens[1:]
    
    # 校验数据长度：每个材料应该是 1 个 ID + 5 个快群 + 4 个热群 = 10 个数值
    values_per_material = 10 
    expected_len = num_materials * values_per_material
    
    if len(data_tokens) != expected_len:
        print(f"警告: {filepath} 数据长度 {len(data_tokens)} 不符合预期 {expected_len}，可能解析有误")
    
    features = []
    
    # 5. 循环提取
    for i in range(num_materials):
        start_idx = i * values_per_material
        end_idx = start_idx + values_per_material
        
        mat_chunk = data_tokens[start_idx : end_idx]
        
        mat_id = int(float(mat_chunk[0])) 
        xs_values = [float(x) for x in mat_chunk[1:]]
        
        features.extend(xs_values)
        
    return np.array(features)

def build_input_dataset(input_dir, num_samples):
    X_list = []
    print("开始处理 xsec 文件...")
    
    for i in range(num_samples):
        filename = os.path.join(input_dir, f"sample_{i}.xsec")
        
        if not os.path.exists(filename):
            print(f"文件缺失: {filename}")
            continue
            
        try:
            sample_features = parse_single_xsec(filename)
            X_list.append(sample_features)
        except Exception as e:
            print(f"处理 {filename} 失败: {e}")
            
    X_matrix = np.array(X_list)
    print(f"处理完成。矩阵形状: {X_matrix.shape}")
    return X_matrix

if __name__ == "__main__":
    INPUT_DIR = './dataset_raw/inputs' 
    NUM_SAMPLES = 200
    
    X = build_input_dataset(INPUT_DIR, NUM_SAMPLES)
    
    np.save('./data-gen/inputs.npy', X)
    print("inputs.npy 已保存。")