import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_reactor_dataset_by_product(file_path, random_seed=42):
    """
    按 Slope_Up * Cut_Time (总扰动量) 的物理边界提取外推集，
    并将划分结果保存至 data-split 
    """
    print(f"正在读取原始数据文件: {file_path} ...")
    df = pd.read_csv(file_path)
    
    df['Perturbation_Total'] = df['slope_up'] * df['cut_time']
    
    # 按照总扰动量排序
    df_sorted = df.sort_values(by='Perturbation_Total')

    # 提取参数空间边缘的数据 (提取两端最极端的各 74 组，共 148 组)
    extrapol_num_per_side = 74
    df_extrapol_bottom = df_sorted.head(extrapol_num_per_side)
    df_extrapol_top = df_sorted.tail(extrapol_num_per_side)
    df_extrapolation = pd.concat([df_extrapol_bottom, df_extrapol_top])

    # ==========================================
    # 2. 提取剩余数据并划分训练/验证集
    # ==========================================
    df_remaining = df.drop(df_extrapolation.index)

    df_extrapolation = df_extrapolation.drop(columns=['Perturbation_Total'])
    df_remaining = df_remaining.drop(columns=['Perturbation_Total'])

    # 划分 150 组验证集，剩下的全部作为训练集
    df_train, df_val = train_test_split(
        df_remaining, 
        test_size=150, 
        random_state=random_seed, 
        shuffle=True
    )

    # ==========================================
    # 3. 创建输出文件夹并保存结果
    # ==========================================
    output_dir = 'data-split'
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'dataset_train.csv')
    val_path = os.path.join(output_dir, 'dataset_val.csv')
    test_path = os.path.join(output_dir, 'dataset_test_extrapolation.csv')
    
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_extrapolation.to_csv(test_path, index=False)
    
    print("\n--- 数据划分与保存完成 ---")
    print(f"训练集 (Training)    : {len(df_train)} 组")
    print(f"验证集 (Validation)  : {len(df_val)} 组")
    print(f"外推测试集 (Extrapol): {len(df_extrapolation)} 组")
    print(f" 文件保存至: ./{output_dir}/ 目录。")

    return df_train, df_val, df_extrapolation

# 执行函数
train_df, val_df, test_df = split_reactor_dataset_by_product('dataset_parameters_cleaned.csv')