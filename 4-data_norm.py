import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib
import os

TRAIN_PATH = 'data-split/dataset_train.csv'
VAL_PATH = 'data-split/dataset_val.csv'
TEST_PATH = 'data-split/dataset_test_extrapolation.csv'

def process_and_normalize_features(train_path, val_path, test_path, output_dir='data-split'):
    """
    对反应堆数据集的输入特征进行分类处理：
    连续变量 -> 归一化 (MinMaxScaler)
    离散变量 -> 独热编码 (OneHotEncoder)
    """
    print("开始进行特征工程：连续变量归一化 + 类别变量独热编码...")

    # 1. 读取之前切分好的三个数据集
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 明确特征分类
    continuous_cols = ['slope_up', 'cut_time']
    categorical_cols = ['material_changing', 'group_changing']

    scaler = MinMaxScaler()
    train_cont_scaled = scaler.fit_transform(df_train[continuous_cols])
    val_cont_scaled = scaler.transform(df_val[continuous_cols])
    test_cont_scaled = scaler.transform(df_test[continuous_cols])

    # 转回 DataFrame 格式
    df_train_cont = pd.DataFrame(train_cont_scaled, columns=continuous_cols)
    df_val_cont = pd.DataFrame(val_cont_scaled, columns=continuous_cols)
    df_test_cont = pd.DataFrame(test_cont_scaled, columns=continuous_cols)

    # ==========================================
    # 3. 处理离散型特征 (OneHotEncoder)
    # ==========================================
    # sparse_output=False 确保输出的是密集的 numpy 数组而不是稀疏矩阵
    # handle_unknown='ignore' 确保如果外推集出现了训练集没见过的类别，全置 0 而不报错
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    train_cat_encoded = encoder.fit_transform(df_train[categorical_cols])
    val_cat_encoded = encoder.transform(df_val[categorical_cols])
    test_cat_encoded = encoder.transform(df_test[categorical_cols])

    # 自动获取独热编码后的新列名 (例如: Material_Changing_1, Material_Changing_2...)
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    
    df_train_cat = pd.DataFrame(train_cat_encoded, columns=encoded_cols)
    df_val_cat = pd.DataFrame(val_cat_encoded, columns=encoded_cols)
    df_test_cat = pd.DataFrame(test_cat_encoded, columns=encoded_cols)

    # ==========================================
    # 4. 拼接最终数据并保留索引列
    # ==========================================
    def build_final_df(original_df, cont_df, cat_df):
        # 重置索引，防止拼接时出现错位
        original_df = original_df.reset_index(drop=True)
        
        # 尝试提取或生成 Case_ID，用于后续匹配功率文件
        if 'case_id' in original_df.columns:
            df_preserve = original_df[['case_id']]
        else:
            print("⚠️ 未发现 'Case_ID' 列，将使用原始行号作为标识。建议在原始数据中加入 Case_ID。")
            df_preserve = pd.DataFrame({'case_id': original_df.index})

        # 横向拼接：ID + 归一化后的连续变量 + 独热编码后的类别变量
        return pd.concat([df_preserve, cont_df, cat_df], axis=1)

    df_train_final = build_final_df(df_train, df_train_cont, df_train_cat)
    df_val_final = build_final_df(df_val, df_val_cont, df_val_cat)
    df_test_final = build_final_df(df_test, df_test_cont, df_test_cat)

    # ==========================================
    # 5. 保存处理结果和编码器模型
    # ==========================================
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存 CSV
    df_train_final.to_csv(os.path.join(output_dir, 'dataset_train_processed.csv'), index=False)
    df_val_final.to_csv(os.path.join(output_dir, 'dataset_val_processed.csv'), index=False)
    df_test_final.to_csv(os.path.join(output_dir, 'dataset_test_extrapolation_processed.csv'), index=False)

    # 保存 scaler 和 encoder，未来用新数据做预测时必须加载它们
    joblib.dump(scaler, os.path.join(output_dir, 'continuous_scaler.pkl'))
    joblib.dump(encoder, os.path.join(output_dir, 'categorical_encoder.pkl'))

    print("\n特征工程与归一化全部完成！")
    print(f"生成的特征矩阵包含 {df_train_final.shape[1]} 列。")
    print(f"最终列名预览: {df_train_final.columns.tolist()}")
    print(f"处理后的文件已保存为 'data_processed.csv'。")
    print(f"Scaler 和 Encoder 对象已保存为 '.pkl' 文件，存放于 {output_dir}/ 目录。")

    return df_train_final, df_val_final, df_test_final

# 执行函数
train_processed, val_processed, test_processed = process_and_normalize_features(TRAIN_PATH, VAL_PATH, TEST_PATH)