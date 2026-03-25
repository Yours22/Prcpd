import os
import numpy as np
import joblib
from pod_utils import PODReducer

CONFIG = {
    'data_dir': '2D-POD-DNN/data',
    'model_dir':'2D-POD-DNN/model',
    'r_fast': 4, 
    'r_therm': 4 
}

def main():
    print("===== 1. 加载高维原始物理场数据 =====")
    Y_train = np.load(os.path.join(CONFIG['data_dir'], 'Y_train_raw.npy'))
    Y_val   = np.load(os.path.join(CONFIG['data_dir'], 'Y_val_raw.npy'))
    Y_test  = np.load(os.path.join(CONFIG['data_dir'], 'Y_test_raw.npy'))
    
    # 拆分快群和热群 (前400是快，后400是热)
    Y_train_fast, Y_train_therm = Y_train[:, :400], Y_train[:, 400:]
    Y_val_fast, Y_val_therm     = Y_val[:, :400],   Y_val[:, 400:]
    Y_test_fast, Y_test_therm   = Y_test[:, :400],  Y_test[:, 400:]

    print("\n===== 2. 执行 POD 降维 (快群) =====")
    pod_fast = PODReducer(r=CONFIG['r_fast'])
    pod_fast.fit(Y_train_fast)
    A_train_fast = pod_fast.transform(Y_train_fast)
    A_val_fast   = pod_fast.transform(Y_val_fast)
    A_test_fast  = pod_fast.transform(Y_test_fast)

    print("\n===== 3. 执行 POD 降维 (热群) =====")
    pod_therm = PODReducer(r=CONFIG['r_therm'])
    pod_therm.fit(Y_train_therm)
    A_train_therm = pod_therm.transform(Y_train_therm)
    A_val_therm   = pod_therm.transform(Y_val_therm)
    A_test_therm  = pod_therm.transform(Y_test_therm)

    print("\n===== 4. 拼接降阶系数 =====")
    # 将快群和热群的系数拼接在一起，作为神经网络的统一目标
    A_train = np.hstack((A_train_fast, A_train_therm))
    A_val   = np.hstack((A_val_fast, A_val_therm))
    A_test  = np.hstack((A_test_fast, A_test_therm))
    
    print(f"最终 DNN 的标签 Y 的形状:")
    print(f"Train: {A_train.shape}, Val: {A_val.shape}, Test: {A_test.shape}")

    print("\n===== 5. 保存数据 =====")
    joblib.dump({'fast': pod_fast, 'therm': pod_therm}, 
                os.path.join(CONFIG['model_dir'], 'pod_models.pkl'))
    
    np.save(os.path.join(CONFIG['data_dir'], 'A_train.npy'), A_train)
    np.save(os.path.join(CONFIG['data_dir'], 'A_val.npy'), A_val)
    np.save(os.path.join(CONFIG['data_dir'], 'A_test.npy'), A_test)
    print("POD 分群提取完成！")

if __name__ == "__main__":
    main()