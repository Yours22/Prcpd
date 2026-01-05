import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import randomized_svd

raw_data = np.load('.\\data-gen\\outputs.npy') 

N = raw_data.shape[0] # N_1 (样本数): 200
T = raw_data.shape[1] # T (时间步): 100
K = raw_data.shape[2] # 空间特征数，即两群中子的40x40网格的数据: 1600 * 2 = 3200

# CSTR（时空耦合降维）格式转换，把时间步和具体数据展平在同一行，也可以考虑用独立空间降维 (ISR)
U_CTSR = raw_data.reshape(N, -1).T 
print(f"CTSR 快照矩阵形状: {U_CTSR.shape}")


def data_processing():
    print("正在分割数据")
    outputs = np.load('.\\data-gen\\outputs.npy') 
    inputs = np.load('.\\data-gen\\inputs.npy') 

    X_train,X_test,Y_raw_train, Y_raw_test = train_test_split(
        inputs,outputs,test_size=0.2,random_state=42
    )

    print(f"训练集样本数 (N1): {Y_raw_train.shape[0]}")
    print(f"测试集样本数 (N2): {Y_raw_test.shape[0]}")

    N1= Y_raw_train.shape[0]
    U_snapshot = Y_raw_train.reshape(N1,-1).T
    print(f"快照矩阵形状为 U1: {U_snapshot.shape} ")

# 进行POD降解，初步打算使用randomized svd
    print("进行svd分解")
    U_basis, Sigma, VT = randomized_svd(
        U_snapshot, 
        n_components=N1,
        random_state=42
    
    )

    energy = Sigma**2
    cumulative_energy = np.cumsum(energy) / np.sum(energy)
    target_energy = 0.9999999 ## 降维之后的数据能够保留这么多的能量
    r = np.searchsorted(cumulative_energy, target_energy) + 1
    print(f"累计能量达到 {target_energy*100}% 需要的模态数 r = {r}")

    # 能量衰减图 (对应论文 Fig. 4)
    plt.figure(figsize=(8, 4))
    plt.plot(cumulative_energy, marker='o', markersize=3)
    plt.axhline(y=target_energy, color='r', linestyle='--', label=f'{target_energy} Threshold')
    plt.axvline(x=r, color='g', linestyle='--', label=f'r={r}')
    plt.xlabel('Mode Number')
    plt.ylabel('Cumulative Energy')
    plt.title('Decay of Mode Energy (Reproducing Fig. 4)')
    plt.legend()
    plt.grid(True)
    plt.savefig('energy_decay.png')
    print("能量衰减图已保存为 energy_decay.png")

    # 计算降阶系数 
    Phi_r = U_basis[:, :r]
    
    # 系数矩阵 A = Phi^T * U
    # 结果形状: (r, N1) -> 转置为 (N1, r) 以便作为神经网络的 Y 标签
    coeffs_train = np.dot(Phi_r.T, U_snapshot).T
    
    # 测试集先投影出来，用于后续验证，方便计算误差
    # 注意：测试集必须用训练集的 Phi 投影！
    U_test = Y_raw_test.reshape(Y_raw_test.shape[0], -1).T
    coeffs_test = np.dot(Phi_r.T, U_test).T

    print(f"训练集系数形状 (DNN Targets): {coeffs_train.shape}")
    
    np.savez('.\\data-gen\\pod_data.npz', 
             X_train=X_train, 
             X_test=X_test, 
             coeffs_train=coeffs_train, 
             coeffs_test=coeffs_test,
             Phi_r=Phi_r,          
             mean_vec=np.mean(U_snapshot, axis=1) 
            )
    print(">>> 所有预处理数据已保存至 pod_data.npz")

if __name__ == "__main__":
    data_processing()