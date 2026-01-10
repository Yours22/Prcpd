import numpy as np
from sklearn.utils.extmath import randomized_svd

class PODReducer:
    """
    处理物理场的本征正交分解 (POD) 逻辑
    包含：去均值、SVD分解、能量截断、投影、重构
    """
    def __init__(self, target_energy=0.999):
        self.target_energy = target_energy
        self.mean_vec = None
        self.basis = None   # Phi_r
        self.r = None       # 保留的模态数
        self.singular_values = None

    def fit(self, snapshots):
        """
        计算 POD 基向量
        snapshots shape: (N_samples, N_features) -> 例如 (160, 320000)
        """
        # 1. 转换为快照矩阵 (Features, Samples)
        U = snapshots.T
        
        # 2. 计算并移除均值
        self.mean_vec = np.mean(U, axis=1, keepdims=True)
        U_centered = U - self.mean_vec
        
        # 3. Randomized SVD (通常取一个较大的 n_components 上限，例如 200 或样本数)
        # 注意：这里 n_components 不能超过 min(U_centered.shape)
        n_components = min(U_centered.shape)
        U_basis, Sigma, VT = randomized_svd(U_centered, n_components=n_components, random_state=42)
        
        # 4. 能量截断
        energy = Sigma**2
        cumulative_energy = np.cumsum(energy) / np.sum(energy)
        self.r = np.searchsorted(cumulative_energy, self.target_energy) + 1
        
        self.basis = U_basis[:, :self.r]
        self.singular_values = Sigma[:self.r]
        
        print(f"[POD Info] 目标能量: {self.target_energy*100}% -> 保留模态 r = {self.r}")
        print(f"[POD Info] 累计解释方差: {cumulative_energy[self.r-1]:.6f}")

    def transform(self, snapshots):
        """
        将物理场投影为低维系数 (Physical Space -> Latent Space)
        return: coefficients (N_samples, r)
        """
        if self.basis is None:
            raise ValueError("Model not fitted yet!")
            
        U = snapshots.T
        U_centered = U - self.mean_vec
        # 公式: Alpha = Phi^T * (U - Mean)
        coeffs = np.dot(self.basis.T, U_centered).T
        return coeffs

    def inverse_transform(self, coeffs):
        """
        将低维系数重构为物理场 (Latent Space -> Physical Space)
        coeffs: (N_samples, r)
        return: reconstructed_snapshots (N_samples, N_features)
        """
        # 公式: U_rec = Mean + Phi * Alpha^T
        U_rec = self.mean_vec + np.dot(self.basis, coeffs.T)
        return U_rec.T

    def save(self, path):
        """保存 POD 核心组件"""
        np.savez(path, mean_vec=self.mean_vec, basis=self.basis, r=self.r)
        print(f"POD 组件已保存至 {path}")

    def load(self, path):
        """加载 POD 核心组件"""
        data = np.load(path)
        self.mean_vec = data['mean_vec']
        self.basis = data['basis']
        self.r = int(data['r'])
        print(f"POD 组件已加载 (r={self.r})")