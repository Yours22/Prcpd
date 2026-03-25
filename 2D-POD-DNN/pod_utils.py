import numpy as np
from sklearn.utils.extmath import randomized_svd

class PODReducer:
    """
    支持固定截断阶数的 POD 降维器
    """
    def __init__(self, r=10):
        self.r = r  # 直接指定保留的模态数
        self.mean_vec = None
        self.basis = None   
        self.singular_values = None

    def fit(self, snapshots):
        U = snapshots.T
        self.mean_vec = np.mean(U, axis=1, keepdims=True)
        U_centered = U - self.mean_vec
        
        # 提取前 r 个模态即可，不需要计算全部
        U_basis, Sigma, VT = randomized_svd(U_centered, n_components=self.r, random_state=42)
        
        self.basis = U_basis
        self.singular_values = Sigma
        
        print(f"[POD Info] 固定提取模态 r = {self.r}")
        print(f"[POD Info] 奇异值: {np.round(Sigma, 2)}")

    def transform(self, snapshots):
        if self.basis is None:
            raise ValueError("Model not fitted yet!")
        U = snapshots.T
        U_centered = U - self.mean_vec
        coeffs = np.dot(self.basis.T, U_centered).T
        return coeffs

    def inverse_transform(self, coeffs):
        U_rec = self.mean_vec + np.dot(self.basis, coeffs.T)
        return U_rec.T