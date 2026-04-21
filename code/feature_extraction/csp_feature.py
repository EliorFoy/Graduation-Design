"""
CSP (Common Spatial Pattern) 特征提取模块

CSP 是一种空域滤波算法，通过优化投影方向使得两类信号的方差差异最大化
广泛应用于运动想象 BCI 系统的特征提取
"""

import numpy as np
import mne
from scipy.linalg import eigh
from matplotlib import pyplot as plt


def extract_csp_features(epochs, n_components=4, reg_param=0.0):
    """
    使用 CSP 算法提取空间滤波特征

    Args:
        epochs: 预处理后的 Epochs 数据 (n_trials, n_channels, n_times)
               注意：应该只包含两类或四类任务标签
        n_components: CSP 成分数（默认 4，即 2 对滤波器）
                     - 对于二分类：通常用 2-4 个成分
                     - 对于四分类：可以用更多成分
        reg_param: 正则化参数（0-1），防止协方差矩阵奇异

    Returns:
        X_csp: CSP 特征矩阵 (n_trials, n_components)
        csp: 训练好的 CSP 对象（包含滤波器等）
    """
    print("\n" + "=" * 60)
    print("CSP 特征提取")
    print("=" * 60)
    
    # 获取数据
    data = epochs.get_data()  # shape: (n_trials, n_channels, n_times)
    labels = epochs.events[:, 2]  # shape: (n_trials,)
    
    n_trials, n_channels, n_times = data.shape
    print(f"   - 输入数据形状：{data.shape}")
    print(f"   - 试次数：{n_trials}")
    print(f"   - 通道数：{n_channels}")
    print(f"   - 类别数：{len(np.unique(labels))}")
    print(f"   - CSP 成分数：{n_components}")
    
    # 1. 为每个类别计算平均协方差矩阵
    unique_labels = np.unique(labels)
    class_covariances = {}
    
    for label in unique_labels:
        # 提取该类别的所有试次
        class_data = data[labels == label]  # (n_class_trials, n_channels, n_times)
        
        # 计算每个试次的协方差矩阵
        cov_matrices = []
        for trial_data in class_data:
            # 协方差矩阵：R = (X @ X.T) / trace(X @ X.T)
            cov = np.cov(trial_data)
            
            # 【修复】迹归一化（极其重要，原代码缺失）
            cov = cov / np.trace(cov)
            
            # 添加正则化
            if reg_param > 0:
                cov += reg_param * np.eye(n_channels)
            
            cov_matrices.append(cov)
        
        # 平均协方差矩阵
        class_covariances[label] = np.mean(cov_matrices, axis=0)
    
    # 2. 构建类间散度矩阵
    R_total = np.zeros((n_channels, n_channels))
    for label in unique_labels:
        R_total += class_covariances[label]
    
    # 3. 白化变换
    eigenvalues, eigenvectors = eigh(R_total)
    
    # 排序（从大到小）
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 白化矩阵：W = D^(-1/2) @ E.T
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + 1e-10))
    W = D_inv_sqrt @ eigenvectors.T
    
    # 4. 对白化后的协方差矩阵进行投影
    P_list = []
    for label in unique_labels:
        P = W @ class_covariances[label] @ W.T
        P_list.append(P)
    
    # 5. 求解最优投影向量
    if len(unique_labels) == 2:
        # 【二分类情况修复】
        P_diff = P_list[0] - P_list[1]
        eigenvalues_proj, eigenvectors_proj = eigh(P_diff)
        
        # 排序
        idx = np.argsort(eigenvalues_proj)[::-1]
        eigenvectors_proj = eigenvectors_proj[:, idx]
        eigenvalues_proj = eigenvalues_proj[idx]
        
        # 构建所有的 CSP 滤波器
        V = eigenvectors_proj.T @ W
        
        # 修复逻辑：CSP必须同时提取特征值最大（对类1敏感）和最小（对类2敏感）的两端滤波器
        half_n = n_components // 2
        if n_components % 2 != 0:
            half_n = (n_components + 1) // 2
            
        top_idx = list(range(half_n))
        bottom_idx = list(range(V.shape[0] - (n_components - half_n), V.shape[0]))
        selected_idx = top_idx + bottom_idx
        
        W_csp = V[selected_idx, :]
        selected_eigenvalues = eigenvalues_proj[selected_idx]
        
    else:
        # 【多分类情况修复：使用 One-Vs-Rest (一对多) 策略】
        n_comp_per_class = max(1, n_components // len(unique_labels))
        W_csp_list = []
        selected_eigenvalues_list = []
        
        for i in range(len(unique_labels)):
            P_i = P_list[i]
            # 计算除自己外其他所有类的平均协方差
            P_others = np.mean([P_list[j] for j in range(len(unique_labels)) if j != i], axis=0)
            P_diff = P_i - P_others
            
            eigenvals, eigenvecs = eigh(P_diff)
            idx = np.argsort(eigenvals)[::-1]
            eigenvecs = eigenvecs[:, idx]
            eigenvals = eigenvals[idx]
            
            V = eigenvecs.T @ W
            
            # 每个类取前 n_comp_per_class 个最大化自己方差的滤波器
            W_csp_list.append(V[:n_comp_per_class, :])
            selected_eigenvalues_list.extend(eigenvals[:n_comp_per_class])
            
        W_csp = np.vstack(W_csp_list)
        selected_eigenvalues = np.array(selected_eigenvalues_list)
        
        # 若因除不尽导致数量超出预期，则进行截断
        if W_csp.shape[0] > n_components:
            W_csp = W_csp[:n_components, :]
            selected_eigenvalues = selected_eigenvalues[:n_components]
    
    print("✅ CSP 滤波器训练完成")
    print(f"   - 滤波器形状：{W_csp.shape}")
    
    # 7. 应用 CSP 滤波器提取特征
    X_csp = []
    for trial_idx in range(n_trials):
        trial_data = data[trial_idx]
        # 投影
        Z = W_csp @ trial_data
        # 计算对数方差
        log_var = np.log(np.var(Z, axis=1))
        X_csp.append(log_var)
    
    X_csp = np.array(X_csp)
    
    print("✅ CSP 特征提取完成")
    print(f"   - 特征形状：{X_csp.shape}")
    
    # 8. 保存 CSP 对象
    # 修复空间模式(Patterns)计算公式：投影矩阵的伪逆
    patterns = np.linalg.pinv(W_csp).T
    
    csp = {
        'filters': W_csp,
        'patterns': patterns,
        'eigenvalues': selected_eigenvalues,
        'n_components': W_csp.shape[0],
        'n_channels': n_channels
    }
    
    return X_csp, csp


def compute_csp_patterns(W_csp, R_total=None):
    """
    计算 CSP 模式（用于可视化拓扑图）
    直接使用滤波器矩阵的伪逆，更稳定。
    """
    return np.linalg.pinv(W_csp).T


def visualize_csp_topo(csp, epochs, save_path='./csp_topoplots.png'):
    """
    可视化 CSP 滤波器的头皮拓扑图
    """
    print("\n绘制 CSP 拓扑图...")
    
    # 检查是否有电极位置信息
    if epochs.info['dig'] is None or len(epochs.info['dig']) == 0:
        print("⚠️  警告：未找到电极位置信息，跳过 CSP 拓扑图绘制")
        print("   提示：这不影响 CSP 特征提取，只是无法可视化拓扑图")
        return
    
    filters = csp['filters']
    patterns = csp['patterns']
    n_components = csp['n_components']
    
    fig, axes = plt.subplots(1, n_components, figsize=(4*n_components, 4))
    
    if n_components == 1:
        axes = [axes]
    
    for i in range(n_components):
        pattern = patterns[i]
        evoked = mne.EvokedArray(pattern.reshape(-1, 1), epochs.info, tmin=0)
        
        # 绘制 topomap
        im = evoked.plot_topomap(times=[0], axes=axes[i], show=False, colorbar=False)
        
        # 添加特征值说明
        eigen_val = csp["eigenvalues"][i]
        axes[i].set_title(f'Component {i+1}\n($\lambda$={eigen_val:.2f})')
    
    plt.suptitle('CSP Spatial Patterns', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ CSP 拓扑图已保存：{save_path}")
    plt.close()


if __name__ == "__main__":
    print("CSP 特征提取模块")
    print("\n使用方法:")
    print("from feature_extraction import extract_csp_features")
    print("X_csp, csp = extract_csp_features(epochs, n_components=4)")
    print("\n对于二分类问题，推荐使用 n_components=2 或 4")
    print("对于四分类问题，建议使用更多成分（如 4-8）")