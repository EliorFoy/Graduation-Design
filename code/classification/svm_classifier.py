"""
SVM 分类器模块

实现支持向量机分类器，用于运动想象 EEG 信号分类
包含网格搜索调参和交叉验证功能
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, 
    classification_report, cohen_kappa_score
)
from matplotlib import pyplot as plt
try:
    from ..feature_extraction.eeg_transformers import EpochsToArray, make_feature_union, FilterBankCSP
except ImportError:
    from feature_extraction.eeg_transformers import EpochsToArray, make_feature_union, FilterBankCSP


def make_eeg_svm_pipeline(
    feature_set='fused',
    n_csp_components=4,
    wavelet='db4',
    wavelet_level=4,
    kernel='rbf',
    random_state=42,
):
    """Create a leakage-safe EEG feature extraction + scaler + SVM pipeline."""

    return Pipeline([
        ('epochs_to_array', EpochsToArray()),
        ('features', make_feature_union(
            feature_set=feature_set,
            n_csp_components=n_csp_components,
            wavelet=wavelet,
            wavelet_level=wavelet_level,
        )),
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel=kernel, random_state=random_state)),
    ])


def train_eeg_svm_pipeline(
    epochs_or_data,
    y,
    feature_set='fused',
    cv_folds=10,
    n_csp_components=4,
    wavelet='db4',
    wavelet_level=4,
    kernel='rbf',
    random_state=42,
    freq_bands=None,  # 【新增】用于 FBCSP
):
    """
    训练并交叉验证 EEG SVM Pipeline（无特征泄漏）
    
    Args:
        epochs_or_data: MNE Epochs 对象或数组数据
        y: 标签向量
        feature_set: 特征集类型 ('csp', 'wavelet', 'fused', 'fb_csp')
        cv_folds: 交叉验证折数
        n_csp_components: CSP 成分数
        wavelet: 小波基类型
        wavelet_level: 小波分解层数
        kernel: SVM 核函数
        random_state: 随机种子
        freq_bands: FBCSP 频段列表，如 [(8,12), (12,16), ...]
    
    Returns:
        pipeline: 训练好的 Pipeline
        cv_scores: 交叉验证得分数组
        mean_accuracy: 平均准确率
    """

    print("\n" + "=" * 60)
    print(f"EEG SVM 管线训练（{feature_set} 特征，无泄漏 CV）")
    print("=" * 60)

    # 【关键修复】显式将 Epochs 转为三维数组，避免 cross_val_score 索引时维度丢失
    if hasattr(epochs_or_data, 'get_data'):
        X = epochs_or_data.get_data()  # 三维数组 (n_trials, n_channels, n_times)
        sfreq = epochs_or_data.info['sfreq']
        print(f"   - 输入数据类型：Epochs → numpy.ndarray")
        print(f"   - 数据形状：{X.shape}")
        print(f"   - 采样率：{sfreq} Hz")
    else:
        X = np.asarray(epochs_or_data)
        sfreq = None
        print(f"   - 输入数据类型：numpy.ndarray")
        print(f"   - 数据形状：{X.shape}")
    
    # 确认形状为 3D
    if X.ndim != 3:
        raise ValueError(f"CSP 需要 3D 输入 (n_trials, n_channels, n_times)，但收到 {X.ndim}D 数据，形状：{X.shape}")
    
    y = np.asarray(y)
    print(f"   - 样本数：{len(y)}")
    print(f"   - 类别分布：{dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"   - 交叉验证折数：{cv_folds}")

    # 【新增】处理 FBCSP 特殊情况
    if feature_set == 'fb_csp':
        if sfreq is None:
            raise ValueError("FBCSP 需要提供 sfreq 参数，请传入 MNE Epochs 对象")
        
        # 默认频带：覆盖运动想象 μ 和 β 节律
        if freq_bands is None:
            freq_bands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]
        
        print(f"   - FBCSP 频带数量: {len(freq_bands)}")
        print(f"   - 每个频段 CSP 成分数: {n_csp_components}")
        print(f"   - 总特征维度: {len(freq_bands) * n_csp_components}")
        
        # 构建 FBCSP Pipeline
        from sklearn.pipeline import Pipeline as SklearnPipeline
        pipeline = SklearnPipeline([
            ('fbcsp', FilterBankCSP(
                freq_bands=freq_bands,
                sfreq=sfreq,
                n_components=n_csp_components,
                log=True,
                reg=None,
            )),
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel=kernel, random_state=random_state)),
        ])
    else:
        # 原有 csp, wavelet, fused 逻辑
        pipeline = make_eeg_svm_pipeline(
            feature_set=feature_set,
            n_csp_components=n_csp_components,
            wavelet=wavelet,
            wavelet_level=wavelet_level,
            kernel=kernel,
            random_state=random_state,
        )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    
    # 【关键修复】在全部数据上拟合 Pipeline（包含特征提取、标准化、分类）
    pipeline.fit(X, y)

    mean_accuracy = cv_scores.mean()
    print("\n✅ EEG SVM Pipeline 训练完成")
    print(f"   - CV 准确率：{mean_accuracy:.4f} ± {cv_scores.std():.4f}")
    print(f"   - 各折得分：{np.round(cv_scores, 4)}")
    return pipeline, cv_scores, mean_accuracy


def train_svm_classifier(X, y, cv_folds=10, kernel='rbf', random_state=42, auto_scale=True):
    """
    训练 SVM 分类器并评估
    
    ⚠️  重要：SVM 对特征尺度敏感，建议启用 auto_scale 或预先标准化输入。
    
    Args:
        X: 特征矩阵 (n_trials, n_features)
        y: 标签向量 (n_trials,)
        cv_folds: 交叉验证折数（默认 10）
        kernel: 核函数类型（'rbf', 'linear', 'poly'）
        random_state: 随机种子
        auto_scale: 是否自动添加 StandardScaler（推荐 True）
    
    Returns:
        clf: 训练好的 SVM 模型（若 auto_scale=True，返回 Pipeline）
        cv_scores: 交叉验证得分数组
        mean_accuracy: 平均准确率
    """
    print("\n" + "=" * 60)
    print("SVM 分类器训练")
    print("=" * 60)
    
    print(f"   - 特征矩阵形状：{X.shape}")
    print(f"   - 样本数：{len(y)}")
    print(f"   - 特征数：{X.shape[1]}")
    print(f"   - 类别分布：{np.bincount(y)}")
    print(f"   - 交叉验证折数：{cv_folds}")
    print(f"   - 核函数：{kernel}")
    print(f"   - 自动标准化：{auto_scale}")
    
    # 【关键修复】创建带标准化的 Pipeline 或直接使用 SVC
    if auto_scale:
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler
        
        clf = SklearnPipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel=kernel, random_state=random_state))
        ])
        print("   ✅ 已启用自动标准化（StandardScaler）")
    else:
        clf = SVC(kernel=kernel, random_state=random_state)
        print("   ⚠️  警告：未启用自动标准化，请确保输入特征已标准化")
    
    # 分层 K 折交叉验证（保持各类别比例）
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # 执行交叉验证
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    
    # 在全部数据上训练最终模型
    clf.fit(X, y)
    
    mean_accuracy = cv_scores.mean()
    std_accuracy = cv_scores.std()
    
    print("\n✅ SVM 训练完成")
    print(f"   - 交叉验证准确率：{mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"   - 各折得分：{np.round(cv_scores, 4)}")
    
    return clf, cv_scores, mean_accuracy


def optimize_svm_hyperparameters(X, y, cv_folds=10, param_grid=None, auto_scale=True):
    """
    使用网格搜索优化 SVM 超参数
    
    ⚠️  重要：默认启用自动标准化，避免特征尺度影响搜索结果。
    
    Args:
        X: 特征矩阵
        y: 标签向量
        cv_folds: 交叉验证折数
        param_grid: 参数网格（为 None 则使用默认配置）
        auto_scale: 是否自动添加 StandardScaler（推荐 True）
    
    Returns:
        best_clf: 最优参数的 SVM 模型（Pipeline 或 SVC）
        best_params: 最优参数组合
        best_score: 最优交叉验证得分
    """
    print("\n" + "=" * 60)
    print("SVM 超参数网格搜索")
    print("=" * 60)
    
    # 默认参数网格
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf']
        }
    
    print("   - 参数搜索空间:")
    for param, values in param_grid.items():
        print(f"     {param}: {values}")
    print(f"   - 自动标准化：{auto_scale}")
    
    # 【关键修复】构建带标准化的 Pipeline
    if auto_scale:
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler
        
        base_estimator = SklearnPipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(random_state=42))
        ])
        # 调整参数网格，添加 'svm__' 前缀
        scaled_param_grid = {}
        for key, values in param_grid.items():
            scaled_param_grid[f'svm__{key}'] = values
        param_grid = scaled_param_grid
        print("   ✅ 已启用自动标准化（StandardScaler）")
    else:
        base_estimator = SVC(random_state=42)
        print("   ⚠️  警告：未启用自动标准化，请确保输入特征已标准化")
    
    # 创建网格搜索对象
    grid_search = GridSearchCV(
        base_estimator,
        param_grid,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1  # 并行计算
    )
    
    # 执行网格搜索
    try:
        print("\n正在进行网格搜索...")
        grid_search.fit(X, y)
    except Exception as e:
        print(f"\n❌ 网格搜索失败：{e}")
        raise
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_clf = grid_search.best_estimator_
    
    print("\n✅ 网格搜索完成")
    print(f"   - 最优参数：{best_params}")
    print(f"   - 最优交叉验证得分：{best_score:.4f}")
    
    # 显示所有参数组合的结果
    print("\n详细结果:")
    results = grid_search.cv_results_
    for mean_score, params in zip(results['mean_test_score'], results['params']):
        print(f"   - 准确率：{mean_score:.4f}, 参数：{params}")
    
    return best_clf, best_params, best_score


def evaluate_model(y_true, y_pred, class_names=None):
    """
    全面评估分类性能
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
    
    Returns:
        metrics_dict: 包含所有指标的字典
    """
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)
    
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    
    # 计算 Cohen's Kappa 系数（衡量一致性）
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 分类报告
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    print(f"   - 准确率：{accuracy:.4f}")
    print(f"   - Cohen's Kappa: {kappa:.4f}")
    print(f"\n混淆矩阵:\n{cm}")
    print(f"\n分类报告:\n{report}")
    
    metrics = {
        'accuracy': accuracy,
        'kappa': kappa,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return metrics


def plot_confusion_matrix(cm, class_names=None, save_path='./output_img/confusion_matrix.png'):
    """
    可视化混淆矩阵
    
    Args:
        cm: 混淆矩阵 (n_classes, n_classes)
        class_names: 类别名称列表
        save_path: 保存路径
    """
    print("\n绘制混淆矩阵图...")
    
    # 确保输出目录存在
    from pathlib import Path
    output_dir = Path(save_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_classes = cm.shape[0]
    
    if class_names is None:
        class_names = [f'Class {i+1}' for i in range(n_classes)]
    
    # 确保类别名称数量与矩阵维度匹配
    if len(class_names) != n_classes:
        print(f"⚠️  警告：类别名称数量 ({len(class_names)}) 与矩阵维度 ({n_classes}) 不匹配")
        print(f"   将使用默认类别名称")
        class_names = [f'Class {i+1}' for i in range(n_classes)]
    
    # 【优化】安全归一化，避免除零错误
    row_sums = cm.sum(axis=1)
    cm_normalized = np.zeros_like(cm, dtype=float)
    for i in range(n_classes):
        if row_sums[i] > 0:
            cm_normalized[i] = cm[i].astype('float') / row_sums[i] * 100
        else:
            print(f"   ⚠️  警告：类别 {i} 在真实标签中未出现，归一化结果为 NaN")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制热力图
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, label='Percentage (%)')
    
    # 添加文本标签 - 直接设置，不使用 set_xticks
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(class_names)
    
    # 在每个单元格中显示数值
    thresh = cm_normalized.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.1f}%)',
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black")
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 混淆矩阵图已保存：{save_path}")
    plt.close()


def plot_cv_results(cv_results, save_path='./output_img/cv_results.png'):
    """
    可视化交叉验证结果
    
    Args:
        cv_results: GridSearchCV 的 cv_results_ 字典
        save_path: 保存路径
    """
    print("\n绘制交叉验证结果图...")
    
    # 提取参数和得分
    params = cv_results['params']
    mean_scores = cv_results['mean_test_score']
    
    # 【优化】仅当参数网格包含 RBF 核且有 gamma 参数时才绘制热力图
    has_rbf = any('gamma' in str(p) or p.get('kernel') in [None, 'rbf'] for p in params)
    
    # 提取 C 的值
    C_values = sorted(list(set([p['C'] for p in params if 'C' in p])))
    
    gamma_values = []
    
    if has_rbf:
        # 【修复】更健壮的 gamma 值提取和排序
        gamma_raw = [p.get('gamma') for p in params if 'gamma' in p and p.get('gamma') is not None]
        if gamma_raw:
            # 分离数值型和字符串型 gamma
            numeric_gammas = sorted([g for g in gamma_raw if isinstance(g, (int, float))])
            string_gammas = sorted([str(g) for g in gamma_raw if not isinstance(g, (int, float))])
            gamma_values = numeric_gammas + string_gammas
    
    if len(C_values) > 1 and len(gamma_values) > 1:
        # 绘制 C-gamma 热力图
        C_gamma_scores = np.zeros((len(C_values), len(gamma_values)))
        
        for i, C in enumerate(C_values):
            for j, gamma in enumerate(gamma_values):
                # 找到对应的参数组合
                for idx, param in enumerate(params):
                    if param['C'] == C and param.get('gamma') == gamma:
                        C_gamma_scores[i, j] = mean_scores[idx]
                        break
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(C_gamma_scores, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Accuracy')
        
        ax.set_xticks(np.arange(len(gamma_values)))
        ax.set_yticks(np.arange(len(C_values)))
        # 将所有 gamma 值转为字符串显示
        ax.set_xticklabels([str(g) for g in gamma_values])
        ax.set_yticklabels(C_values)
        
        ax.set_xlabel('Gamma')
        ax.set_ylabel('C')
        ax.set_title('Cross-Validation Accuracy')
        
        # 显示数值
        thresh = C_gamma_scores.max() / 2.
        for i in range(len(C_values)):
            for j in range(len(gamma_values)):
                ax.text(j, i, f'{C_gamma_scores[i, j]:.3f}',
                        ha="center", va="center",
                        color="white" if C_gamma_scores[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 交叉验证结果图已保存：{save_path}")
        plt.close()


# ========== 使用示例 ==========
if __name__ == "__main__":
    print("SVM 分类器模块")
    print("\n使用方法:")
    print("from classification import train_svm_classifier, optimize_svm_hyperparameters")
    print("# 1. 基础训练")
    print("clf, cv_scores, acc = train_svm_classifier(X, y)")
    print("\n# 2. 网格搜索调参")
    print("best_clf, best_params, best_score = optimize_svm_hyperparameters(X, y)")
    print("\n# 3. 模型评估")
    print("metrics = evaluate_model(y_true, y_pred)")
    print("\n推荐配置:")
    print("  - 核函数：RBF")
    print("  - 交叉验证：10 折分层抽样")
    print("  - 参数搜索：C=[0.1, 1, 10, 100], gamma=['scale', 0.001, 0.01, 0.1, 1]")
