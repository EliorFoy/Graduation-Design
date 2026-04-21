"""
SVM 分类器模块

实现支持向量机分类器，用于运动想象 EEG 信号分类
包含网格搜索调参和交叉验证功能
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    accuracy_score, confusion_matrix, 
    classification_report, cohen_kappa_score
)
from matplotlib import pyplot as plt


def train_svm_classifier(X, y, cv_folds=10, kernel='rbf', random_state=42):
    """
    训练 SVM 分类器并评估
    
    Args:
        X: 特征矩阵 (n_trials, n_features)
        y: 标签向量 (n_trials,)
        cv_folds: 交叉验证折数（默认 10）
        kernel: 核函数类型（'rbf', 'linear', 'poly'）
        random_state: 随机种子
    
    Returns:
        clf: 训练好的 SVM 模型
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
    
    # 创建 SVM 分类器
    clf = SVC(kernel=kernel, random_state=random_state)
    
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


def optimize_svm_hyperparameters(X, y, cv_folds=10, param_grid=None):
    """
    使用网格搜索优化 SVM 超参数
    
    Args:
        X: 特征矩阵
        y: 标签向量
        cv_folds: 交叉验证折数
        param_grid: 参数网格（为 None 则使用默认配置）
    
    Returns:
        best_clf: 最优参数的 SVM 模型
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
    
    # 创建网格搜索对象
    grid_search = GridSearchCV(
        SVC(random_state=42),
        param_grid,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1  # 并行计算
    )
    
    # 执行网格搜索
    print("\n正在进行网格搜索...")
    grid_search.fit(X, y)
    
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


def plot_confusion_matrix(cm, class_names=None, save_path='./confusion_matrix.png'):
    """
    可视化混淆矩阵
    
    Args:
        cm: 混淆矩阵 (n_classes, n_classes)
        class_names: 类别名称列表
        save_path: 保存路径
    """
    print("\n绘制混淆矩阵图...")
    
    n_classes = cm.shape[0]
    
    if class_names is None:
        class_names = [f'Class {i+1}' for i in range(n_classes)]
    
    # 确保类别名称数量与矩阵维度匹配
    if len(class_names) != n_classes:
        print(f"⚠️  警告：类别名称数量 ({len(class_names)}) 与矩阵维度 ({n_classes}) 不匹配")
        print(f"   将使用默认类别名称")
        class_names = [f'Class {i+1}' for i in range(n_classes)]
    
    # 归一化（转换为百分比）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
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


def plot_cv_results(cv_results, save_path='./cv_results.png'):
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
    
    # 提取 C 的值
    C_values = sorted(list(set([p['C'] for p in params])))
    
    # 【修复 Bug】处理 gamma 包含字符串 (如 'scale') 和浮点数混合排序的问题
    def gamma_sort_key(g):
        return g if isinstance(g, (int, float)) else float('inf')
    
    gamma_values = sorted(list(set([p.get('gamma') for p in params if 'gamma' in p])), key=gamma_sort_key)
    
    if len(C_values) > 1 and len(gamma_values) > 1:
        # 绘制热力图
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