"""
多种分类器对比实验模块

用于比较不同机器学习/深度学习分类器在 EEG 运动想象任务上的性能，
包括：
- 传统机器学习：SVM、KNN、朴素贝叶斯、随机森林、逻辑回归
- 神经网络：MLP（反向传播）、CNN、LSTM
"""

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline
import time


def compare_classifiers(X, y, feature_name="未知特征", cv_folds=10, random_state=42):
    """
    对比多种分类器的性能
    
    Args:
        X: 特征矩阵 (n_samples, n_features)
        y: 标签向量 (n_samples,)
        feature_name: 特征名称（用于输出）
        cv_folds: 交叉验证折数
        random_state: 随机种子
    
    Returns:
        results: 字典，包含各分类器的性能指标
    """
    print("\n" + "=" * 70)
    print(f"分类器对比实验 - {feature_name}")
    print("=" * 70)
    print(f"数据形状: {X.shape}")
    print(f"样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")
    print(f"类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")
    print("=" * 70)
    
    # 交叉验证配置
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # 定义分类器列表
    classifiers = {
        # === 传统机器学习方法 ===
        'SVM (RBF)': {
            'model': SklearnPipeline([
                ('scaler', StandardScaler()),
                ('clf', _create_svm_classifier(random_state))
            ]),
            'description': '支持向量机（RBF核）'
        },
        'SVM (Linear)': {
            'model': SklearnPipeline([
                ('scaler', StandardScaler()),
                ('clf', _create_svm_linear_classifier(random_state))
            ]),
            'description': '支持向量机（线性核）'
        },
        'KNN': {
            'model': SklearnPipeline([
                ('scaler', StandardScaler()),
                ('clf', _create_knn_classifier())
            ]),
            'description': 'K近邻算法'
        },
        'Naive Bayes': {
            'model': _create_naive_bayes_classifier(),
            'description': '高斯朴素贝叶斯'
        },
        'Random Forest': {
            'model': _create_random_forest_classifier(random_state),
            'description': '随机森林'
        },
        'Logistic Regression': {
            'model': SklearnPipeline([
                ('scaler', StandardScaler()),
                ('clf', _create_logistic_regression_classifier(random_state))
            ]),
            'description': '逻辑回归'
        },
        
        # === 神经网络方法 ===
        'MLP': {
            'model': SklearnPipeline([
                ('scaler', StandardScaler()),
                ('clf', _create_mlp_classifier(random_state))
            ]),
            'description': '多层感知机（反向传播）'
        },
    }
    
    # 尝试导入深度学习库
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        # 添加 CNN
        classifiers['CNN'] = {
            'model': 'deep_learning',
            'create_fn': lambda: _create_cnn_model(X.shape[1], len(np.unique(y))),
            'train_fn': lambda model: _train_deep_model(model, X, y, 'CNN', random_state),
            'description': '卷积神经网络'
        }
        
        # 添加 LSTM
        # 注意：LSTM 需要 3D 输入，这里暂时跳过或需要特殊处理
        # classifiers['LSTM'] = {...}
        
        has_deep_learning = True
        print("\n✅ 检测到 PyTorch，启用深度学习分类器")
    except ImportError:
        has_deep_learning = False
        print("\n⚠️  未检测到 PyTorch，跳过深度学习分类器")
        print("   如需使用 CNN/LSTM，请安装: pip install torch torchvision")
    
    # 评估每个分类器
    results = {}
    for name, config in classifiers.items():
        print(f"\n{'─' * 70}")
        print(f"测试: {name} ({config['description']})")
        print(f"{'─' * 70}")
        
        start_time = time.time()
        
        try:
            if config.get('model') == 'deep_learning':
                # 深度学习模型特殊处理
                model = config['create_fn']()
                accuracy, std = config['train_fn'](model)
                elapsed = time.time() - start_time
            else:
                # 传统机器学习模型
                model = config['model']
                scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
                accuracy = scores.mean()
                std = scores.std()
                elapsed = time.time() - start_time
            
            results[name] = {
                'accuracy': accuracy,
                'std': std,
                'time': elapsed,
                'description': config['description']
            }
            
            print(f"✅ {name:20s} | 准确率: {accuracy:.4f} ± {std:.4f} | 耗时: {elapsed:.2f}s")
            
        except Exception as e:
            print(f"❌ {name:20s} | 失败: {type(e).__name__}: {str(e)[:50]}")
            results[name] = {
                'accuracy': np.nan,
                'std': np.nan,
                'time': time.time() - start_time,
                'error': str(e)
            }
    
    # 汇总结果
    _print_comparison_table(results, feature_name)
    
    return results


def _create_svm_classifier(random_state=42):
    """创建 SVM 分类器（RBF 核）"""
    from sklearn.svm import SVC
    return SVC(kernel='rbf', C=1.0, gamma='scale', random_state=random_state)


def _create_svm_linear_classifier(random_state=42):
    """创建线性 SVM 分类器"""
    from sklearn.svm import LinearSVC
    return LinearSVC(C=1.0, random_state=random_state, max_iter=2000)


def _create_knn_classifier():
    """创建 KNN 分类器"""
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)


def _create_naive_bayes_classifier():
    """创建朴素贝叶斯分类器"""
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB()


def _create_random_forest_classifier(random_state=42):
    """创建随机森林分类器"""
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        random_state=random_state,
        n_jobs=-1
    )


def _create_logistic_regression_classifier(random_state=42):
    """创建逻辑回归分类器"""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(
        C=1.0, 
        max_iter=1000,
        random_state=random_state,
        multi_class='multinomial'
    )


def _create_mlp_classifier(random_state=42):
    """创建多层感知机（反向传播神经网络）"""
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.2
    )


def _create_cnn_model(input_dim, num_classes):
    """创建简单的 1D CNN 模型"""
    import torch
    import torch.nn as nn
    
    class SimpleCNN(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(2)
            self.fc1 = nn.Linear(64 * (input_dim // 4), 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            # x shape: (batch, features) -> (batch, 1, features)
            x = x.unsqueeze(1)
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    return SimpleCNN(input_dim, num_classes)


def _train_deep_model(model, X, y, model_name, random_state=42, epochs=50, batch_size=32):
    """训练深度学习模型并进行交叉验证"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   使用设备: {device}")

    # 标准化数据（深度学习模型需要手动标准化）
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)

    # 转换为 Tensor
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)
    
    # 简单的 k-fold 交叉验证
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # 创建数据加载器
        train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 重新初始化模型
        fold_model = type(model)(model.fc2.in_features, model.fc2.out_features).to(device)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(fold_model.parameters(), lr=0.001)
        
        # 训练
        best_val_acc = 0
        for epoch in range(epochs):
            fold_model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = fold_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # 验证
            fold_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = fold_model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            val_acc = correct / total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        accuracies.append(best_val_acc)
        print(f"   Fold {fold+1}/5: Val Acc = {best_val_acc:.4f}")
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    return mean_acc, std_acc


def _print_comparison_table(results, feature_name):
    """打印分类器对比表格"""
    print("\n" + "=" * 90)
    print(f"分类器性能对比总结 - {feature_name}")
    print("=" * 90)
    print(f"{'分类器':<25} {'准确率':>10} {'标准差':>10} {'耗时(s)':>10} {'状态':>8}")
    print("-" * 90)
    
    # 按准确率排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    best_name = None
    best_acc = 0
    
    for name, res in sorted_results:
        if np.isnan(res['accuracy']):
            status = "❌ 失败"
            acc_str = "N/A"
            std_str = "N/A"
        else:
            status = "✅"
            acc_str = f"{res['accuracy']:.4f}"
            std_str = f"{res['std']:.4f}"
            
            if res['accuracy'] > best_acc:
                best_acc = res['accuracy']
                best_name = name
        
        time_str = f"{res['time']:.2f}"
        print(f"{name:<25} {acc_str:>10} {std_str:>10} {time_str:>10} {status:>8}")
    
    print("=" * 90)
    
    if best_name:
        print(f"\n🏆 最佳分类器: {best_name}")
        print(f"   准确率: {results[best_name]['accuracy']:.4f} ± {results[best_name]['std']:.4f}")
        print(f"   耗时: {results[best_name]['time']:.2f}s")
        
        # 与 SVM 对比
        if 'SVM (RBF)' in results and not np.isnan(results['SVM (RBF)']['accuracy']):
            svm_acc = results['SVM (RBF)']['accuracy']
            diff = results[best_name]['accuracy'] - svm_acc
            if diff > 0:
                print(f"   ⚠️  注意: {best_name} 比 SVM 高 {diff*100:.2f}%")
            elif diff < 0:
                print(f"   ✅ SVM 比 {best_name} 高 {abs(diff)*100:.2f}%")
            else:
                print(f"   ✅ SVM 与 {best_name} 性能相当")
    
    print("\n💡 结论建议:")
    print("   - 如果 SVM 表现最佳或接近最佳，说明选择合理")
    print("   - 如果其他分类器显著优于 SVM，考虑更换或集成")
    print("   - 综合考虑准确率、训练时间和模型复杂度")


if __name__ == "__main__":
    # 示例：生成随机数据进行测试
    from sklearn.datasets import make_classification
    
    print("生成测试数据...")
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_classes=4,
        n_informative=15,
        random_state=42
    )
    
    results = compare_classifiers(X, y, feature_name="合成数据测试")
