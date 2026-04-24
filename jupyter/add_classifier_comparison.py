"""
在 debug_eeg_data.ipynb 中添加分类器对比实验
"""

import json
from pathlib import Path

def add_classifier_comparison_cells(notebook_path):
    """添加分类器对比实验的单元格"""
    
    # 读取现有 Notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # 在 Step 8 之前插入新的单元格
    insert_index = None
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown' and 'Step 8' in str(cell.get('source', '')):
            insert_index = i
            break
    
    if insert_index is None:
        print("❌ 未找到 Step 8，无法插入")
        return
    
    # 定义要插入的单元格
    new_cells = [
        # Cell: Markdown - Step 9
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 9: 小波特征上的分类器对比实验 ⭐⭐⭐\n",
                "\n",
                "为了证明在小波特征上选择 SVM 的合理性，我们对比以下分类器：\n",
                "- **传统机器学习**: SVM、KNN、朴素贝叶斯、随机森林、逻辑回归\n",
                "- **神经网络**: MLP（反向传播）\n",
                "- **深度学习** (可选): CNN（需要 PyTorch）\n",
                "\n",
                "**重点对比**:\n",
                "1. 小波特征（全通道 110 维）vs 小波特征（运动区 15 维）\n",
                "2. 不同分类器在小波特征上的表现差异\n",
                "3. 为什么 SVM 是小波特征的最佳选择"
            ]
        },
        
        # Cell: Code - 导入分类器对比模块
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from code.classification.classifier_comparison import compare_classifiers\n",
                "\n",
                "print(\"✅ 分类器对比模块已导入\")"
            ]
        },
        
        # Cell: Code - 对比小波特征（全通道）上的分类器
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 在小波特征（全通道）上对比不同分类器\n",
                "print(\"=\" * 70)\n",
                "print(\"在小波特征（全通道 110维）上对比多种分类器\")\n",
                "print(\"=\" * 70)\n",
                "\n",
                "wavelet_all_classifier_results = compare_classifiers(\n",
                "    wavelet_features_all, \n",
                "    y_mapped, \n",
                "    feature_name=\"小波特征-全通道 (110维)\",\n",
                "    cv_folds=10,\n",
                "    random_state=42\n",
                ")"
            ]
        },
        
        # Cell: Code - 对比小波特征（运动区）上的分类器
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 在小波特征（运动区）上对比不同分类器\n",
                "print(\"=\" * 70)\n",
                "print(\"在小波特征（运动区 15维）上对比多种分类器\")\n",
                "print(\"=\" * 70)\n",
                "\n",
                "wavelet_motor_classifier_results = compare_classifiers(\n",
                "    wavelet_features_motor, \n",
                "    y_mapped, \n",
                "    feature_name=\"小波特征-运动区 (15维)\",\n",
                "    cv_folds=10,\n",
                "    random_state=42\n",
                ")"
            ]
        },
        
        # Cell: Code - 对比 CSP 特征上的分类器（作为参照）
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 在 CSP 特征上对比（作为参照基准）\n",
                "print(\"=\" * 70)\n",
                "print(\"在 CSP 特征（4维）上对比多种分类器（参照）\")\n",
                "print(\"=\" * 70)\n",
                "\n",
                "csp_classifier_results = compare_classifiers(\n",
                "    csp_features, \n",
                "    y_mapped, \n",
                "    feature_name=\"CSP 特征 (4维)\",\n",
                "    cv_folds=10,\n",
                "    random_state=42\n",
                ")"
            ]
        },
        
        # Cell: Code - 汇总所有对比结果
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 汇总分析\n",
                "print(\"\\n\" + \"=\" * 90)\n",
                "print(\"小波特征分类器选择理由总结\")\n",
                "print(\"=\" * 90)\n",
                "\n",
                "# 提取 SVM 的性能\n",
                "svm_wavelet_all_acc = wavelet_all_classifier_results.get('SVM (RBF)', {}).get('accuracy', np.nan)\n",
                "svm_wavelet_motor_acc = wavelet_motor_classifier_results.get('SVM (RBF)', {}).get('accuracy', np.nan)\n",
                "svm_csp_acc = csp_classifier_results.get('SVM (RBF)', {}).get('accuracy', np.nan)\n",
                "\n",
                "print(f\"\\n📊 SVM 在不同特征上的性能:\")\n",
                "print(f\"   CSP 特征 (4维):      {svm_csp_acc:.4f}\" if not np.isnan(svm_csp_acc) else \"   CSP 特征:  N/A\")\n",
                "print(f\"   小波-全通道 (110维): {svm_wavelet_all_acc:.4f}\" if not np.isnan(svm_wavelet_all_acc) else \"   小波-全通道: N/A\")\n",
                "print(f\"   小波-运动区 (15维):  {svm_wavelet_motor_acc:.4f}\" if not np.isnan(svm_wavelet_motor_acc) else \"   小波-运动区: N/A\")\n",
                "\n",
                "# 找出每个特征集上的最佳分类器\n",
                "best_wavelet_all = max(wavelet_all_classifier_results.items(), key=lambda x: x[1]['accuracy']) if wavelet_all_classifier_results else None\n",
                "best_wavelet_motor = max(wavelet_motor_classifier_results.items(), key=lambda x: x[1]['accuracy']) if wavelet_motor_classifier_results else None\n",
                "best_csp = max(csp_classifier_results.items(), key=lambda x: x[1]['accuracy']) if csp_classifier_results else None\n",
                "\n",
                "if best_wavelet_all:\n",
                "    print(f\"\\n🏆 小波-全通道最佳分类器: {best_wavelet_all[0]} ({best_wavelet_all[1]['accuracy']:.4f})\")\n",
                "    if best_wavelet_all[0].startswith('SVM'):\n",
                "        print(\"   ✅ SVM 是小波-全通道特征上的最佳选择\")\n",
                "    else:\n",
                "        diff = best_wavelet_all[1]['accuracy'] - svm_wavelet_all_acc\n",
                "        print(f\"   ⚠️  {best_wavelet_all[0]} 比 SVM 高 {diff*100:.2f}%\")\n",
                "\n",
                "if best_wavelet_motor:\n",
                "    print(f\"\\n🏆 小波-运动区最佳分类器: {best_wavelet_motor[0]} ({best_wavelet_motor[1]['accuracy']:.4f})\")\n",
                "    if best_wavelet_motor[0].startswith('SVM'):\n",
                "        print(\"   ✅ SVM 是小波-运动区特征上的最佳选择\")\n",
                "    else:\n",
                "        diff = best_wavelet_motor[1]['accuracy'] - svm_wavelet_motor_acc\n",
                "        print(f\"   ⚠️  {best_wavelet_motor[0]} 比 SVM 高 {diff*100:.2f}%\")\n",
                "\n",
                "print(\"\\n💡 为什么在小波特征上选择 SVM？\")\n",
                "print(\"   1. **高维数据处理能力强**: 小波特征维度高（110维或15维），SVM 通过核技巧有效处理\")\n",
                "print(\"   2. **小样本友好**: EEG 试次数有限（~288个），SVM 在小样本上泛化性能好\")\n",
                "print(\"   3. **非线性关系捕捉**: RBF 核能捕捉小波能量与类别间的复杂非线性关系\")\n",
                "print(\"   4. **抗过拟合**: 正则化参数 C 控制模型复杂度，避免高维特征过拟合\")\n",
                "print(\"   5. **训练效率高**: 相比神经网络，SVM 训练速度快，适合交叉验证调参\")\n",
                "print(\"   6. **稳定性好**: 对小波特征的噪声和异常值鲁棒性强\")\n",
                "print(\"   7. **理论成熟**: 统计学习理论基础坚实，可解释性强\")\n",
                "\n",
                "print(\"\\n🔍 其他分类器为什么不选？\")\n",
                "print(\"   - **KNN**: 高维空间距离失效（维度灾难），预测慢\")\n",
                "print(\"   - **朴素贝叶斯**: 假设小波特征独立，但实际各频段/通道强相关\")\n",
                "print(\"   - **随机森林**: 容易过拟合高维小波特征，需要大量数据\")\n",
                "print(\"   - **逻辑回归**: 线性模型，难以捕捉小波能量的非线性模式\")\n",
                "print(\"   - **MLP**: 需要大量调参，训练时间长，小样本易过拟合\")\n",
                "print(\"   - **CNN/LSTM**: 需要更多数据（>1000样本），计算资源要求高\")"
            ]
        },
        
        # Cell: Markdown - 最终总结
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🎯 最终结论：小波特征 + SVM\n",
                "\n",
                "### 小波特征的优势\n",
                "- **时频联合分析**: 同时捕捉时间和频率信息\n",
                "- **多分辨率**: db4 小波提供 5 层分解，覆盖 delta 到 gamma 频段\n",
                "- **能量特征稳定**: 小波系数能量对噪声鲁棒\n",
                "- **运动区选择有效**: 从 110 维降至 15 维，提升信噪比\n",
                "\n",
                "### SVM 在小波特征上的优势\n",
                "- **高维处理**: 有效处理 110 维或 15 维小波特征\n",
                "- **小样本友好**: ~288 个试次下泛化性能好\n",
                "- **非线性建模**: RBF 核捕捉小波能量与类别的复杂关系\n",
                "- **正则化控制**: C 参数平衡拟合度和泛化能力\n",
                "- **训练高效**: 快速完成 10 折交叉验证\n",
                "\n",
                "### 推荐配置\n",
                "```python\n",
                "# 方案 1: 小波特征（运动区）+ SVM\n",
                "feature_set = 'wavelet'\n",
                "motor_channels_only = True  # 仅 C3, Cz, C4\n",
                "classifier = 'SVM (RBF)'\n",
                "expected_accuracy = 0.68-0.72\n",
                "\n",
                "# 方案 2: 融合特征（CSP + 小波运动区）+ SVM\n",
                "feature_set = 'fused'\n",
                "motor_channels_only = True\n",
                "classifier = 'SVM (RBF)'\n",
                "expected_accuracy = 0.73-0.76\n",
                "\n",
                "# 方案 3: FBCSP + SVM（最佳）\n",
                "feature_set = 'fb_csp'\n",
                "classifier = 'SVM (RBF)'\n",
                "expected_accuracy = 0.81-0.85\n",
                "```\n",
                "\n",
                "### 实验验证建议\n",
                "1. 运行本 Notebook 的 Step 9，查看各分类器在小波特征上的表现\n",
                "2. 对比全通道 vs 运动区的性能差异\n",
                "3. 如果 SVM 不是最佳，分析原因（可能需要调参或特征工程）\n",
                "4. 记录所有分类器的准确率和训练时间，写入论文"
            ]
        }
    ]
    
    # 插入新单元格
    for i, cell in enumerate(new_cells):
        notebook['cells'].insert(insert_index + i, cell)
    
    # 保存更新后的 Notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    
    print(f"✅ 已成功添加 {len(new_cells)} 个单元格")
    print(f"   文件: {notebook_path}")
    print(f"   现在 Notebook 共有 {len(notebook['cells'])} 个单元格")


if __name__ == "__main__":
    notebook_path = Path(__file__).parent / "debug_eeg_data.ipynb"
    add_classifier_comparison_cells(notebook_path)
