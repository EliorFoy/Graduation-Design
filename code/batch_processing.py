"""
BCIC IV-2a 数据批量预处理脚本
功能：一键处理所有被试 (A01-A09) 的训练集和测试集
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, CODE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mne
import pandas as pd
import numpy as np
import traceback

from code.config import DEFAULT_CONFIG, resolve_data_path
from code.pretreatment.complete_preprocessing import complete_preprocessing_pipeline, save_processed_data


def batch_process_subjects(subject_ids=None, sessions=None, data_root=None):
    """批量处理所有被试"""
    if subject_ids is None:
        subject_ids = [f'A{i:02d}' for i in range(1, 10)]
    if sessions is None:
        sessions = ['T', 'E']
    
    print("=" * 80)
    print("开始批量预处理")
    print(f"被试列表：{subject_ids}")
    print(f"会话类型：{sessions}")
    print(f"总任务数：{len(subject_ids) * len(sessions)}")
    print("=" * 80)
    
    results = []
    output_dir = Path('./results/preprocessed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_dir = Path('./results/reports')
    report_dir.mkdir(parents=True, exist_ok=True)
    
    total_count = len(subject_ids) * len(sessions)
    current_count = 0
    
    for subj_id in subject_ids:
        for session in sessions:
            current_count += 1
            subject_session = f"{subj_id}{session}"
            
            print(f"\n进度：{current_count}/{total_count} | 处理被试：{subject_session}")
            
            try:
                data_path = resolve_data_path(subject_session, data_root=data_root, config=DEFAULT_CONFIG)
                
                if not data_path.exists():
                    print(f"数据文件不存在：{data_path}")
                    results.append({
                        '被试': subject_session,
                        '状态': '失败',
                        '试次数': 0,
                        '剔除伪迹数': 0,
                        '信噪比改善_dB': 0,
                        '错误信息': '数据文件不存在'
                    })
                    continue
                
                # 执行完整的预处理管线
                epochs_final, ica_model = complete_preprocessing_pipeline(
                    subject=subject_session,
                    data_root=data_root,
                    config=DEFAULT_CONFIG,
                    plot_comparison=False,
                )
                
                # 保存预处理后的数据
                output_path = output_dir / f'{subject_session}_epochs.fif'
                save_processed_data(epochs_final, str(output_path))
                
                # 计算指标
                snr_improvement = 0.0
                n_epochs = len(epochs_final)
                n_artifacts_removed = len(ica_model.exclude)
                
                results.append({
                    '被试': subject_session,
                    '状态': '成功',
                    '试次数': n_epochs,
                    '剔除伪迹数': n_artifacts_removed,
                    '信噪比改善_dB': round(snr_improvement, 2),
                    '错误信息': ''
                })
                
                print(f"处理完成：试次数={n_epochs}, ICA 剔除成分数={n_artifacts_removed}, SNR 改善={snr_improvement:.2f} dB")
                
            except Exception as e:
                print(f"处理失败：{str(e)}")
                traceback.print_exc()
                
                results.append({
                    '被试': subject_session,
                    '状态': '失败',
                    '试次数': 0,
                    '剔除伪迹数': 0,
                    '信噪比改善_dB': 0,
                    '错误信息': str(e)
                })
    
    print("\n生成汇总报告...")
    results_df = pd.DataFrame(results)
    
    report_path = report_dir / 'preprocessing_summary.csv'
    results_df.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"报告已保存：{report_path}")
    
    print("\n批量处理汇总:")
    n_success = len(results_df[results_df['状态'] == '成功'])
    n_failed = len(results_df[results_df['状态'] == '失败'])
    print(f"总处理数：{len(results_df)}, 成功：{n_success}, 失败：{n_failed}")
    
    if n_success > 0:
        successful_df = results_df[results_df['状态'] == '成功']
        print(f"平均试次数：{successful_df['试次数'].mean():.1f}")
        print(f"平均信噪比改善：{successful_df['信噪比改善_dB'].mean():.2f} dB")
    
    return results_df


def compute_snr_improvement(raw_original, epochs_clean):
    """计算信噪比改善"""
    # 1. 计算原始 Raw 数据的 PSD，维度: (n_channels, n_freqs)
    psd_orig = raw_original.compute_psd(fmin=0, fmax=50, verbose=False)
    psd_data_orig, freqs_orig = psd_orig.get_data(return_freqs=True)
    
    signal_mask_orig = (freqs_orig >= 8) & (freqs_orig <= 30)
    noise_mask_orig = (freqs_orig >= 40) & (freqs_orig <= 50)
    
    signal_power_orig = np.mean(psd_data_orig[:, signal_mask_orig], axis=1)
    noise_power_orig = np.mean(psd_data_orig[:, noise_mask_orig], axis=1)
    snr_orig = np.mean(10 * np.log10(signal_power_orig / (noise_power_orig + 1e-10)))
    
    # 2. 计算清洗后 Epochs 的 PSD，维度: (n_epochs, n_channels, n_freqs)
    psd_clean = epochs_clean.compute_psd(fmin=0, fmax=50, verbose=False)
    psd_data_clean, freqs_clean = psd_clean.get_data(return_freqs=True)
    
    # 将 Epochs 维度求平均，统一转换为 (n_channels, n_freqs)，解决维度不一致的 Bug
    psd_data_clean = np.mean(psd_data_clean, axis=0) 
    
    signal_mask_clean = (freqs_clean >= 8) & (freqs_clean <= 30)
    noise_mask_clean = (freqs_clean >= 40) & (freqs_clean <= 50)
    
    signal_power_clean = np.mean(psd_data_clean[:, signal_mask_clean], axis=1)
    noise_power_clean = np.mean(psd_data_clean[:, noise_mask_clean], axis=1)
    snr_clean = np.mean(10 * np.log10(signal_power_clean / (noise_power_clean + 1e-10)))
    
    snr_improvement = snr_clean - snr_orig
    return snr_improvement


if __name__ == "__main__":
    print("请选择运行模式:")
    print("1. 快速测试 (仅处理 A01T)")
    print("2. 完整处理 (所有 9 个被试)")
    
    choice = input("\n请输入选项 (1/2): ").strip()
    
    if choice == '1':
        results_df = batch_process_subjects(subject_ids=['A01'], sessions=['T'])
    else:
        results_df = batch_process_subjects(
            subject_ids=[f'A{i:02d}' for i in range(1, 10)],
            sessions=['T', 'E']
        )
    
    print("\n批量预处理全部完成！")
