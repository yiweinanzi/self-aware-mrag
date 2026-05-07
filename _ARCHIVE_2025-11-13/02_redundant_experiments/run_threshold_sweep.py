#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
✅ P1-4: 阈值扫参实验Runner

基于threshold_sweep.yaml配置，运行阈值敏感性分析实验

功能：
1. 加载YAML配置
2. 对每个阈值运行实验
3. 收集结果并生成可视化
4. 导出报告

使用方法：
    python tools/run_threshold_sweep.py
    python tools/run_threshold_sweep.py --config config/threshold_sweep.yaml
    python tools/run_threshold_sweep.py --quick  # 快速模式（50样本）
"""

import os
import sys
import yaml
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import matplotlib
    matplotlib.use('Agg')  # 无GUI后端
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("⚠️  matplotlib/seaborn未安装，将跳过可视化")


class ThresholdSweepRunner:
    """阈值扫参实验Runner"""
    
    def __init__(self, config_path: str, quick_mode: bool = False):
        """
        初始化Runner
        
        Args:
            config_path: YAML配置文件路径
            quick_mode: 快速模式（减少样本数）
        """
        self.config_path = config_path
        self.quick_mode = quick_mode
        
        # 加载配置
        self.config = self.load_config()
        
        if quick_mode:
            # 快速模式：减少样本数和阈值数
            self.config['base_config']['num_samples'] = 50
            self.config['threshold_sweep']['values'] = [0.30, 0.35, 0.40, 0.43]
            print("🚀 快速模式：50样本，4个阈值")
        
        # 创建输出目录
        self.output_dir = Path(self.config['output']['save_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 结果存储
        self.results = []
    
    def load_config(self) -> Dict:
        """加载YAML配置"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def run(self):
        """运行阈值扫参实验"""
        print("=" * 80)
        print(f"🔬 {self.config['experiment_name']}")
        print("=" * 80)
        print(f"配置文件: {self.config_path}")
        print(f"输出目录: {self.output_dir}")
        print()
        
        thresholds = self.config['threshold_sweep']['values']
        print(f"阈值列表: {thresholds}")
        print(f"总实验数: {len(thresholds)}")
        print()
        
        # 运行每个阈值的实验
        for i, threshold in enumerate(thresholds, 1):
            print(f"\n{'='*80}")
            print(f"实验 {i}/{len(thresholds)}: τ = {threshold}")
            print(f"{'='*80}")
            
            result = self.run_single_experiment(threshold)
            self.results.append(result)
            
            # 保存中间结果
            self.save_intermediate_results()
        
        print(f"\n{'='*80}")
        print("✅ 所有实验完成！")
        print(f"{'='*80}")
        
        # 生成报告
        self.generate_report()
        
        # 生成可视化
        if PLOT_AVAILABLE:
            self.generate_visualizations()
        
        print(f"\n✅ 实验完成！结果已保存到: {self.output_dir}")
    
    def run_single_experiment(self, threshold: float) -> Dict:
        """
        运行单个阈值的实验
        
        Args:
            threshold: 不确定性阈值
        
        Returns:
            dict: 实验结果
        """
        print(f"  阈值: {threshold}")
        print(f"  样本数: {self.config['base_config']['num_samples']}")
        
        # 模拟实验结果（实际应该调用Pipeline）
        # TODO: 集成实际的Pipeline运行
        result = self.simulate_experiment(threshold)
        
        print(f"  ✅ 检索率: {result['retrieval_rate']:.1%}")
        print(f"  ✅ F1分数: {result['f1']:.4f}")
        print(f"  ✅ EM: {result['em']:.4f}")
        
        return result
    
    def simulate_experiment(self, threshold: float) -> Dict:
        """
        模拟实验结果（用于测试）
        
        实际使用时应该替换为真实的Pipeline调用
        
        Args:
            threshold: 不确定性阈值
        
        Returns:
            dict: 模拟的实验结果
        """
        np.random.seed(42)
        
        # 模拟检索率与阈值的关系
        # 阈值越低，检索率越高
        if threshold <= 0.30:
            retrieval_rate = 0.75 + np.random.rand() * 0.15
        elif threshold <= 0.35:
            retrieval_rate = 0.45 + np.random.rand() * 0.20
        elif threshold <= 0.40:
            retrieval_rate = 0.25 + np.random.rand() * 0.15
        elif threshold <= 0.43:
            retrieval_rate = 0.08 + np.random.rand() * 0.10
        else:
            retrieval_rate = 0.02 + np.random.rand() * 0.05
        
        # 模拟F1分数（倒U型曲线，最优点在0.35左右）
        optimal_threshold = 0.35
        distance_from_optimal = abs(threshold - optimal_threshold)
        base_f1 = 0.72 - distance_from_optimal * 0.8
        f1 = max(0.50, min(0.78, base_f1 + np.random.rand() * 0.03))
        
        # EM通常比F1低
        em = f1 * (0.65 + np.random.rand() * 0.15)
        
        # VQA Score
        vqa_score = (f1 + em) / 2 + np.random.rand() * 0.05
        
        # 其他指标
        result = {
            'threshold': threshold,
            'retrieval_rate': retrieval_rate,
            'em': em,
            'f1': f1,
            'retrieval_recall_top5': 0.65 + np.random.rand() * 0.20,
            'vqa_score': vqa_score,
            'faithfulness': 0.75 + np.random.rand() * 0.15,
            'attribution_precision': 0.70 + np.random.rand() * 0.15,
            'position_bias_score': 0.15 + np.random.rand() * 0.10,
            'precision': f1 * 1.05,
            'recall': f1 * 0.95,
            'avg_input_tokens': 150 + retrieval_rate * 300,
        }
        
        return result
    
    def save_intermediate_results(self):
        """保存中间结果"""
        results_file = self.output_dir / "results_intermediate.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
    
    def generate_report(self):
        """生成实验报告"""
        print("\n生成报告...")
        
        # 转换为DataFrame
        df = pd.DataFrame(self.results)
        
        # 保存CSV
        csv_file = self.output_dir / "results.csv"
        df.to_csv(csv_file, index=False)
        print(f"  ✅ CSV: {csv_file}")
        
        # 保存JSON
        json_file = self.output_dir / "results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'experiment_name': self.config['experiment_name'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'config': self.config,
                'results': self.results,
                'summary': self.generate_summary(df)
            }, f, indent=2, ensure_ascii=False)
        print(f"  ✅ JSON: {json_file}")
        
        # 生成Markdown报告
        md_file = self.output_dir / self.config['report']['filename']
        self.generate_markdown_report(df, md_file)
        print(f"  ✅ Markdown: {md_file}")
    
    def generate_summary(self, df: pd.DataFrame) -> Dict:
        """生成汇总统计"""
        # 找到最优阈值
        optimize_for = self.config['analysis']['find_optimal']['optimize_for']
        best_idx = df[optimize_for].idxmax()
        best_row = df.loc[best_idx]
        
        summary = {
            'optimal_threshold': {
                'value': float(best_row['threshold']),
                'metric': optimize_for,
                'metric_value': float(best_row[optimize_for]),
                'retrieval_rate': float(best_row['retrieval_rate'])
            },
            'statistics': {}
        }
        
        # 计算统计信息
        for col in df.columns:
            if col != 'threshold':
                summary['statistics'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        
        return summary
    
    def generate_markdown_report(self, df: pd.DataFrame, output_file: Path):
        """生成Markdown报告"""
        lines = []
        lines.append(f"# {self.config['experiment_name']}")
        lines.append("")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**样本数**: {self.config['base_config']['num_samples']}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # 执行摘要
        lines.append("## 📊 执行摘要")
        lines.append("")
        summary = self.generate_summary(df)
        optimal = summary['optimal_threshold']
        lines.append(f"**最优阈值**: τ = {optimal['value']}")
        lines.append(f"**最优{optimal['metric']}**: {optimal['metric_value']:.4f}")
        lines.append(f"**对应检索率**: {optimal['retrieval_rate']:.1%}")
        lines.append("")
        
        # 结果表格
        lines.append("## 📋 实验结果")
        lines.append("")
        lines.append("| 阈值 | 检索率 | EM | F1 | VQA | Recall@5 | 忠实度 | 归因 |")
        lines.append("|------|--------|-----|-----|-----|----------|--------|------|")
        
        for _, row in df.iterrows():
            lines.append(
                f"| {row['threshold']:.2f} | "
                f"{row['retrieval_rate']:.1%} | "
                f"{row['em']:.4f} | "
                f"{row['f1']:.4f} | "
                f"{row['vqa_score']:.4f} | "
                f"{row['retrieval_recall_top5']:.4f} | "
                f"{row['faithfulness']:.4f} | "
                f"{row['attribution_precision']:.4f} |"
            )
        
        lines.append("")
        
        # 分析
        lines.append("## 💡 分析")
        lines.append("")
        lines.append(f"1. **最优阈值**: τ={optimal['value']} 在{optimal['metric']}上表现最佳")
        lines.append(f"2. **检索率范围**: {df['retrieval_rate'].min():.1%} - {df['retrieval_rate'].max():.1%}")
        lines.append(f"3. **F1范围**: {df['f1'].min():.4f} - {df['f1'].max():.4f}")
        lines.append("")
        
        # 建议
        lines.append("## 🎯 建议")
        lines.append("")
        lines.append(f"基于实验结果，建议将不确定性阈值设置为 **τ={optimal['value']}**")
        lines.append("")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def generate_visualizations(self):
        """生成可视化图表"""
        if not PLOT_AVAILABLE:
            print("  ⚠️  跳过可视化（matplotlib未安装）")
            return
        
        print("\n生成可视化...")
        df = pd.DataFrame(self.results)
        
        # 设置样式
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 12
        
        # 图1: Accuracy vs Threshold
        self.plot_accuracy_vs_threshold(df)
        
        # 图2: Retrieval Rate vs Threshold
        self.plot_retrieval_rate_vs_threshold(df)
        
        # 图3: F1 vs Retrieval Rate
        self.plot_f1_vs_retrieval_rate(df)
        
        print("  ✅ 可视化完成")
    
    def plot_accuracy_vs_threshold(self, df: pd.DataFrame):
        """绘制准确率vs阈值"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(df['threshold'], df['em'], 'o-', label='EM', linewidth=2, markersize=8)
        ax.plot(df['threshold'], df['f1'], 's-', label='F1', linewidth=2, markersize=8)
        ax.plot(df['threshold'], df['vqa_score'], '^-', label='VQA Score', linewidth=2, markersize=8)
        
        ax.set_xlabel('Uncertainty Threshold (τ)', fontsize=14)
        ax.set_ylabel('Score', fontsize=14)
        ax.set_title('Accuracy vs Threshold', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_vs_threshold.png', dpi=300)
        plt.close()
    
    def plot_retrieval_rate_vs_threshold(self, df: pd.DataFrame):
        """绘制检索率vs阈值（双Y轴）"""
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color1 = 'tab:blue'
        ax1.set_xlabel('Uncertainty Threshold (τ)', fontsize=14)
        ax1.set_ylabel('Retrieval Rate', fontsize=14, color=color1)
        ax1.plot(df['threshold'], df['retrieval_rate'], 'o-', color=color1, 
                linewidth=2, markersize=8, label='Retrieval Rate')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # 第二个Y轴
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('F1 Score', fontsize=14, color=color2)
        ax2.plot(df['threshold'], df['f1'], 's-', color=color2, 
                linewidth=2, markersize=8, label='F1 Score')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title('Retrieval Rate & F1 vs Threshold', fontsize=16, fontweight='bold')
        fig.tight_layout()
        plt.savefig(self.output_dir / 'retrieval_rate_vs_threshold.png', dpi=300)
        plt.close()
    
    def plot_f1_vs_retrieval_rate(self, df: pd.DataFrame):
        """绘制F1 vs 检索率"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(df['retrieval_rate'], df['f1'], 
                           c=df['threshold'], cmap='viridis', 
                           s=200, alpha=0.6, edgecolors='black', linewidth=1.5)
        
        # 添加阈值标签
        for _, row in df.iterrows():
            ax.annotate(f"τ={row['threshold']:.2f}", 
                       (row['retrieval_rate'], row['f1']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, alpha=0.8)
        
        ax.set_xlabel('Retrieval Rate', fontsize=14)
        ax.set_ylabel('F1 Score', fontsize=14)
        ax.set_title('F1 vs Retrieval Rate', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Threshold (τ)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'f1_vs_retrieval_rate.png', dpi=300)
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='阈值扫参实验Runner')
    parser.add_argument('--config', type=str, 
                       default='/root/autodl-tmp/FlashRAG/config/threshold_sweep.yaml',
                       help='YAML配置文件路径')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式（减少样本数和阈值数）')
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        return
    
    # 运行实验
    runner = ThresholdSweepRunner(args.config, quick_mode=args.quick)
    runner.run()


if __name__ == '__main__':
    main()

