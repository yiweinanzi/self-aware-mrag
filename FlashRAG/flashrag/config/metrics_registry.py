#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
✅ P0-5: 指标统一配置

只对外报告7项核心指标，其他指标仅供内部调试

核心指标（Primary Metrics）:
1. EM (Exact Match): 精确匹配率
2. F1: Token级F1分数
3. Recall@5: 检索召回率（top-5）
4. VQA-Score: VQA评分
5. Faithfulness: 忠实度（答案与检索文档的一致性）
6. AttributionPrecision: 归因精度（归因的准确性）
7. PositionBiasScore: 位置偏差分数（量化位置偏差缓解效果）

辅助指标（Secondary Metrics，仅内部使用）:
- Precision: Token级精确率
- Recall: Token级召回率
- Retrieval_Precision@k: 检索精确率
- Rouge-1/2/L: Rouge分数
- BLEU: BLEU分数
- Input_Tokens: 平均输入token数
"""

from typing import List, Dict, Any, Set
from dataclasses import dataclass
import warnings


@dataclass
class MetricConfig:
    """指标配置"""
    name: str
    category: str  # "primary" 或 "secondary"
    description: str
    display_name: str
    higher_is_better: bool = True
    format_str: str = ".4f"  # 格式化字符串
    
    def format_value(self, value: float) -> str:
        """格式化指标值"""
        return f"{value:{self.format_str}}"


# ========================================================================
# 核心指标（Primary Metrics）- 对外报告
# ========================================================================

PRIMARY_METRICS = {
    "em": MetricConfig(
        name="em",
        category="primary",
        description="精确匹配率 - 预测答案与标准答案完全一致的比例",
        display_name="EM",
        higher_is_better=True,
        format_str=".4f"
    ),
    
    "f1": MetricConfig(
        name="f1",
        category="primary",
        description="Token级F1分数 - 平衡精确率和召回率",
        display_name="F1",
        higher_is_better=True,
        format_str=".4f"
    ),
    
    "retrieval_recall_top5": MetricConfig(
        name="retrieval_recall_top5",
        category="primary",
        description="检索召回率@5 - 查询级macro平均，top-5检索的召回率",
        display_name="Recall@5",
        higher_is_better=True,
        format_str=".4f"
    ),
    
    "vqa_score": MetricConfig(
        name="vqa_score",
        category="primary",
        description="VQA评分 - 视觉问答任务的标准评分",
        display_name="VQA-Score",
        higher_is_better=True,
        format_str=".4f"
    ),
    
    "faithfulness": MetricConfig(
        name="faithfulness",
        category="primary",
        description="忠实度 - 生成答案与检索文档的一致性",
        display_name="Faithfulness",
        higher_is_better=True,
        format_str=".4f"
    ),
    
    "attribution_precision": MetricConfig(
        name="attribution_precision",
        category="primary",
        description="归因精度 - 细粒度归因的准确性",
        display_name="Attribution",
        higher_is_better=True,
        format_str=".4f"
    ),
    
    "position_bias_score": MetricConfig(
        name="position_bias_score",
        category="primary",
        description="位置偏差分数 - 量化位置偏差缓解效果（越低越好）",
        display_name="PosBias",
        higher_is_better=False,  # 注意：越低越好
        format_str=".4f"
    ),
}


# ========================================================================
# 辅助指标（Secondary Metrics）- 仅内部使用
# ========================================================================

SECONDARY_METRICS = {
    "precision": MetricConfig(
        name="precision",
        category="secondary",
        description="Token级精确率",
        display_name="Precision",
        higher_is_better=True,
        format_str=".4f"
    ),
    
    "recall": MetricConfig(
        name="recall",
        category="secondary",
        description="Token级召回率",
        display_name="Recall",
        higher_is_better=True,
        format_str=".4f"
    ),
    
    "retrieval_precision_top5": MetricConfig(
        name="retrieval_precision_top5",
        category="secondary",
        description="检索精确率@5",
        display_name="Ret-Prec@5",
        higher_is_better=True,
        format_str=".4f"
    ),
    
    "rouge-1": MetricConfig(
        name="rouge-1",
        category="secondary",
        description="Rouge-1分数",
        display_name="Rouge-1",
        higher_is_better=True,
        format_str=".4f"
    ),
    
    "rouge-2": MetricConfig(
        name="rouge-2",
        category="secondary",
        description="Rouge-2分数",
        display_name="Rouge-2",
        higher_is_better=True,
        format_str=".4f"
    ),
    
    "rouge-l": MetricConfig(
        name="rouge-l",
        category="secondary",
        description="Rouge-L分数",
        display_name="Rouge-L",
        higher_is_better=True,
        format_str=".4f"
    ),
    
    "bleu": MetricConfig(
        name="bleu",
        category="secondary",
        description="BLEU分数",
        display_name="BLEU",
        higher_is_better=True,
        format_str=".4f"
    ),
    
    "avg_input_tokens": MetricConfig(
        name="avg_input_tokens",
        category="secondary",
        description="平均输入token数（用于效率分析）",
        display_name="Avg-Tokens",
        higher_is_better=False,
        format_str=".1f"
    ),
}


# ========================================================================
# 指标注册表管理器
# ========================================================================

class MetricsRegistry:
    """
    指标注册表管理器
    
    功能：
    1. 管理核心/辅助指标
    2. 过滤报告指标
    3. 格式化指标输出
    4. 生成指标报告
    
    使用示例：
    ```python
    registry = MetricsRegistry()
    
    # 获取核心指标列表
    primary = registry.get_primary_metrics()
    
    # 过滤结果（只保留核心指标）
    filtered = registry.filter_results(all_results)
    
    # 格式化输出
    report = registry.format_report(results)
    ```
    """
    
    def __init__(self):
        self.primary_metrics = PRIMARY_METRICS
        self.secondary_metrics = SECONDARY_METRICS
        
        print("=" * 80)
        print("📊 指标注册表初始化")
        print("=" * 80)
        print(f"✅ 核心指标（对外报告）: {len(self.primary_metrics)}项")
        for name, config in self.primary_metrics.items():
            direction = "↑" if config.higher_is_better else "↓"
            print(f"   {direction} {config.display_name:15s} ({name}): {config.description}")
        print(f"ℹ️  辅助指标（内部使用）: {len(self.secondary_metrics)}项")
        for name, config in self.secondary_metrics.items():
            print(f"     {config.display_name:15s} ({name})")
        print("=" * 80)
    
    def get_primary_metrics(self) -> Dict[str, MetricConfig]:
        """获取核心指标"""
        return self.primary_metrics
    
    def get_secondary_metrics(self) -> Dict[str, MetricConfig]:
        """获取辅助指标"""
        return self.secondary_metrics
    
    def get_primary_metric_names(self) -> List[str]:
        """获取核心指标名称列表"""
        return list(self.primary_metrics.keys())
    
    def is_primary_metric(self, metric_name: str) -> bool:
        """判断是否为核心指标"""
        return metric_name in self.primary_metrics
    
    def filter_results(self, results: Dict[str, float], mode: str = "primary") -> Dict[str, float]:
        """
        过滤结果，只保留指定类别的指标
        
        Args:
            results: 所有指标结果
            mode: "primary"（只保留核心指标）或 "all"（保留所有）
        
        Returns:
            Dict[str, float]: 过滤后的结果
        """
        if mode == "all":
            return results
        
        filtered = {}
        for metric_name, value in results.items():
            if metric_name in self.primary_metrics:
                filtered[metric_name] = value
            elif mode == "primary":
                # 核心指标模式下，跳过辅助指标
                pass
        
        return filtered
    
    def format_value(self, metric_name: str, value: float) -> str:
        """格式化指标值"""
        config = self.primary_metrics.get(metric_name) or self.secondary_metrics.get(metric_name)
        
        if config:
            return config.format_value(value)
        else:
            return f"{value:.4f}"
    
    def format_report(self, results: Dict[str, float], mode: str = "primary") -> str:
        """
        格式化指标报告
        
        Args:
            results: 指标结果
            mode: "primary"（只报告核心指标）或 "all"（报告所有）
        
        Returns:
            str: 格式化的报告文本
        """
        filtered = self.filter_results(results, mode)
        
        lines = []
        lines.append("=" * 80)
        lines.append("📊 指标报告" + (" (核心指标)" if mode == "primary" else " (所有指标)"))
        lines.append("=" * 80)
        
        # 按类别分组
        if mode == "primary" or mode == "all":
            lines.append("\n✅ 核心指标:")
            for name in self.get_primary_metric_names():
                if name in filtered:
                    config = self.primary_metrics[name]
                    value_str = self.format_value(name, filtered[name])
                    direction = "↑" if config.higher_is_better else "↓"
                    lines.append(f"   {direction} {config.display_name:18s}: {value_str}")
        
        if mode == "all":
            lines.append("\nℹ️  辅助指标:")
            for name, config in self.secondary_metrics.items():
                if name in filtered:
                    value_str = self.format_value(name, filtered[name])
                    lines.append(f"     {config.display_name:18s}: {value_str}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def export_csv_header(self, mode: str = "primary") -> str:
        """导出CSV表头"""
        if mode == "primary":
            names = [config.display_name for config in self.primary_metrics.values()]
        else:
            names = [config.display_name for config in {**self.primary_metrics, **self.secondary_metrics}.values()]
        
        return ",".join(names)
    
    def export_csv_row(self, results: Dict[str, float], mode: str = "primary") -> str:
        """导出CSV行"""
        filtered = self.filter_results(results, mode)
        
        if mode == "primary":
            metric_names = self.get_primary_metric_names()
        else:
            metric_names = list({**self.primary_metrics, **self.secondary_metrics}.keys())
        
        values = []
        for name in metric_names:
            if name in filtered:
                values.append(self.format_value(name, filtered[name]))
            else:
                values.append("N/A")
        
        return ",".join(values)


# ========================================================================
# 全局单例
# ========================================================================

_metrics_registry_instance = None

def get_metrics_registry() -> MetricsRegistry:
    """获取指标注册表单例"""
    global _metrics_registry_instance
    if _metrics_registry_instance is None:
        _metrics_registry_instance = MetricsRegistry()
    return _metrics_registry_instance


# ========================================================================
# 便捷函数
# ========================================================================

def get_primary_metric_names() -> List[str]:
    """获取核心指标名称列表（便捷函数）"""
    registry = get_metrics_registry()
    return registry.get_primary_metric_names()


def filter_primary_metrics(results: Dict[str, float]) -> Dict[str, float]:
    """过滤核心指标（便捷函数）"""
    registry = get_metrics_registry()
    return registry.filter_results(results, mode="primary")


def format_metrics_report(results: Dict[str, float], mode: str = "primary") -> str:
    """格式化指标报告（便捷函数）"""
    registry = get_metrics_registry()
    return registry.format_report(results, mode)


if __name__ == '__main__':
    print("指标注册表测试")
    print("=" * 80)
    
    # 创建注册表
    registry = MetricsRegistry()
    
    # 模拟结果
    mock_results = {
        "em": 0.4567,
        "f1": 0.6123,
        "retrieval_recall_top5": 0.7234,
        "vqa_score": 0.5678,
        "faithfulness": 0.8123,
        "attribution_precision": 0.7456,
        "position_bias_score": 0.0234,  # 越低越好
        "precision": 0.6345,  # 辅助指标
        "recall": 0.5901,  # 辅助指标
        "rouge-1": 0.5123,  # 辅助指标
    }
    
    print("\n" + "=" * 80)
    print("测试：过滤核心指标")
    print("=" * 80)
    filtered = registry.filter_results(mock_results, mode="primary")
    print(f"原始指标数: {len(mock_results)}")
    print(f"核心指标数: {len(filtered)}")
    print(f"核心指标: {list(filtered.keys())}")
    
    print("\n" + registry.format_report(mock_results, mode="primary"))
    
    print("\n" + registry.format_report(mock_results, mode="all"))
    
    print("\n" + "=" * 80)
    print("CSV导出:")
    print("=" * 80)
    print("Header:", registry.export_csv_header(mode="primary"))
    print("Row:   ", registry.export_csv_row(mock_results, mode="primary"))
    
    print("=" * 80)

