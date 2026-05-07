#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
✅ P0-4: 评测数据集注册表

固定四个核心评测数据集，禁用InfoSeek/WebQA

核心数据集：
1. OK-VQA: 外部知识VQA基准
2. A-OKVQA: 可解释的知识VQA
3. MultiModalQA: 多跳多模态QA
4. MRAG-Bench: 多模态RAG综合基准

禁用数据集：
- InfoSeek: 部分下载，但不用于评测
- WebQA: 加载器未实现，暂不支持
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import warnings


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str
    enabled: bool
    description: str
    path: Optional[str] = None
    loader: Optional[str] = None
    num_samples: Optional[int] = None
    version: Optional[str] = None
    
    def __post_init__(self):
        if not self.enabled:
            warnings.warn(f"数据集 {self.name} 已禁用: {self.description}")


# ========================================================================
# 核心评测数据集（已启用）
# ========================================================================

ENABLED_DATASETS = {
    "okvqa": DatasetConfig(
        name="OK-VQA",
        enabled=True,
        description="Outside Knowledge VQA - 需要外部知识的视觉问答基准",
        path="/root/autodl-tmp/FlashRAG/datasets/okvqa",
        loader="flashrag.dataset.okvqa_loader",
        version="v1.0",
        num_samples=5046  # 验证集样本数
    ),
    
    "aokvqa": DatasetConfig(
        name="A-OKVQA",
        enabled=True,
        description="Augmented OK-VQA - 可解释的知识增强VQA",
        path="/root/autodl-tmp/FlashRAG/datasets/aokvqa",
        loader="flashrag.dataset.aokvqa_loader",
        version="v1.1",
        num_samples=6702  # 验证集样本数
    ),
    
    "multimodalqa": DatasetConfig(
        name="MultiModalQA",
        enabled=True,
        description="多跳多模态问答 - 需要跨模态推理",
        path="/root/autodl-tmp/FlashRAG/datasets/multimodalqa",
        loader="flashrag.dataset.multimodalqa_loader",
        version="v1.0",
        num_samples=6701  # 开发集样本数
    ),
    
    "mragbench": DatasetConfig(
        name="MRAG-Bench",
        enabled=True,
        description="多模态RAG综合基准 - 11个任务的综合评测",
        path="/root/autodl-tmp/FlashRAG/datasets/mragbench",
        loader="flashrag.dataset.mragbench_loader",
        version="v1.0",
        num_samples=6153  # 总样本数（11个任务）
    ),
}


# ========================================================================
# 禁用数据集（不用于评测）
# ========================================================================

DISABLED_DATASETS = {
    "infoseek": DatasetConfig(
        name="InfoSeek",
        enabled=False,
        description="禁用原因：部分下载，但不用于核心评测",
        path="/root/autodl-tmp/FlashRAG/datasets/infoseek",
        loader=None,
        version=None,
        num_samples=None
    ),
    
    "webqa": DatasetConfig(
        name="WebQA",
        enabled=False,
        description="禁用原因：加载器未实现，暂不支持",
        path=None,
        loader=None,
        version=None,
        num_samples=None
    ),
}


# ========================================================================
# 数据集注册表管理器
# ========================================================================

class DatasetRegistry:
    """
    数据集注册表管理器
    
    功能：
    1. 管理启用/禁用的数据集
    2. 提供数据集查询接口
    3. 导出数据集清单
    4. 验证数据集配置
    
    使用示例：
    ```python
    registry = DatasetRegistry()
    
    # 获取所有启用的数据集
    enabled = registry.get_enabled_datasets()
    
    # 检查数据集是否启用
    if registry.is_enabled("okvqa"):
        ...
    
    # 获取数据集配置
    config = registry.get_config("okvqa")
    
    # 导出数据集清单
    manifest = registry.export_manifest()
    ```
    """
    
    def __init__(self):
        self.enabled_datasets = ENABLED_DATASETS
        self.disabled_datasets = DISABLED_DATASETS
        
        print("=" * 80)
        print("📋 数据集注册表初始化")
        print("=" * 80)
        print(f"✅ 启用数据集: {len(self.enabled_datasets)}个")
        for name, config in self.enabled_datasets.items():
            print(f"   - {config.name:20s} ({name})")
        print(f"❌ 禁用数据集: {len(self.disabled_datasets)}个")
        for name, config in self.disabled_datasets.items():
            print(f"   - {config.name:20s} ({name}): {config.description}")
        print("=" * 80)
    
    def get_enabled_datasets(self) -> Dict[str, DatasetConfig]:
        """获取所有启用的数据集"""
        return self.enabled_datasets
    
    def get_disabled_datasets(self) -> Dict[str, DatasetConfig]:
        """获取所有禁用的数据集"""
        return self.disabled_datasets
    
    def is_enabled(self, dataset_name: str) -> bool:
        """检查数据集是否启用"""
        return dataset_name.lower() in self.enabled_datasets
    
    def get_config(self, dataset_name: str) -> Optional[DatasetConfig]:
        """获取数据集配置"""
        dataset_name = dataset_name.lower()
        
        if dataset_name in self.enabled_datasets:
            return self.enabled_datasets[dataset_name]
        elif dataset_name in self.disabled_datasets:
            return self.disabled_datasets[dataset_name]
        else:
            return None
    
    def validate_dataset(self, dataset_name: str) -> bool:
        """验证数据集是否可用"""
        config = self.get_config(dataset_name)
        
        if config is None:
            warnings.warn(f"数据集 {dataset_name} 未在注册表中")
            return False
        
        if not config.enabled:
            warnings.warn(f"数据集 {config.name} 已禁用: {config.description}")
            return False
        
        # 检查路径
        if config.path:
            import os
            if not os.path.exists(config.path):
                warnings.warn(f"数据集路径不存在: {config.path}")
                return False
        
        return True
    
    def export_manifest(self) -> List[Dict[str, Any]]:
        """
        导出数据集清单（用于P1-3任务）
        
        Returns:
            List[Dict]: 数据集清单列表
        """
        manifest = []
        
        for name, config in {**self.enabled_datasets, **self.disabled_datasets}.items():
            manifest.append({
                "dataset_id": name,
                "name": config.name,
                "enabled": config.enabled,
                "description": config.description,
                "path": config.path,
                "loader": config.loader,
                "version": config.version,
                "num_samples": config.num_samples,
            })
        
        return manifest
    
    def get_enabled_names(self) -> List[str]:
        """获取所有启用的数据集名称列表"""
        return list(self.enabled_datasets.keys())
    
    def filter_datasets(self, dataset_list: List[str]) -> List[str]:
        """
        过滤数据集列表，只保留启用的数据集
        
        Args:
            dataset_list: 数据集名称列表
        
        Returns:
            List[str]: 过滤后的数据集列表（只包含启用的）
        """
        filtered = []
        
        for name in dataset_list:
            name_lower = name.lower()
            if name_lower in self.enabled_datasets:
                filtered.append(name)
            elif name_lower in self.disabled_datasets:
                print(f"⚠️  跳过禁用数据集: {name}")
            else:
                print(f"⚠️  未知数据集: {name}")
        
        return filtered


# ========================================================================
# 全局单例
# ========================================================================

_registry_instance = None

def get_dataset_registry() -> DatasetRegistry:
    """获取数据集注册表单例"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = DatasetRegistry()
    return _registry_instance


# ========================================================================
# 便捷函数
# ========================================================================

def is_dataset_enabled(dataset_name: str) -> bool:
    """检查数据集是否启用（便捷函数）"""
    registry = get_dataset_registry()
    return registry.is_enabled(dataset_name)


def get_enabled_dataset_names() -> List[str]:
    """获取所有启用的数据集名称（便捷函数）"""
    registry = get_dataset_registry()
    return registry.get_enabled_names()


def validate_dataset(dataset_name: str) -> bool:
    """验证数据集（便捷函数）"""
    registry = get_dataset_registry()
    return registry.validate_dataset(dataset_name)


if __name__ == '__main__':
    print("数据集注册表测试")
    print("=" * 80)
    
    # 创建注册表
    registry = DatasetRegistry()
    
    # 测试查询
    print("\n测试查询:")
    for name in ["okvqa", "infoseek", "webqa", "unknown"]:
        enabled = registry.is_enabled(name)
        config = registry.get_config(name)
        print(f"  {name:15s}: 启用={str(enabled):5s}, 配置={config is not None}")
    
    # 导出清单
    print("\n导出清单:")
    manifest = registry.export_manifest()
    print(f"  总数据集: {len(manifest)}个")
    print(f"  启用: {sum(1 for m in manifest if m['enabled'])}个")
    print(f"  禁用: {sum(1 for m in manifest if not m['enabled'])}个")
    
    # 过滤测试
    print("\n过滤测试:")
    test_list = ["okvqa", "infoseek", "aokvqa", "webqa", "unknown"]
    filtered = registry.filter_datasets(test_list)
    print(f"  原列表: {test_list}")
    print(f"  过滤后: {filtered}")
    
    print("=" * 80)

