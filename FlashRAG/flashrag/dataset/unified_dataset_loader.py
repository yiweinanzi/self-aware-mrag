#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一数据集加载器 - 支持四个数据集
Unified Dataset Loader for Four Datasets

支持的数据集：
1. OK-VQA - 原始OK-VQA数据集
2. A-OKVQA - A-OKVQA数据集（包含推理链）
3. MultiModalQA - 多模态问答数据集
4. MRAG-Bench - 多模态RAG评测基准

所有数据集统一转换为相同的格式，便于使用相同的评测指标
"""

import os
import sys
import json
import gzip
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import warnings

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

class UnifiedDataset:
    """统一数据集格式"""

    def __init__(self, name: str, data: List[Dict], config: Dict = None):
        self.name = name
        self.data = data
        self.config = config or {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_stats(self):
        """获取数据集统计信息"""
        if not self.data:
            return {}

        stats = {
            'total_samples': len(self.data),
            'has_images': 0,
            'has_multiple_choice': 0,
            'avg_question_length': 0,
            'avg_answer_length': 0
        }

        total_q_len = 0
        total_a_len = 0

        for item in self.data:
            if item.get('image') or item.get('image_path'):
                stats['has_images'] += 1

            if 'choices' in item or ('A' in item and 'B' in item):
                stats['has_multiple_choice'] += 1

            question = item.get('question', '')
            answers = item.get('golden_answers', [])

            total_q_len += len(question.split())
            if answers:
                if isinstance(answers, list):
                    total_a_len += sum(len(str(a).split()) for a in answers[:3])
                else:
                    total_a_len += len(str(answers).split())

        if len(self.data) > 0:
            stats['avg_question_length'] = total_q_len / len(self.data)
            stats['avg_answer_length'] = total_a_len / len(self.data)

        return stats


class UnifiedDatasetLoader:
    """统一数据集加载器"""

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # 数据集路径配置
        self.dataset_paths = {
            'okvqa': {
                'data_path': '/data1/userdata/zqwang/ACL_data/OK-VQA',
                'image_path': '/data1/userdata/zqwang/ACL_data/OK-VQA/images'
            },
            'a-okvqa': {
                'data_path': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/A-OKVQA',
                'image_path': '/data1/userdata/zqwang/ACL_data/COCO/train2014'
            },
            'multimodalqa': {
                'data_path': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA',
                'image_path': None  # 图像路径在数据中指定
            },
            'mrag-bench': {
                'data_path': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench',
                'image_path': '/data1/userdata/zqwang/ACL_data/MRAG-Bench/images'
            }
        }

    def load_dataset(self, dataset_name: str, split: str = 'val', max_samples: Optional[int] = None) -> UnifiedDataset:
        """加载指定数据集"""

        if dataset_name not in self.dataset_paths:
            raise ValueError(f"不支持的数据集: {dataset_name}")

        print(f"\n加载 {dataset_name.upper()} 数据集 ({split})...")

        if dataset_name == 'okvqa':
            data = self._load_okvqa(split)
        elif dataset_name == 'a-okvqa':
            data = self._load_a_okvqa(split)
        elif dataset_name == 'multimodalqa':
            data = self._load_multimodalqa(split)
        elif dataset_name == 'mrag-bench':
            data = self._load_mrag_bench(split)
        else:
            raise ValueError(f"未知数据集: {dataset_name}")

        # 限制样本数
        if max_samples and len(data) > max_samples:
            data = data[:max_samples]

        dataset = UnifiedDataset(dataset_name, data, {
            'split': split,
            'max_samples': max_samples
        })

        # 打印统计信息
        stats = dataset.get_stats()
        print(f"✅ {dataset_name.upper()} 加载完成:")
        print(f"   样本数: {stats['total_samples']}")
        print(f"   包含图像: {stats['has_images']}")
        print(f"   多选题: {stats['has_multiple_choice']}")
        print(f"   平均问题长度: {stats['avg_question_length']:.1f} 词")
        print(f"   平均答案长度: {stats['avg_answer_length']:.1f} 词")

        return dataset

    def _load_okvqa(self, split: str) -> List[Dict]:
        """加载OK-VQA数据集"""
        from flashrag.dataset.okvqa_dataset import load_okvqa_dataset

        try:
            raw_data = load_okvqa_dataset(split=split, max_samples=None)
            data = []

            for item in raw_data:
                # 转换为统一格式
                sample = {
                    'id': item.get('question_id', ''),
                    'question': item['question'],
                    'golden_answers': item.get('golden_answers', []),
                    'image': item.get('image'),
                    'image_path': item.get('image_path', ''),
                    'dataset': 'okvqa'
                }
                data.append(sample)

            return data

        except Exception as e:
            print(f"⚠️ OK-VQA加载失败: {e}")
            # 返回示例数据
            return self._get_dummy_data('okvqa', 100)

    def _load_a_okvqa(self, split: str) -> List[Dict]:
        """加载A-OKVQA数据集"""
        try:
            # A-OKVQA数据文件路径
            data_file = Path(self.dataset_paths['a-okvqa']['data_path']) / f'aokvqa_{split}.json'

            if not data_file.exists():
                print(f"⚠️ A-OKVQA数据文件不存在: {data_file}")
                return self._get_dummy_data('a-okvqa', 100)

            with open(data_file, 'r') as f:
                raw_data = json.load(f)

            data = []
            for item in raw_data:
                # 转换为统一格式
                sample = {
                    'id': item.get('question_id', ''),
                    'question': item['question'],
                    'golden_answers': item.get('choices', []) if 'choices' in item else [item.get('answer', '')],
                    'image_path': self._get_aokvqa_image_path(item.get('image_id', '')),
                    'rationale': item.get('rationale', ''),  # A-OKVQA特有的推理链
                    'dataset': 'a-okvqa'
                }

                # 尝试加载图像
                if sample['image_path'] and os.path.exists(sample['image_path']):
                    try:
                        sample['image'] = Image.open(sample['image_path']).convert('RGB')
                    except:
                        sample['image'] = None

                data.append(sample)

            return data[:500]  # 限制样本数

        except Exception as e:
            print(f"⚠️ A-OKVQA加载失败: {e}")
            return self._get_dummy_data('a-okvqa', 100)

    def _load_multimodalqa(self, split: str) -> List[Dict]:
        """加载MultiModalQA数据集"""
        try:
            # MultiModalQA数据文件路径
            data_file = Path(self.dataset_paths['multimodalqa']['data_path']) / f'MMQA_{split}.jsonl.gz'

            if not data_file.exists():
                print(f"⚠️ MultiModalQA数据文件不存在: {data_file}")
                return self._get_dummy_data('multimodalqa', 100)

            data = []
            with gzip.open(data_file, 'rt') as f:
                for line in f:
                    item = json.loads(line.strip())

                    # 转换为统一格式
                    sample = {
                        'id': item.get('id', ''),
                        'question': item['question'],
                        'golden_answers': [item.get('answer', '')],
                        'dataset': 'multimodalqa'
                    }

                    # 处理图像
                    if 'image' in item and item['image']:
                        sample['image_path'] = item['image']
                        # 尝试加载图像
                        if os.path.exists(item['image']):
                            try:
                                sample['image'] = Image.open(item['image']).convert('RGB')
                            except:
                                sample['image'] = None

                    # 处理表格
                    if 'table' in item:
                        sample['table'] = item['table']

                    # 处理文本段落
                    if 'text' in item:
                        sample['context'] = item['text']

                    data.append(sample)

            return data[:500]  # 限制样本数

        except Exception as e:
            print(f"⚠️ MultiModalQA加载失败: {e}")
            return self._get_dummy_data('multimodalqa', 100)

    def _load_mrag_bench(self, split: str) -> List[Dict]:
        """加载MRAG-Bench数据集"""
        try:
            # MRAG-Bench数据文件路径
            data_file = Path(self.dataset_paths['mrag-bench']['data_path']) / f'mragbench_{split}.json'

            if not data_file.exists():
                print(f"⚠️ MRAG-Bench数据文件不存在: {data_file}")
                return self._get_dummy_data('mrag-bench', 100)

            with open(data_file, 'r') as f:
                raw_data = json.load(f)

            data = []
            for item in raw_data:
                # 转换为统一格式
                sample = {
                    'id': item.get('id', ''),
                    'question': item['question'],
                    # MRAG-Bench是多选题格式
                    'A': item['choices'][0],
                    'B': item['choices'][1],
                    'C': item['choices'][2],
                    'D': item['choices'][3],
                    'golden_answers': [item['gt_choice']],  # 如 'A', 'B', 'C', 'D'
                    'scenario': item.get('scenario', 'Unknown'),
                    'dataset': 'mrag-bench'
                }

                # 处理图像
                if 'image_id' in item:
                    image_path = os.path.join(
                        self.dataset_paths['mrag-bench']['image_path'],
                        f"{item['image_id']}.jpg"
                    )
                    sample['image_path'] = image_path
                    if os.path.exists(image_path):
                        try:
                            sample['image'] = Image.open(image_path).convert('RGB')
                        except:
                            sample['image'] = None

                data.append(sample)

            return data[:500]  # 限制样本数

        except Exception as e:
            print(f"⚠️ MRAG-Bench加载失败: {e}")
            return self._get_dummy_data('mrag-bench', 100)

    def _get_aokvqa_image_path(self, image_id: str) -> str:
        """获取A-OKVQA图像路径"""
        # A-OKVQA使用COCO图像
        if self.dataset_paths['a-okvqa']['image_path']:
            return os.path.join(
                self.dataset_paths['a-okvqa']['image_path'],
                f"COCO_train2014_{image_id:012d}.jpg"
            )
        return ''

    def _get_dummy_data(self, dataset_name: str, num_samples: int) -> List[Dict]:
        """生成示例数据（用于测试）"""
        print(f"使用 {dataset_name} 示例数据 ({num_samples} 样本)")

        dummy_data = []
        for i in range(num_samples):
            if dataset_name == 'mrag-bench':
                # MRAG-Bench是多选题
                sample = {
                    'id': f'{dataset_name}_{i}',
                    'question': f'Sample question {i} for {dataset_name}?',
                    'A': f'Option A for question {i}',
                    'B': f'Option B for question {i}',
                    'C': f'Option C for question {i}',
                    'D': f'Option D for question {i}',
                    'golden_answers': ['A'],  # 默认答案
                    'scenario': 'Test',
                    'dataset': dataset_name
                }
            else:
                # 其他数据集是开放题
                sample = {
                    'id': f'{dataset_name}_{i}',
                    'question': f'Sample question {i} for {dataset_name}?',
                    'golden_answers': [f'Answer {i}'],
                    'dataset': dataset_name
                }

            dummy_data.append(sample)

        return dummy_data


# 便捷函数
def load_unified_dataset(dataset_name: str, split: str = 'val', max_samples: Optional[int] = None,
                        config: Dict = None) -> UnifiedDataset:
    """便捷的数据集加载函数"""
    loader = UnifiedDatasetLoader(config)
    return loader.load_dataset(dataset_name, split, max_samples)