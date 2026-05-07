#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
InfoSeek数据集加载器

数据集：InfoSeek (Chen et al., 2023)
规模：8.9K human-written questions  
特点：Fine-grained visual knowledge
用途：测试细粒度归因

数据位置：/root/autodl-tmp/FlashRAG/flashrag/data/InfoSeek/
注意：图像未下载（太大），只有文本标注
"""

import json
import os
from typing import List, Dict, Any
from tqdm import tqdm

class InfoSeekDataset:
    """
    InfoSeek数据集加载器
    
    注意：不包含图像（图像太大未下载）
    只加载问题、答案和知识库映射
    
    使用示例：
    ```python
    dataset = InfoSeekDataset(
        data_dir='/root/autodl-tmp/FlashRAG/flashrag/data/InfoSeek',
        split='val'
    )
    
    samples = dataset.load()
    print(f"加载了 {len(samples)} 个样本")
    ```
    """
    
    def __init__(self, data_dir: str, split: str = 'val'):
        """
        初始化
        
        Args:
            data_dir: 数据目录
            split: 'train', 'val', 'test', 'human'
        """
        self.data_dir = data_dir
        self.split = split
        
        # 查找数据文件
        self.data_files = self._find_data_files()
        
        print(f"✅ InfoSeek数据集初始化")
        print(f"  Split: {split}")
        print(f"  数据目录: {data_dir}")
        print(f"  找到文件: {len(self.data_files)} 个")
    
    def _find_data_files(self):
        """查找数据文件"""
        files = []
        
        # 可能的文件模式
        patterns = [
            f'{self.split}_annotation.json',
            f'{self.split}_annotations.json',
            f'{self.split}.json',
            f'{self.split}.jsonl',
            f'infoseek_{self.split}.json',
        ]
        
        for pattern in patterns:
            filepath = os.path.join(self.data_dir, pattern)
            if os.path.exists(filepath):
                files.append(filepath)
        
        # 如果没找到，列出目录下所有json文件
        if not files:
            for f in os.listdir(self.data_dir):
                if f.endswith('.json') or f.endswith('.jsonl'):
                    files.append(os.path.join(self.data_dir, f))
        
        return files
    
    def load(self, max_samples: int = None) -> List[Dict]:
        """
        加载数据集
        
        Args:
            max_samples: 最大样本数
            
        Returns:
            List[Dict]: 样本列表
        """
        print(f"\n加载InfoSeek {self.split} split...")
        
        if not self.data_files:
            print(f"⚠️  未找到数据文件")
            return []
        
        samples = []
        
        for data_file in self.data_files:
            print(f"读取: {os.path.basename(data_file)}")
            
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    if data_file.endswith('.jsonl'):
                        # JSONL格式（逐行）
                        for i, line in enumerate(f):
                            if max_samples and len(samples) >= max_samples:
                                break
                            data = json.loads(line)
                            samples.append(self._process_sample(data, i))
                    else:
                        # JSON格式
                        data = json.load(f)
                        
                        # 可能是列表或字典
                        if isinstance(data, list):
                            items = data
                        elif isinstance(data, dict):
                            # 可能在某个key下
                            items = data.get('data', data.get('questions', data.get('samples', [data])))
                        else:
                            items = [data]
                        
                        for i, item in enumerate(items):
                            if max_samples and len(samples) >= max_samples:
                                break
                            samples.append(self._process_sample(item, i))
            
            except Exception as e:
                print(f"⚠️  读取文件失败: {e}")
                continue
        
        print(f"✅ 加载了 {len(samples)} 个样本")
        
        return samples
    
    def _process_sample(self, data: Dict, idx: int) -> Dict:
        """处理单个样本"""
        return {
            'id': data.get('id', data.get('question_id', idx)),
            'question': data.get('question', data.get('query', '')),
            'answer': data.get('answer', data.get('answers', [])),
            'entity': data.get('entity', ''),
            'entity_id': data.get('entity_id', ''),
            'image_id': data.get('image_id', ''),
            # 注：图像未下载，这里只是ID
            'metadata': {
                'dataset': 'infoseek',
                'split': self.split,
                'has_image': False  # 标记图像未下载
            }
        }
    
    def __len__(self):
        """返回数据集大小估计"""
        # 根据split返回估计值
        if self.split == 'train':
            return 6000
        elif self.split == 'val':
            return 1500
        elif self.split == 'test':
            return 1400
        elif self.split == 'human':
            return 500
        return 1000
    
    def __getitem__(self, idx):
        """获取单个样本"""
        samples = self.load(max_samples=idx+1)
        return samples[idx] if idx < len(samples) else None


# 便捷函数
def load_infoseek(data_dir: str, split: str = 'val', max_samples: int = None):
    """快速加载InfoSeek"""
    dataset = InfoSeekDataset(data_dir, split)
    return dataset.load(max_samples)


if __name__ == '__main__':
    print("InfoSeek数据集加载器")
    print("=" * 70)
    
    data_dir = '/root/autodl-tmp/FlashRAG/flashrag/data/InfoSeek'
    
    if os.path.exists(data_dir):
        # 列出目录内容
        print(f"\n目录内容:")
        for f in os.listdir(data_dir):
            filepath = os.path.join(data_dir, f)
            size = os.path.getsize(filepath) if os.path.isfile(filepath) else 0
            print(f"  {f}: {size/1024/1024:.1f} MB")
        
        print("\n尝试加载数据集...")
        dataset = InfoSeekDataset(data_dir, 'val')
        
        samples = dataset.load(max_samples=5)
        
        if samples:
            print(f"\n✅ 加载成功: {len(samples)} 个样本")
            print(f"\n示例样本:")
            sample = samples[0]
            print(f"  ID: {sample['id']}")
            print(f"  Question: {sample['question'][:100] if sample['question'] else 'N/A'}...")
            print(f"  Answer: {sample['answer']}")
            print(f"  Has Image: {sample['metadata']['has_image']}")
            print()
            print("⚠️  注意：图像未下载（太大），只有文本标注")
            print("    可用于：测试归因算法（不需要图像）")
            print("    不可用于：完整VQA评测（需要图像）")
        else:
            print("⚠️  未能加载样本")
    else:
        print(f"⚠️  数据目录不存在: {data_dir}")
    
    print("=" * 70)


