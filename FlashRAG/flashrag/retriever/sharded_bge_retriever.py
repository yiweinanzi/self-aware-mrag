# -*- coding: utf-8 -*-
"""
分片BGE检索器

加载多个BGE分片，检索时查询所有分片并合并结果
"""

import torch
import json
import os
from typing import List, Tuple
import numpy as np


class ShardedBGERetriever:
    """
    分片BGE检索器
    
    功能：
    - 加载多个BGE分片
    - 检索时查询所有分片
    - 合并top-k结果
    
    使用：
    retriever = ShardedBGERetriever()
    retriever.load_shards('cache/bge_shards')
    results = retriever.retrieve("query", top_k=5)
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.shards = []
        self.n_shards = 0
    
    def load_shards(self, shard_dir: str):
        """
        加载所有分片
        
        Args:
            shard_dir: 分片目录
        """
        # 读取清单
        manifest_file = f'{shard_dir}/manifest.json'
        with open(manifest_file) as f:
            manifest = json.load(f)
        
        self.n_shards = manifest['n_shards']
        
        print(f"正在加载{self.n_shards}个分片...")
        
        for shard_id in range(self.n_shards):
            shard_file = f'{shard_dir}/shard_{shard_id:02d}.pt'
            
            shard_data = torch.load(shard_file, map_location='cpu')
            
            # 只加载corpus，embeddings按需加载（节省显存）
            self.shards.append({
                'corpus': shard_data['corpus'],
                'embeddings_file': shard_file,
                'embeddings': None,  # 延迟加载
                'shard_id': shard_id
            })
            
            print(f"  分片{shard_id+1}/{self.n_shards}: {len(shard_data['corpus']):,} 条", flush=True)
        
        total_docs = sum(len(shard['corpus']) for shard in self.shards)
        print(f"✅ 分片加载完成，总计{total_docs:,}条文档")
    
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """
        检索所有分片并合并结果
        
        Args:
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            (results, scores)
        """
        # 编码查询（使用第一个分片的模型配置）
        # 为简化，这里使用预加载的query encoder
        # 实际应该加载BGE模型编码query
        
        all_results = []
        all_scores = []
        
        # 查询每个分片
        for shard in self.shards:
            # 加载这个分片的embeddings（如果未加载）
            if shard['embeddings'] is None:
                shard_data = torch.load(shard['embeddings_file'])
                shard['embeddings'] = shard_data['embeddings'].to(self.device)
            
            # 查询（这里简化，实际需要query encoder）
            # scores = compute_similarity(query_emb, shard['embeddings'])
            
            # 临时：返回前几个（演示用）
            # 实际应该基于相似度
            pass
        
        # 合并所有分片的结果，取总top-k
        # ...
        
        return all_results[:top_k], all_scores[:top_k]

