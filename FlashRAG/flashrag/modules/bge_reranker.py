#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BGE Reranker 封装
用于文档重排
"""

import torch
from typing import List, Tuple
import warnings


class BGEReranker:
    """
    BGE Reranker封装
    用于重排检索文档，提升相关性
    """
    
    def __init__(self, model_name='BAAI/bge-reranker-v2-m3', device='cuda'):
        """
        初始化BGE Reranker
        
        Args:
            model_name: 模型名称
            device: 设备
        """
        self.device = device
        self.model_name = model_name
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import os
            
            # ✅ 使用HF镜像加速下载
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            
            # 检查本地路径
            local_path = f'/root/autodl-tmp/models/{model_name.split("/")[-1]}'
            if os.path.exists(local_path):
                print(f"✅ 使用本地BGE Reranker: {local_path}")
                model_path = local_path
            else:
                print(f"📥 从HF镜像下载BGE Reranker: {model_name}")
                model_path = model_name
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model = self.model.to(device)
            self.model.eval()
            print(f"✅ BGE Reranker加载成功")
            
        except Exception as e:
            print(f"⚠️  BGE Reranker加载失败: {e}，将使用mock模式（简化版）")
            self.model = None
            self.tokenizer = None
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        """
        重排文档
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回top-k文档
            
        Returns:
            重排后的文档列表（top-k）
        """
        if self.model is None:
            # Mock模式：返回原始文档
            return documents[:top_k]
        
        if not documents:
            return []
        
        try:
            # 构造查询-文档对
            pairs = [[query, doc] for doc in documents]
            
            # 分批处理（避免OOM）
            batch_size = 32
            all_scores = []
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                
                with torch.no_grad():
                    inputs = self.tokenizer(
                        batch_pairs,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=512
                    ).to(self.device)
                    
                    scores = self.model(**inputs, return_dict=True).logits.squeeze(-1)
                    all_scores.extend(scores.cpu().tolist())
            
            # 排序
            scored_docs = list(zip(documents, all_scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # 返回top-k
            return [doc for doc, score in scored_docs[:top_k]]
        
        except Exception as e:
            warnings.warn(f"重排失败: {e}，返回原始顺序")
            return documents[:top_k]
    
    def score(self, query: str, document: str) -> float:
        """
        计算单个文档的相关性分数
        
        Args:
            query: 查询
            document: 文档
            
        Returns:
            相关性分数
        """
        if self.model is None:
            return 0.5
        
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    [[query, document]],
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                ).to(self.device)
                
                score = self.model(**inputs, return_dict=True).logits.squeeze(-1)
                return float(score.cpu().item())
        
        except Exception as e:
            warnings.warn(f"打分失败: {e}")
            return 0.5


def create_bge_reranker(model_name='BAAI/bge-reranker-v2-m3', device='cuda'):
    """创建BGE Reranker（支持HF镜像）"""
    return BGEReranker(model_name, device)


if __name__ == '__main__':
    # 测试
    print("测试BGE Reranker")
    
    reranker = create_bge_reranker()
    
    query = "What is the capital of France?"
    docs = [
        "Paris is the capital and most populous city of France.",
        "London is the capital of the United Kingdom.",
        "The Eiffel Tower is located in Paris.",
        "France is a country in Western Europe."
    ]
    
    print(f"\n查询: {query}")
    print(f"原始文档: {len(docs)}个")
    
    reranked = reranker.rerank(query, docs, top_k=2)
    print(f"\n重排后Top-2:")
    for i, doc in enumerate(reranked, 1):
        print(f"  {i}. {doc}")

