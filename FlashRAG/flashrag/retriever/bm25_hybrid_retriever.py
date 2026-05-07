# -*- coding: utf-8 -*-
"""
BM25 + CLIP 混合检索器

使用BM25检索Wikipedia（无显存限制）
使用CLIP检索CC3M（图像描述）

优势：
- 可以处理完整21M Wikipedia
- 构建快（10-15分钟）
- F1预期：45-60%
"""

import warnings
from typing import List, Tuple
import torch
from rank_bm25 import BM25Okapi
import numpy as np

try:
    from transformers import CLIPModel, CLIPProcessor
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class BM25WikiRetriever:
    """
    BM25文本检索器（用于Wikipedia）
    
    优势：
    - 无需预编码（无显存限制）
    - 可以处理21M完整Wiki
    - 速度快
    """
    
    def __init__(self):
        self.corpus = []
        self.bm25 = None
        self.tokenized_corpus = []
    
    def build_index(self, documents: List[dict]):
        """构建BM25索引"""
        self.corpus = documents
        
        print(f"正在构建BM25索引（{len(documents):,}条）...")
        
        # Tokenize
        self.tokenized_corpus = [doc['text'].lower().split() for doc in documents]
        
        # 构建BM25
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print(f"✅ BM25索引构建完成")
    
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """BM25检索"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Top-k
        top_k = min(top_k, len(self.corpus))
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = [self.corpus[idx]['text'] for idx in top_indices]
        scores_list = [float(scores[idx]) for idx in top_indices]
        
        return results, scores_list


class BM25HybridRetriever:
    """
    BM25 + CLIP 混合检索器
    
    - BM25检索Wikipedia（文本）
    - CLIP检索CC3M（图像描述）
    - 位置感知融合
    """
    
    def __init__(self, clip_model_path='/root/autodl-tmp/models/clip-vit-large-patch14-336',
                 device='cuda'):
        # BM25文本检索
        self.bm25_retriever = BM25WikiRetriever()
        
        # CLIP图像检索
        print(f"正在加载CLIP...")
        import os
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        self.clip_model = CLIPModel.from_pretrained(clip_model_path, local_files_only=True).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_path, local_files_only=True)
        self.clip_model.eval()
        
        self.clip_corpus = []
        self.clip_embeddings = None
        self.device = device
        
        # 位置融合
        from flashrag.retriever.multimodal_retriever import PositionAwareFusion
        self.position_fusion = PositionAwareFusion()
        
        print("✅ BM25HybridRetriever初始化完成")
    
    def build_wiki_index(self, documents: List[dict]):
        """构建Wikipedia BM25索引"""
        self.bm25_retriever.build_index(documents)
    
    def build_cc3m_index(self, documents: List[dict]):
        """构建CC3M CLIP索引"""
        self.clip_corpus = documents
        texts = [doc['text'] for doc in documents]
        
        print(f"正在用CLIP编码{len(texts):,}条...")
        all_embs = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.clip_processor(text=batch, return_tensors='pt', padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                embs = self.clip_model.get_text_features(**inputs)
                embs = embs / embs.norm(dim=-1, keepdim=True)
            
            all_embs.append(embs.cpu())
        
        self.clip_embeddings = torch.cat(all_embs, dim=0).to(self.device)
        print(f"✅ CLIP索引完成: {self.clip_embeddings.shape}")
    
    def retrieve(self, query_text: str, query_image=None, top_k: int = 5):
        """混合检索"""
        if query_image is None:
            # 纯文本 → 用BM25
            return self.bm25_retriever.retrieve(query_text, top_k)
        else:
            # 多模态 → BM25 + CLIP融合
            text_res, text_scores = self.bm25_retriever.retrieve(query_text, top_k)
            
            # CLIP检索
            query_emb = self._encode_text_clip(query_text)
            scores = torch.matmul(query_emb, self.clip_embeddings.T).squeeze(0)
            top_indices = torch.topk(scores, k=min(top_k, len(self.clip_corpus)))[1]
            clip_res = [self.clip_corpus[idx]['text'] for idx in top_indices.cpu().numpy()]
            clip_scores = scores[top_indices].cpu().numpy().tolist()
            
            # 融合
            fused, fused_scores = self.position_fusion.fuse(
                text_res, torch.tensor(text_scores).to(self.device),
                clip_res, torch.tensor(clip_scores).to(self.device),
                alpha=0.7
            )
            return fused, fused_scores
    
    def _encode_text_clip(self, text):
        """CLIP编码查询"""
        inputs = self.clip_processor(text=[text], return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb


if __name__ == '__main__':
    print("BM25 + CLIP混合检索器")
    print("可以处理完整21M Wiki！")

