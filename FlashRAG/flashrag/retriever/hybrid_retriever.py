# -*- coding: utf-8 -*-
"""
混合检索器：BGE(文本) + CLIP(图像)

完全按照文档第224-264行要求实现：
- 文本检索：BGE-large-en-v1.5
- 图像检索：CLIP
- 无图像时：只用BGE
- 有图像时：BGE + CLIP融合

这是修复F1性能的关键！
"""

import warnings
from typing import List, Tuple, Optional
import torch
import numpy as np
from PIL import Image

try:
    from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class BGETextRetriever:
    """
    BGE文本检索器（用于Wikipedia）
    
    按照文档第224-231行要求
    """
    
    def __init__(self, model_path: str = "BAAI/bge-large-en-v1.5", 
                 device: str = 'cuda'):
        self.device = device
        self.model_path = model_path
        
        print(f"正在加载BGE文本检索器: {model_path}")
        
        import os
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        # 加载BGE
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
        self.model.eval()
        
        print("✅ BGE模型加载成功")
        
        self.corpus = []
        self.corpus_embeddings = None
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """BGE编码文本"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 取[CLS] token
                embeddings = outputs.last_hidden_state[:, 0]
                # 归一化
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def build_index(self, documents: List[dict]):
        """构建索引"""
        self.corpus = documents
        texts = [doc['text'] for doc in documents]
        
        print(f"正在用BGE编码 {len(texts):,} 条文档...")
        self.corpus_embeddings = self.encode_texts(texts).to(self.device)
        print(f"✅ BGE索引构建完成: {self.corpus_embeddings.shape}")
    
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """BGE检索"""
        query_emb = self.encode_texts([query]).to(self.device)
        scores = torch.matmul(query_emb, self.corpus_embeddings.T).squeeze(0)
        
        top_k = min(top_k, len(self.corpus))
        top_scores, top_indices = torch.topk(scores, k=top_k)
        
        results = [self.corpus[idx]['text'] for idx in top_indices.cpu().numpy()]
        scores_list = top_scores.cpu().numpy().tolist()
        
        return results, scores_list


class CLIPImageRetriever:
    """
    CLIP图像检索器（用于CC3M）
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        
        print(f"正在加载CLIP图像检索器: {model_path}")
        
        import os
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        self.clip_model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        self.clip_model.eval()
        
        print("✅ CLIP模型加载成功")
        
        self.corpus = []
        self.corpus_embeddings = None
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """CLIP编码文本（用于图像描述）"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            inputs = self.clip_processor(
                text=batch,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                embeddings = self.clip_model.get_text_features(**inputs)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def build_index(self, documents: List[dict]):
        """构建索引"""
        self.corpus = documents
        texts = [doc['text'] for doc in documents]
        
        print(f"正在用CLIP编码 {len(texts):,} 条图像描述...")
        self.corpus_embeddings = self.encode_texts(texts).to(self.device)
        print(f"✅ CLIP索引构建完成: {self.corpus_embeddings.shape}")
    
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """CLIP检索"""
        query_emb = self.encode_texts([query]).to(self.device)
        scores = torch.matmul(query_emb, self.corpus_embeddings.T).squeeze(0)
        
        top_k = min(top_k, len(self.corpus))
        top_scores, top_indices = torch.topk(scores, k=top_k)
        
        results = [self.corpus[idx]['text'] for idx in top_indices.cpu().numpy()]
        scores_list = top_scores.cpu().numpy().tolist()
        
        return results, scores_list


class HybridRetriever:
    """
    混合检索器：BGE + CLIP
    
    完全按照文档第234-263行实现：
    - 纯文本查询：用BGE
    - 有图像查询：BGE + CLIP融合
    - 应用位置感知融合
    """
    
    def __init__(self, 
                 bge_model_path='/root/autodl-tmp/models/bge-large-en-v1.5',
                 clip_model_path='/root/autodl-tmp/models/clip-vit-large-patch14-336',
                 device='cuda'):
        """
        初始化混合检索器
        
        按照文档第247-249行
        """
        # BGE文本检索器
        self.text_retriever = BGETextRetriever(bge_model_path, device)
        
        # CLIP图像检索器  
        self.clip_retriever = CLIPImageRetriever(clip_model_path, device)
        
        # 位置感知融合
        from flashrag.retriever.multimodal_retriever import PositionAwareFusion
        self.position_fusion = PositionAwareFusion(fusion_method='weighted')
        
        self.device = device
        
        print("✅ HybridRetriever初始化完成")
        print("   - BGE（文本检索）")
        print("   - CLIP（图像检索）")
        print("   - 位置感知融合")
    
    def build_wiki_index(self, wiki_documents: List[dict]):
        """用BGE构建Wikipedia索引"""
        self.text_retriever.build_index(wiki_documents)
    
    def build_cc3m_index(self, cc3m_documents: List[dict]):
        """用CLIP构建CC3M索引"""
        self.clip_retriever.build_index(cc3m_documents)
    
    def retrieve(self, query_text: str, query_image: Optional[Image.Image] = None,
                top_k: int = 5) -> Tuple[List[str], List[float]]:
        """
        混合检索（文档第251-263行）
        
        Args:
            query_text: 查询文本
            query_image: 查询图像（可选）
            top_k: 返回数量
            
        Returns:
            (results, scores)
        """
        if query_image is None:
            # 第253-254行：纯文本 → 只用BGE
            return self.text_retriever.retrieve(query_text, top_k)
        
        else:
            # 第257-258行：多模态 → BGE + CLIP融合
            text_results, text_scores = self.text_retriever.retrieve(query_text, top_k)
            image_results, image_scores = self.clip_retriever.retrieve(query_text, top_k)
            
            # 第260-261行：位置感知融合
            fused_results, fused_scores = self.position_fusion.fuse(
                text_results, torch.tensor(text_scores).to(self.device),
                image_results, torch.tensor(image_scores).to(self.device),
                alpha=0.6  # 文本权重
            )
            
            return fused_results, fused_scores
    
    def save_index(self, save_path: str):
        """保存索引"""
        import torch
        
        index_data = {
            'bge_corpus': self.text_retriever.corpus,
            'bge_embeddings': self.text_retriever.corpus_embeddings.cpu(),
            'clip_corpus': self.clip_retriever.corpus,
            'clip_embeddings': self.clip_retriever.corpus_embeddings.cpu()
        }
        
        torch.save(index_data, save_path)
        print(f"✅ 混合索引已保存: {save_path}")
    
    def load_index(self, load_path: str):
        """加载索引"""
        import torch
        
        index_data = torch.load(load_path)
        
        self.text_retriever.corpus = index_data['bge_corpus']
        self.text_retriever.corpus_embeddings = index_data['bge_embeddings'].to(self.device)
        
        self.clip_retriever.corpus = index_data['clip_corpus']
        self.clip_retriever.corpus_embeddings = index_data['clip_embeddings'].to(self.device)
        
        print(f"✅ 混合索引加载完成")
        print(f"   BGE语料: {len(self.text_retriever.corpus):,} 条")
        print(f"   CLIP语料: {len(self.clip_retriever.corpus):,} 条")


if __name__ == '__main__':
    print("=" * 70)
    print("HybridRetriever测试（符合文档第234-263行）")
    print("=" * 70)
    
    # 测试
    retriever = HybridRetriever(device='cuda')
    
    # 添加测试文档
    wiki_docs = [
        {'text': 'Motocross is off-road motorcycle racing'},
        {'text': 'Soccer is a team sport'}
    ]
    
    cc3m_docs = [
        {'text': 'A person riding a motorcycle on a track'},
        {'text': 'People playing soccer on a field'}
    ]
    
    retriever.build_wiki_index(wiki_docs)
    retriever.build_cc3m_index(cc3m_docs)
    
    # 测试纯文本检索（应该用BGE）
    print("\n测试1：纯文本检索（用BGE）")
    results, scores = retriever.retrieve("motorcycle sport", top_k=2)
    for i, (doc, score) in enumerate(zip(results, scores)):
        print(f"{i+1}. [{score:.3f}] {doc}")
    
    print("\n✅ HybridRetriever测试完成")
    print("=" * 70)

