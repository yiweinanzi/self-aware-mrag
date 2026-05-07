# -*- coding: utf-8 -*-
"""
BGE文本检索器（不依赖sentence-transformers）

直接使用transformers加载BGE模型，避免版本冲突

参考文档：第224-230行，使用BGE进行文本检索
"""

import warnings
from typing import List, Tuple
import torch

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class BGERetriever:
    """
    BGE文本检索器
    
    使用BGE-M3或BGE-large进行文本语义检索
    适用于Wikipedia等纯文本语料
    
    使用示例：
    ```python
    retriever = BGERetriever(device='cuda')
    retriever.add_documents(wikipedia_texts)
    results = retriever.retrieve("What is motocross?", top_k=5)
    ```
    """
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 device: str = 'cuda'):
        """
        初始化BGE检索器
        
        Args:
            model_name: BGE模型名称
                - BAAI/bge-small-en-v1.5（小模型，快）
                - BAAI/bge-base-en-v1.5（中等）
                - BAAI/bge-large-en-v1.5（大模型，准确）
            device: 设备
        """
        self.device = device
        self.model_name = model_name
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("需要transformers")
        
        print(f"正在加载BGE模型: {model_name}")
        
        # 使用transformers直接加载（避免sentence-transformers冲突）
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.model.eval()
            print("✅ BGE模型加载成功")
        except Exception as e:
            # 如果模型未下载，使用备用方案（本地CLIP）
            warnings.warn(f"BGE加载失败: {e}")
            print("⚠️ 降级方案：使用CLIP进行文本检索")
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained(
                '/root/autodl-tmp/models/clip-vit-large-patch14-336',
                local_files_only=True
            ).to(device)
            self.tokenizer = CLIPProcessor.from_pretrained(
                '/root/autodl-tmp/models/clip-vit-large-patch14-336',
                local_files_only=True
            )
            self.is_clip_fallback = True
        
        self.corpus = []
        self.corpus_embeddings = None
    
    def encode_text(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        编码文本为embeddings
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            
        Returns:
            torch.Tensor: embeddings [N, hidden_dim]
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Encode
            with torch.no_grad():
                if hasattr(self, 'is_clip_fallback') and self.is_clip_fallback:
                    # CLIP降级方案
                    outputs = self.model.get_text_features(**inputs)
                else:
                    # BGE标准方案
                    outputs = self.model(**inputs)
                    # 取[CLS] token的embedding
                    embeddings = outputs.last_hidden_state[:, 0]  # [batch, hidden]
                    # 归一化
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    outputs = embeddings
            
            all_embeddings.append(outputs.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def add_documents(self, documents: List[dict]):
        """添加文档并构建索引"""
        self.corpus = documents
        
        # 提取文本
        texts = [doc['text'] for doc in documents]
        
        print(f"正在编码 {len(texts):,} 条文档...")
        self.corpus_embeddings = self.encode_text(texts).to(self.device)
        print(f"✅ 索引构建完成: {self.corpus_embeddings.shape}")
    
    def retrieve(self, query: str, top_k: int = 5,
                return_score: bool = False) -> Tuple[List[str], List[float]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回数量
            return_score: 是否返回分数
            
        Returns:
            (results, scores)
        """
        # 编码查询
        query_emb = self.encode_text([query]).to(self.device)
        
        # 计算相似度
        scores = torch.matmul(query_emb, self.corpus_embeddings.T).squeeze(0)
        
        # Top-k
        top_k = min(top_k, len(self.corpus))
        top_scores, top_indices = torch.topk(scores, k=top_k)
        
        results = [self.corpus[idx]['text'] for idx in top_indices.cpu().numpy()]
        scores_list = top_scores.cpu().numpy().tolist()
        
        if return_score:
            return results, scores_list
        return results


if __name__ == '__main__':
    print("BGE文本检索器测试")
    print("=" * 70)
    
    # 测试（使用CLIP降级方案）
    retriever = BGERetriever(device='cuda')
    
    # 添加测试文档
    docs = [
        {'text': 'Motocross is off-road motorcycle racing'},
        {'text': 'Soccer is a team sport'},
        {'text': 'A cat is a small animal'}
    ]
    
    retriever.add_documents(docs)
    
    # 测试检索
    results = retriever.retrieve("motorcycle sport", top_k=2)
    
    print("\n测试查询: motorcycle sport")
    print("结果:")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc}")
    
    print("\n✅ BGE检索器测试完成")

