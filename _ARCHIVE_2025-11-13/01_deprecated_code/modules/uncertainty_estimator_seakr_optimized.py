# -*- coding: utf-8 -*-
"""
跨模态不确定性估计模块（SeaKR优化版）
Cross-Modal Uncertainty Estimator (SeaKR-Optimized)

基于SeaKR (ACL 2024)的实现进行优化
参考文档：创新点1-自感知多模态RAG-实施方案.md 第789-849行
参考代码：SeaKR-main/vllm_uncertainty/vllm/engine/llm_engine.py

核心创新：
1. 文本不确定性：eigen_score（协方差矩阵对数行列式）+ perplexity
2. 视觉不确定性：attention variance（新创新）
3. 跨模态对齐不确定性：JS散度（新创新）

关键改进（基于SeaKR源码）：
- 使用SeaKR的eigen_score计算方法（而非简单的Gram矩阵）
- 添加eigen_threshold参数（默认-6.0）
- 添加perplexity和energy_score支持
- 保留我们的多模态扩展（visual + alignment uncertainty）
"""

import warnings
from typing import Dict, Optional, Tuple, Union, List
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("torch未安装，不确定性估计功能受限")


class SeaKROptimizedUncertaintyEstimator:
    """
    基于SeaKR优化的跨模态不确定性估计器
    
    参考：
    - SeaKR (ACL 2024): Self-Aware Knowledge Retrieval
    - 代码：SeaKR-main/vllm_uncertainty/vllm/engine/llm_engine.py
    
    核心算法（从SeaKR源码提取）：
    1. eigen_score = (1/k) * log|Σ + α*I|
       其中：Σ = z * J_d * z^T
       J_d = I_d - (1/d) * 1_d * 1_d^T
    
    2. perplexity = exp(-mean(log_probs))
    
    3. energy_score = logsumexp(logits, dim=-1)
    
    使用示例：
    ```python
    estimator = SeaKROptimizedUncertaintyEstimator(
        mllm_model=your_model,
        config={'eigen_threshold': -6.0}
    )
    
    # 文本不确定性
    uncertainty = estimator.estimate_text_uncertainty(text, embeddings)
    
    # 完整不确定性
    uncertainties = estimator.estimate(text, image)
    
    # 判断是否检索
    should_retrieve, modality = estimator.should_retrieve(uncertainties)
    ```
    """
    
    def __init__(self, mllm_model=None, config=None):
        """
        初始化不确定性估计器
        
        Args:
            mllm_model: 多模态大模型（如LLaVA-1.5）
            config: 配置字典
        """
        self.mllm_model = mllm_model
        self.config = config or {}
        
        # SeaKR关键参数
        self.eigen_threshold = self.config.get('eigen_threshold', -6.0)  # SeaKR默认值
        self.eigen_alpha = self.config.get('eigen_alpha', 1e-10)  # 正则化参数
        
        # 不确定性阈值
        self.uncertainty_threshold = self.config.get('uncertainty_threshold', 0.5)
        
        # 权重配置（用于总不确定性计算）
        self.alpha = self.config.get('text_weight', 0.4)  # 文本不确定性权重
        self.beta = self.config.get('visual_weight', 0.3)  # 视觉不确定性权重
        self.gamma = self.config.get('alignment_weight', 0.3)  # 对齐不确定性权重
        
        if not TORCH_AVAILABLE:
            warnings.warn("torch未安装，将使用简化版不确定性估计")
    
    def compute_eigen_score(self, embeddings: torch.Tensor) -> float:
        """
        计算eigen_score（SeaKR核心算法）
        
        参考：SeaKR-main/vllm_uncertainty/vllm/engine/llm_engine.py 第738-744行
        
        公式：
        eigen_score = (1/k) * log|Σ + α*I|
        其中：
        - Σ = z * J_d * z^T  （协方差矩阵）
        - J_d = I_d - (1/d) * 1_d * 1_d^T  （centering matrix）
        - k: 样本数
        - d: 嵌入维度
        - α: 正则化参数（防止奇异）
        
        Args:
            embeddings: shape [k, d]，k个样本的嵌入向量
            
        Returns:
            float: eigen_score，通常在[-10, 0]范围
                  越接近0表示不确定性越低
                  < eigen_threshold（-6.0）表示需要检索
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("需要torch来计算eigen_score")
        
        z = embeddings.to(torch.float32)
        k, d = z.shape
        
        # 构建centering matrix J_d
        # J_d = I_d - (1/d) * 1_d * 1_d^T
        j_d = torch.eye(d) - (1/d) * torch.ones(d, d)
        j_d = j_d.to(z.device)
        
        # 计算协方差矩阵 Σ = z * J_d * z^T
        # 使用einsum提高效率
        sigma = torch.einsum('ij,jk,kl->il', z, j_d, z.t())
        
        # 计算 log|Σ + α*I|
        # 加入正则化项防止矩阵奇异
        eigen_score = (1/k) * torch.logdet(
            sigma + self.eigen_alpha * torch.eye(k, device=sigma.device)
        )
        
        return eigen_score.item()
    
    def compute_perplexity(self, log_probs: List[float]) -> float:
        """
        计算perplexity（困惑度）
        
        参考：SeaKR-main/vllm_uncertainty/vllm/engine/llm_engine.py 第746-751行
        
        公式：perplexity = exp(-mean(log_probs))
        
        Args:
            log_probs: 每个token的对数概率列表
            
        Returns:
            float: perplexity值，越小表示越确定
        """
        if not log_probs or len(log_probs) == 0:
            return 1e3  # 默认高困惑度
        
        # 移除最后一个token（通常是EOS）
        valid_log_probs = log_probs[:-1] if len(log_probs) > 1 else log_probs
        
        if not valid_log_probs:
            return 1e3
        
        mean_log_prob = np.mean(valid_log_probs)
        perplexity = np.exp(-mean_log_prob)
        
        return perplexity
    
    def compute_energy_score(self, logits: torch.Tensor) -> float:
        """
        计算energy_score
        
        参考：SeaKR-main/vllm_uncertainty/vllm/model_executor/layers/sampler.py
        
        公式：energy_score = logsumexp(logits, dim=-1)
        
        Args:
            logits: 模型输出的logits，shape [vocab_size] 或 [batch, vocab_size]
            
        Returns:
            float: energy score
        """
        if not TORCH_AVAILABLE:
            return 0.0
        
        energy = torch.logsumexp(logits, dim=-1)
        
        # 如果是batch，取平均
        if energy.dim() > 0:
            energy = energy.mean()
        
        return energy.item()
    
    def estimate_text_uncertainty(self, text: str = None, 
                                  embeddings: torch.Tensor = None,
                                  log_probs: List[float] = None,
                                  return_details: bool = False) -> Union[float, Dict]:
        """
        估计文本不确定性（SeaKR优化版本）
        
        使用SeaKR的方法：
        1. 如果有多个样本的embeddings：计算eigen_score
        2. 如果有log_probs：计算perplexity
        3. 两者结合或单独使用
        
        Args:
            text: 文本（可选，用于简化版）
            embeddings: token embeddings，shape [k, d]
            log_probs: token的对数概率列表
            return_details: 是否返回详细信息
            
        Returns:
            float 或 dict: 不确定性分数
        """
        if not TORCH_AVAILABLE:
            return self._estimate_text_simplified(text)
        
        uncertainty_dict = {}
        
        try:
            # 方法1：使用eigen_score（需要多个样本的embeddings）
            if embeddings is not None and embeddings.shape[0] > 1:
                eigen_score = self.compute_eigen_score(embeddings)
                uncertainty_dict['eigen_score'] = eigen_score
                
                # 归一化：SeaKR的eigen_score通常在[-10, 0]范围
                # 转换为[0, 1]：(score + 10) / 10
                normalized_eigen = max(0.0, min(1.0, (eigen_score + 10) / 10))
                uncertainty_dict['eigen_uncertainty'] = normalized_eigen
            
            # 方法2：使用perplexity
            if log_probs is not None:
                perplexity = self.compute_perplexity(log_probs)
                uncertainty_dict['perplexity'] = perplexity
                
                # 归一化：perplexity通常在[1, 100+]范围
                # 使用log scale: log(perplexity) / log(100)
                normalized_perplexity = min(1.0, np.log(perplexity) / np.log(100))
                uncertainty_dict['perplexity_uncertainty'] = normalized_perplexity
            
            # 综合不确定性
            if 'eigen_uncertainty' in uncertainty_dict and 'perplexity_uncertainty' in uncertainty_dict:
                # 两者都有，取最大值（更保守）
                combined_uncertainty = max(
                    uncertainty_dict['eigen_uncertainty'],
                    uncertainty_dict['perplexity_uncertainty']
                )
            elif 'eigen_uncertainty' in uncertainty_dict:
                combined_uncertainty = uncertainty_dict['eigen_uncertainty']
            elif 'perplexity_uncertainty' in uncertainty_dict:
                combined_uncertainty = uncertainty_dict['perplexity_uncertainty']
            else:
                # 都没有，使用简化版
                combined_uncertainty = self._estimate_text_simplified(text)
            
            if return_details:
                uncertainty_dict['combined'] = combined_uncertainty
                return uncertainty_dict
            else:
                return combined_uncertainty
        
        except Exception as e:
            warnings.warn(f"文本不确定性计算失败: {e}")
            return self._estimate_text_simplified(text)
    
    def _estimate_text_simplified(self, text: str) -> float:
        """简化版文本不确定性估计（不需要模型）"""
        if not text:
            return 1.0
        
        # 基于文本长度的启发式
        # 短文本通常更不确定
        text_length = len(text.split())
        uncertainty = max(0.0, min(1.0, 1.0 - (text_length / 50)))
        
        return uncertainty
    
    def estimate_visual_uncertainty(self, image, 
                                    attention_weights: torch.Tensor = None,
                                    return_details: bool = False) -> float:
        """
        估计视觉不确定性（我们的创新）
        
        方法：基于视觉注意力分布的方差
        - attention variance越大，不确定性越高
        
        这是我们的创新，SeaKR没有视觉模态
        
        Args:
            image: 图像（可选）
            attention_weights: 注意力权重，shape [n_heads, n_tokens]
            return_details: 是否返回详细信息
            
        Returns:
            float: 视觉不确定性 [0, 1]
        """
        if not TORCH_AVAILABLE or attention_weights is None:
            return 0.5  # 默认中等不确定性
        
        try:
            # 计算attention分布的方差
            attention_variance = torch.var(attention_weights, dim=-1)
            visual_uncertainty = torch.mean(attention_variance).item()
            
            # 归一化到[0, 1]
            normalized_uncertainty = min(1.0, visual_uncertainty)
            
            return normalized_uncertainty
        
        except Exception as e:
            warnings.warn(f"视觉不确定性计算失败: {e}")
            return 0.5
    
    def estimate_alignment_uncertainty(self, text_embeddings: torch.Tensor = None,
                                      visual_embeddings: torch.Tensor = None,
                                      return_details: bool = False) -> float:
        """
        估计跨模态对齐不确定性（我们的创新）
        
        方法：Jensen-Shannon散度
        - 文本和视觉嵌入的分布差异越大，不确定性越高
        
        这是我们的创新，SeaKR没有跨模态场景
        
        Args:
            text_embeddings: 文本嵌入
            visual_embeddings: 视觉嵌入
            return_details: 是否返回详细信息
            
        Returns:
            float: 对齐不确定性 [0, 1]
        """
        if not TORCH_AVAILABLE or text_embeddings is None or visual_embeddings is None:
            return 0.0  # 默认无不确定性
        
        try:
            # 转换为概率分布
            text_dist = F.softmax(text_embeddings, dim=-1)
            visual_dist = F.softmax(visual_embeddings, dim=-1)
            
            # 计算JS散度
            js_div = self._jensen_shannon_divergence(text_dist, visual_dist)
            
            # JS散度范围[0, 1]，已经是归一化的
            return js_div.item()
        
        except Exception as e:
            warnings.warn(f"对齐不确定性计算失败: {e}")
            return 0.0
    
    def _jensen_shannon_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        计算Jensen-Shannon散度
        
        公式：JS(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M)
        其中：M = 0.5*(P + Q)
        """
        m = 0.5 * (p + q)
        
        # KL散度：sum(P * log(P/Q))
        kl_pm = torch.sum(p * torch.log((p + 1e-10) / (m + 1e-10)), dim=-1)
        kl_qm = torch.sum(q * torch.log((q + 1e-10) / (m + 1e-10)), dim=-1)
        
        js = 0.5 * kl_pm + 0.5 * kl_qm
        
        return torch.mean(js)
    
    def estimate(self, text: str = None, image=None,
                text_embeddings: torch.Tensor = None,
                visual_embeddings: torch.Tensor = None,
                log_probs: List[float] = None,
                attention_weights: torch.Tensor = None,
                return_details: bool = False) -> Dict[str, float]:
        """
        估计完整的跨模态不确定性
        
        综合SeaKR的方法和我们的多模态扩展
        
        Args:
            text: 文本查询
            image: 图像查询
            text_embeddings: 文本嵌入（用于eigen_score）
            visual_embeddings: 视觉嵌入
            log_probs: token概率
            attention_weights: 注意力权重
            return_details: 是否返回详细信息
            
        Returns:
            dict: {
                'text': 文本不确定性,
                'visual': 视觉不确定性,
                'alignment': 对齐不确定性,
                'total': 总不确定性,
                'eigen_score': eigen_score（如果计算了）
            }
        """
        uncertainties = {}
        
        # 1. 文本不确定性（SeaKR方法）
        text_unc = self.estimate_text_uncertainty(
            text=text,
            embeddings=text_embeddings,
            log_probs=log_probs,
            return_details=return_details
        )
        
        if isinstance(text_unc, dict):
            uncertainties.update(text_unc)
            text_unc = text_unc.get('combined', 0.5)
        
        uncertainties['text'] = text_unc
        
        # 2. 视觉不确定性（我们的创新）
        if image is not None or attention_weights is not None:
            visual_unc = self.estimate_visual_uncertainty(
                image=image,
                attention_weights=attention_weights
            )
            uncertainties['visual'] = visual_unc
        else:
            uncertainties['visual'] = 0.0
        
        # 3. 跨模态对齐不确定性（我们的创新）
        if text_embeddings is not None and visual_embeddings is not None:
            alignment_unc = self.estimate_alignment_uncertainty(
                text_embeddings=text_embeddings,
                visual_embeddings=visual_embeddings
            )
            uncertainties['alignment'] = alignment_unc
        else:
            uncertainties['alignment'] = 0.0
        
        # 4. 总不确定性（加权组合）
        total_uncertainty = (
            self.alpha * uncertainties['text'] +
            self.beta * uncertainties['visual'] +
            self.gamma * uncertainties['alignment']
        )
        uncertainties['total'] = total_uncertainty
        
        return uncertainties
    
    def should_retrieve(self, uncertainties: Dict[str, float] = None,
                       eigen_score: float = None,
                       threshold: float = None) -> Tuple[bool, Optional[str]]:
        """
        判断是否需要检索（SeaKR的自适应触发）
        
        参考：SeaKR-main/SEAKR/reasoner.py 第379-380行
        判断逻辑：if eigen_score > eigen_threshold: 需要检索
        
        Args:
            uncertainties: 不确定性字典
            eigen_score: eigen_score值（如果有）
            threshold: 自定义阈值（可选）
            
        Returns:
            (bool, str): (是否检索, 检索模态)
        """
        # 方法1：使用SeaKR的eigen_score判断
        if eigen_score is not None:
            # SeaKR的判断：eigen_score > -6.0 表示不确定，需要检索
            if eigen_score > self.eigen_threshold:
                # 根据不确定性来源选择检索模态
                if uncertainties:
                    modality = self.select_retrieval_modality(uncertainties)
                else:
                    modality = 'both'  # 默认都检索
                
                return True, modality
            else:
                return False, None
        
        # 方法2：使用总不确定性判断
        if uncertainties and 'total' in uncertainties:
            threshold = threshold or self.uncertainty_threshold
            
            if uncertainties['total'] > threshold:
                modality = self.select_retrieval_modality(uncertainties)
                return True, modality
            else:
                return False, None
        
        # 默认不检索
        return False, None
    
    def select_retrieval_modality(self, uncertainties: Dict[str, float]) -> str:
        """
        选择检索模态
        
        基于各模态的不确定性决定检索什么
        
        Args:
            uncertainties: 不确定性字典
            
        Returns:
            str: 'text', 'image', 'both'
        """
        text_unc = uncertainties.get('text', 0.0)
        visual_unc = uncertainties.get('visual', 0.0)
        
        # 如果两者都高，检索两者
        if text_unc > 0.6 and visual_unc > 0.6:
            return 'both'
        
        # 哪个不确定性高，检索哪个
        if text_unc > visual_unc:
            return 'text'
        elif visual_unc > text_unc:
            return 'image'
        else:
            return 'both'


# 向后兼容：保留原始类名
CrossModalUncertaintyEstimator = SeaKROptimizedUncertaintyEstimator


