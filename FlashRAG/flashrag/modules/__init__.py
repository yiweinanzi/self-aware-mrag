# -*- coding: utf-8 -*-
"""
FlashRAG创新模块

包含：
1. CrossModalUncertaintyEstimator - 跨模态不确定性估计（SeaKR扩展）
2. FineGrainedMultimodalAttribution - 细粒度多模态归因
3. MultimodalOutputComposition - 多模态输出组合（MRAG 3.0）
4. PositionAwareCrossModalFusion - 位置感知跨模态融合（新增）
5. MLLMWrapper - MLLM模型封装器（LLaVA/Qwen-VL）
"""

try:
    from flashrag.modules.uncertainty_estimator import CrossModalUncertaintyEstimator
except ImportError:
    import warnings
    warnings.warn("CrossModalUncertaintyEstimator导入失败，可能缺少依赖")

try:
    from flashrag.modules.attribution import FineGrainedMultimodalAttribution
except ImportError:
    pass

try:
    from flashrag.modules.multimodal_output import MultimodalOutputComposition
except ImportError:
    pass

# 位置感知融合模块（新增）
try:
    from flashrag.modules.position_aware_fusion import (
        PositionAwareCrossModalFusion,
        LearnedPositionalEncoding,
        create_position_aware_fusion
    )
    POSITION_FUSION_AVAILABLE = True
except ImportError:
    POSITION_FUSION_AVAILABLE = False
    import warnings
    warnings.warn("PositionAwareCrossModalFusion导入失败，可能缺少torch")

# MLLM封装器（可选导入）
try:
    from flashrag.modules.mllm_wrapper import (
        MLLMWrapper,
        LLaVAWrapper,
        QwenVLWrapper
    )
    MLLM_AVAILABLE = True
except ImportError:
    MLLM_AVAILABLE = False
    import warnings
    warnings.warn("MLLM Wrapper导入失败，LLaVA可能未安装")

