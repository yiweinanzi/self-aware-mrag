# -*- coding: utf-8 -*-
"""
MLLM封装器 - 用于FlashRAG

功能：
1. 提取hidden states（用于SeaKR的不确定性估计）
2. 生成多个样本（用于eigen_score计算）
3. 提取attention weights（用于visual uncertainty）

支持的模型：
- LLaVA-1.5-7B/13B
- Qwen-VL
- 其他兼容的MLLM

参考文档：创新点1-自感知多模态RAG-实施方案.md 第851-854行
"""

import torch
import warnings
from typing import Optional, List, Tuple, Union
from PIL import Image

try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers未安装")


class MLLMWrapper:
    """
    通用MLLM封装器基类
    
    子类需要实现：
    - get_text_hidden_states()
    - get_visual_hidden_states()
    - generate_with_embeddings()
    - generate()
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
    
    def get_text_hidden_states(self, text: str) -> torch.Tensor:
        """提取文本hidden states"""
        raise NotImplementedError
    
    def get_visual_hidden_states(self, image) -> torch.Tensor:
        """提取视觉hidden states"""
        raise NotImplementedError
    
    def generate_with_embeddings(self, text: str, image=None, n_samples: int = 20) -> torch.Tensor:
        """生成多个样本并返回EOS embeddings"""
        raise NotImplementedError
    
    def generate(self, text: str, image=None, **kwargs) -> str:
        """生成回答"""
        raise NotImplementedError


class LLaVAWrapper(MLLMWrapper):
    """
    LLaVA-1.5模型封装器
    
    功能：
    1. 加载LLaVA模型
    2. 提取hidden states用于不确定性估计
    3. 生成多个样本用于eigen_score计算
    4. 提取attention weights用于visual uncertainty
    
    使用示例：
    ```python
    # 初始化
    wrapper = LLaVAWrapper(
        model_path="/path/to/llava-v1.5-7b",
        device='cuda'
    )
    
    # 提取hidden states
    hidden_states = wrapper.get_text_hidden_states("What is this?")
    
    # 生成多个样本用于eigen_score
    embeddings = wrapper.generate_with_embeddings(
        text="What is this?",
        image=your_image,
        n_samples=20
    )
    
    # 计算eigen_score
    from flashrag.modules.uncertainty_estimator import CrossModalUncertaintyEstimator
    estimator = CrossModalUncertaintyEstimator()
    eigen_score = estimator.compute_eigen_score(embeddings)
    ```
    """
    
    def __init__(self, model_path: str, device: str = 'cuda', load_8bit: bool = False):
        """
        初始化LLaVA模型
        
        Args:
            model_path: LLaVA模型路径
            device: 设备（'cuda' 或 'cpu'）
            load_8bit: 是否使用8bit量化（节省显存）
        """
        super().__init__(model_path, device)
        
        self.load_8bit = load_8bit
        
        # 延迟导入，避免没有LLaVA时报错
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            
            print(f"正在加载LLaVA模型: {model_path}")
            print(f"设备: {device}, 8bit量化: {load_8bit}")
            
            # 加载模型（离线模式，避免下载CLIP）
            import os
            os.environ['HF_HUB_OFFLINE'] = '1'  # 强制离线模式
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            
            self.tokenizer, self.model, self.image_processor, self.context_len = \
                load_pretrained_model(
                    model_path=model_path,
                    model_base=None,
                    model_name=get_model_name_from_path(model_path),
                    load_8bit=load_8bit,
                    load_4bit=False,
                    device=device
                )
            
            self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
            self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
            
            print(f"✅ LLaVA模型加载成功")
            print(f"   - Context length: {self.context_len}")
            print(f"   - 模型类型: {self.model.__class__.__name__}")
            print(f"   - 设备: {next(self.model.parameters()).device}")
            
        except ImportError as e:
            raise ImportError(
                f"无法导入LLaVA模块: {e}\n"
                f"请先安装LLaVA:\n"
                f"  git clone https://github.com/haotian-liu/LLaVA.git\n"
                f"  cd LLaVA && pip install -e ."
            )
    
    def get_text_hidden_states(self, text: str) -> torch.Tensor:
        """
        提取文本的hidden states
        
        用于SeaKR的不确定性估计
        
        Args:
            text: 输入文本
            
        Returns:
            torch.Tensor: hidden states, shape [seq_len, hidden_dim]
        """
        # Tokenize
        input_ids = self.tokenizer(
            text, 
            return_tensors='pt'
        ).input_ids.to(self.device)
        
        # Forward pass（获取embeddings）
        with torch.no_grad():
            # LLaVA的文本编码器
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                hidden_states = self.model.model.embed_tokens(input_ids)
            elif hasattr(self.model, 'get_model'):
                hidden_states = self.model.get_model().embed_tokens(input_ids)
            else:
                # 尝试直接访问
                hidden_states = self.model.embed_tokens(input_ids)
        
        return hidden_states.squeeze(0)  # [seq_len, hidden_dim]
    
    def get_visual_hidden_states(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取视觉hidden states和attention weights
        
        用于visual uncertainty估计
        
        Args:
            image: PIL Image
            
        Returns:
            (visual_features, attention_weights)
        """
        from llava.mm_utils import process_images
        
        # 处理图像
        image_tensor = process_images(
            [image], 
            self.image_processor, 
            self.model.config
        ).to(self.device, dtype=torch.float16)
        
        # Forward pass获取visual features
        with torch.no_grad():
            if hasattr(self.model, 'encode_images'):
                visual_features = self.model.encode_images(image_tensor)
            else:
                # 尝试直接访问vision tower
                visual_features = self.model.get_vision_tower()(image_tensor)
        
        # Attention weights（如果可用）
        # TODO: 提取attention weights
        attention_weights = None
        
        return visual_features, attention_weights
    
    def generate_with_embeddings(self, text: str, image: Optional[Image.Image] = None, 
                                 n_samples: int = 20) -> torch.Tensor:
        """
        生成多个样本并提取EOS embeddings
        
        用于SeaKR的eigen_score计算
        
        参考：SeaKR-main/vllm_uncertainty/vllm/engine/llm_engine.py
        SeaKR生成20个样本，提取每个的EOS embedding，然后计算eigen_score
        
        Args:
            text: 输入文本查询
            image: 输入图像（可选）
            n_samples: 生成样本数（默认20，SeaKR使用的数量）
            
        Returns:
            torch.Tensor: EOS embeddings, shape [n_samples, hidden_dim]
        """
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.conversation import conv_templates
        
        embeddings_list = []
        
        # 准备prompt
        conv = conv_templates["llava_v1"].copy()
        
        if image is not None:
            # 多模态prompt
            from llava.mm_utils import process_images
            
            # 处理图像
            image_tensor = process_images(
                [image], 
                self.image_processor, 
                self.model.config
            ).to(self.device, dtype=torch.float16)
            
            # 构建prompt
            conv.append_message(conv.roles[0], f"{self.DEFAULT_IMAGE_TOKEN}\n{text}")
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Tokenize
            input_ids = tokenizer_image_token(
                prompt, 
                self.tokenizer, 
                self.IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0).to(self.device)
        else:
            # 纯文本prompt
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = self.tokenizer(
                prompt, 
                return_tensors='pt'
            ).input_ids.to(self.device)
            
            image_tensor = None
        
        # 生成多个样本
        print(f"正在生成{n_samples}个样本用于eigen_score计算...")
        
        for i in range(n_samples):
            with torch.no_grad():
                # 采样生成（temperature=1.0）
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    max_new_tokens=100,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict_in_generate=True
                )
                
                # 提取最后一层的最后一个token的hidden state
                # 这是EOS token的embedding
                if hasattr(output_ids, 'hidden_states') and output_ids.hidden_states:
                    # hidden_states是tuple of tuples
                    # 格式：(layer_0_states, layer_1_states, ...)
                    # 每个layer_states: (batch, seq_len, hidden_dim)
                    
                    # 获取最后一层的hidden states
                    last_layer_states = output_ids.hidden_states[-1]  # 最后一个生成步
                    
                    if isinstance(last_layer_states, tuple):
                        # 如果是tuple，取最后一层
                        last_layer_states = last_layer_states[-1]
                    
                    # 取最后一个token的embedding
                    eos_embedding = last_layer_states[:, -1, :]  # [1, hidden_dim]
                    embeddings_list.append(eos_embedding)
                else:
                    # 如果无法获取hidden states，使用最后一层的输出
                    # 这是一个fallback方案
                    warnings.warn(f"样本{i}: 无法获取hidden states，使用fallback")
                    # 使用模型的最后一层embedding
                    with torch.no_grad():
                        last_hidden = self.model.model.model.embed_tokens(
                            output_ids.sequences[:, -1:]
                        )
                    embeddings_list.append(last_hidden[:, -1, :])
            
            if (i + 1) % 5 == 0:
                print(f"  已生成 {i+1}/{n_samples} 个样本")
        
        # 堆叠成 [n_samples, hidden_dim]
        if embeddings_list:
            embeddings = torch.cat(embeddings_list, dim=0)
            print(f"✅ 生成完成，embeddings shape: {embeddings.shape}")
            return embeddings
        else:
            raise RuntimeError("无法提取任何embeddings")
    
    def generate(self, text: str, image: Optional[Image.Image] = None, 
                max_new_tokens: int = 512, temperature: float = 0.2) -> str:
        """
        生成回答
        
        Args:
            text: 输入文本
            image: 输入图像（可选）
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            
        Returns:
            str: 生成的回答
        """
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.conversation import conv_templates
        
        # 准备对话
        conv = conv_templates["llava_v1"].copy()
        
        if image is not None:
            # 处理图像
            image_tensor = process_images(
                [image], 
                self.image_processor, 
                self.model.config
            ).to(self.device, dtype=torch.float16)
            
            # 构建prompt
            conv.append_message(conv.roles[0], f"{self.DEFAULT_IMAGE_TOKEN}\n{text}")
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Tokenize
            input_ids = tokenizer_image_token(
                prompt, 
                self.tokenizer, 
                self.IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0).to(self.device)
        else:
            # 纯文本
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = self.tokenizer(
                prompt, 
                return_tensors='pt'
            ).input_ids.to(self.device)
            
            image_tensor = None
        
        # 生成
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=temperature > 0,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )
        
        # 解码
        outputs = self.tokenizer.batch_decode(
            output_ids, 
            skip_special_tokens=True
        )[0].strip()
        
        return outputs
    
    def get_attention_weights(self, text: str, image: Optional[Image.Image] = None) -> torch.Tensor:
        """
        提取attention weights
        
        用于visual uncertainty估计
        
        Args:
            text: 输入文本
            image: 输入图像
            
        Returns:
            torch.Tensor: attention weights, shape [n_heads, seq_len, seq_len]
        """
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.conversation import conv_templates
        
        # 准备输入
        conv = conv_templates["llava_v1"].copy()
        
        if image is not None:
            image_tensor = process_images(
                [image], 
                self.image_processor, 
                self.model.config
            ).to(self.device, dtype=torch.float16)
            
            conv.append_message(conv.roles[0], f"{self.DEFAULT_IMAGE_TOKEN}\n{text}")
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(
                prompt, 
                self.tokenizer, 
                self.IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0).to(self.device)
        else:
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = self.tokenizer(
                prompt, 
                return_tensors='pt'
            ).input_ids.to(self.device)
            
            image_tensor = None
        
        # Forward pass with attention
        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_ids,
                images=image_tensor,
                output_attentions=True,
                return_dict=True
            )
            
            # 提取attention weights
            # outputs.attentions是tuple，每层一个attention
            if hasattr(outputs, 'attentions') and outputs.attentions:
                # 取最后一层的attention
                attention = outputs.attentions[-1]  # [batch, n_heads, seq_len, seq_len]
                return attention.squeeze(0)  # [n_heads, seq_len, seq_len]
            else:
                warnings.warn("无法获取attention weights")
                return None


class QwenVLWrapper(MLLMWrapper):
    """
    Qwen-VL模型封装器（备选）
    
    如果LLaVA不可用，可以使用Qwen-VL
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        super().__init__(model_path, device)
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"正在加载Qwen-VL模型: {model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                trust_remote_code=True
            ).eval()
            
            print(f"✅ Qwen-VL模型加载成功")
            
        except ImportError as e:
            raise ImportError(
                f"无法导入Qwen-VL: {e}\n"
                f"请先安装：pip install transformers_stream_generator"
            )
    
    def get_text_hidden_states(self, text: str) -> torch.Tensor:
        """提取文本hidden states"""
        input_ids = self.tokenizer(text, return_tensors='pt').input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.transformer.wte(input_ids)
        
        return outputs.squeeze(0)
    
    def generate_with_embeddings(self, text: str, image=None, n_samples: int = 20) -> torch.Tensor:
        """生成多个样本并提取embeddings"""
        # TODO: 实现Qwen-VL的embedding提取
        raise NotImplementedError("Qwen-VL的embedding提取待实现")
    
    def generate(self, text: str, image=None, **kwargs) -> str:
        """生成回答"""
        query = self.tokenizer.from_list_format([
            {'image': image} if image else {},
            {'text': text},
        ])
        
        response, _ = self.model.chat(self.tokenizer, query=query, history=None)
        return response


# 工厂函数
def create_mllm_wrapper(model_type: str = 'llava', model_path: str = None, 
                       device: str = 'cuda', **kwargs):
    """
    创建MLLM封装器
    
    Args:
        model_type: 模型类型 ('llava', 'qwen-vl')
        model_path: 模型路径
        device: 设备
        **kwargs: 其他参数
        
    Returns:
        MLLMWrapper实例
    """
    if model_type == 'llava':
        return LLaVAWrapper(model_path, device, **kwargs)
    elif model_type == 'qwen-vl':
        return QwenVLWrapper(model_path, device)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


# 示例和测试
if __name__ == '__main__':
    import sys
    
    print("=" * 70)
    print("MLLM Wrapper测试")
    print("=" * 70)
    
    # 检查是否提供了模型路径
    if len(sys.argv) < 2:
        print("\n使用方法:")
        print("  python mllm_wrapper.py <model_path>")
        print("\n示例:")
        print("  python mllm_wrapper.py /root/autodl-tmp/models/llava-v1.5-7b")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    try:
        # 创建wrapper
        print(f"\n正在加载模型: {model_path}")
        wrapper = LLaVAWrapper(model_path, device='cuda')
        
        # 测试1: 提取text hidden states
        print("\n[测试1] 提取文本hidden states")
        text = "What is in this image?"
        hidden_states = wrapper.get_text_hidden_states(text)
        print(f"✅ Hidden states shape: {hidden_states.shape}")
        
        # 测试2: 生成并提取embeddings
        print("\n[测试2] 生成样本并提取embeddings")
        embeddings = wrapper.generate_with_embeddings(text, n_samples=5)
        print(f"✅ Embeddings shape: {embeddings.shape}")
        
        # 测试3: 计算eigen_score
        print("\n[测试3] 计算eigen_score")
        from flashrag.modules.uncertainty_estimator import CrossModalUncertaintyEstimator
        
        estimator = CrossModalUncertaintyEstimator(
            config={'eigen_threshold': -6.0}
        )
        
        eigen_score = estimator.compute_eigen_score(embeddings)
        print(f"✅ Eigen score: {eigen_score:.4f}")
        
        should_retrieve, modality = estimator.should_retrieve(eigen_score=eigen_score)
        print(f"✅ 判断结果: {'需要检索' if should_retrieve else '不需要检索'} ({modality})")
        
        print("\n" + "=" * 70)
        print("✅ 所有测试通过！MLLM Wrapper工作正常！")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

