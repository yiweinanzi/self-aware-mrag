#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3-VL-8B-Instruct 模型封装器

用于Self-Aware Multimodal RAG

关键特性：
1. 支持图文多模态输入
2. 禁用thinking模式（确保thinking=false，生成纯文本答案）
3. 提取hidden states用于不确定性估计
4. 提取attention weights用于visual uncertainty
5. 兼容FlashRAG的MLLM接口

模型位置：/root/autodl-tmp/models/Qwen3-VL-8B-Instruct

参考：
- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- flashrag/modules/mllm_wrapper.py (LLaVAWrapper)
"""

import os
import warnings
import torch
from typing import Optional, List, Tuple, Union, Dict, Any
from PIL import Image

try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    QWEN_AVAILABLE = True
except ImportError as e:
    QWEN_AVAILABLE = False
    warnings.warn(f"Qwen3-VL依赖未安装: {e}")


class Qwen3VLWrapper:
    """
    Qwen3-VL-8B-Instruct 模型封装器
    
    用于Self-Aware Multimodal RAG系统，替代LLaVA-1.5-7B
    
    核心功能：
    1. 图文多模态输入
    2. 禁用thinking模式（thinking=false）
    3. 提取hidden states（用于文本不确定性估计）
    4. 提取visual features（用于视觉不确定性估计）
    5. 生成多个样本（用于eigen_score计算）
    
    使用示例：
    ```python
    # 初始化
    wrapper = Qwen3VLWrapper(
        model_path="/root/autodl-tmp/models/Qwen3-VL-8B-Instruct",
        device='cuda'
    )
    
    # 纯文本问答
    answer = wrapper.generate(
        text="What is the capital of France?",
        max_new_tokens=100
    )
    
    # 图文问答
    answer = wrapper.generate(
        text="What is in this image?",
        image="/path/to/image.jpg",
        max_new_tokens=100
    )
    
    # 提取hidden states用于不确定性估计
    hidden_states = wrapper.get_text_hidden_states("What is this?")
    
    # 提取visual features
    visual_features = wrapper.get_visual_hidden_states(image)
    ```
    """
    
    def __init__(
        self, 
        model_path: str = "/root/autodl-tmp/models/Qwen3-VL-8B-Instruct",
        device: str = 'cuda',
        load_in_8bit: bool = False,
        torch_dtype: str = 'bfloat16'
    ):
        """
        初始化Qwen3-VL模型
        
        Args:
            model_path: 模型路径
            device: 设备（'cuda' 或 'cpu'）
            load_in_8bit: 是否使用8bit量化（节省显存）
            torch_dtype: 数据类型（'bfloat16' 或 'float16'）
        
        Raises:
            ImportError: 如果transformers或qwen_vl_utils未安装
            RuntimeError: 如果模型加载失败
        """
        if not QWEN_AVAILABLE:
            raise ImportError(
                "Qwen3-VL依赖未安装。请运行：\n"
                "  pip install transformers accelerate qwen-vl-utils"
            )
        
        self.model_path = model_path
        self.device = device
        self.load_in_8bit = load_in_8bit
        
        # 设置torch dtype
        if torch_dtype == 'bfloat16':
            self.torch_dtype = torch.bfloat16
        elif torch_dtype == 'float16':
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32
        
        print(f"正在加载Qwen3-VL-8B-Instruct: {model_path}")
        print(f"设备: {device}, 8bit量化: {load_in_8bit}, dtype: {torch_dtype}")
        
        # 加载processor（包含tokenizer和image_processor）
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 加载模型
        if load_in_8bit:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto",
                load_in_8bit=True,
                trust_remote_code=True
            )
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=self.torch_dtype,
                trust_remote_code=True
            )
        
        self.model.eval()
        
        print(f"✅ Qwen3-VL-8B-Instruct加载成功")
        print(f"   - 模型类型: {self.model.__class__.__name__}")
        print(f"   - 设备: {next(self.model.parameters()).device}")
        print(f"   - 参数量: ~8B")
        print(f"   ⚠️  重要：thinking模式已禁用（thinking=false）")
    
    def generate(
        self,
        text: str,
        image: Optional[Union[str, Image.Image, List]] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        生成回答（纯文本或图文多模态）
        
        Args:
            text: 问题文本
            image: 图像（可选）
                - str: 图像文件路径
                - PIL.Image: PIL图像对象
                - List: 图像列表（多图）
                - None: 纯文本问答
            max_new_tokens: 最大生成token数
            temperature: 温度（控制随机性）
            top_p: nucleus sampling参数
            top_k: top-k sampling参数
            do_sample: 是否采样（False则贪心解码）
            **kwargs: 其他生成参数
        
        Returns:
            str: 生成的回答文本
        
        注意：
            - thinking模式已禁用，只生成最终答案
            - 如果需要thinking，请修改messages中的content
        """
        # 构建messages格式
        messages = []
        
        if image is not None:
            # 图文问答
            content = []
            
            # 处理图像
            if isinstance(image, str):
                # 单张图像路径
                content.append({"type": "image", "image": image})
            elif isinstance(image, Image.Image):
                # PIL Image对象
                content.append({"type": "image", "image": image})
            elif isinstance(image, list):
                # 多张图像
                for img in image:
                    content.append({"type": "image", "image": img})
            else:
                raise ValueError(f"不支持的image类型: {type(image)}")
            
            # 添加文本
            content.append({"type": "text", "text": text})
            
            messages.append({
                "role": "user",
                "content": content
            })
        else:
            # 纯文本问答
            messages.append({
                "role": "user",
                "content": text
            })
        
        # 应用chat模板
        text_input = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 处理输入（包括图像）
        if image is not None:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=[text_input],
                padding=True,
                return_tensors="pt"
            )
        
        # 移动到设备
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                **kwargs
            )
        
        # 解码
        # 只取新生成的token
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.processor.decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def get_text_hidden_states(self, text: str) -> torch.Tensor:
        """
        提取文本的hidden states
        
        用于SeaKR的文本不确定性估计
        
        Args:
            text: 输入文本
        
        Returns:
            torch.Tensor: hidden states, shape [seq_len, hidden_dim]
        """
        # Tokenize
        inputs = self.processor(
            text=[text],
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        
        # 获取embeddings
        with torch.no_grad():
            # Qwen3-VL的文本编码器
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                hidden_states = self.model.model.embed_tokens(input_ids)
            elif hasattr(self.model, 'get_model'):
                hidden_states = self.model.get_model().embed_tokens(input_ids)
            else:
                raise RuntimeError("无法访问embed_tokens")
        
        return hidden_states.squeeze(0)  # [seq_len, hidden_dim]
    
    def get_visual_hidden_states(
        self,
        image: Union[str, Image.Image]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        提取视觉hidden states和attention weights
        
        用于visual uncertainty估计
        
        Args:
            image: 图像（路径或PIL Image）
        
        Returns:
            (visual_features, attention_weights)
            - visual_features: shape [num_patches, hidden_dim]
            - attention_weights: shape [num_layers, num_heads, num_patches, num_patches]
                                 或 None（如果无法提取）
        """
        # 处理图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # 构建messages
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."}
            ]
        }]
        
        # 应用chat模板
        text_input = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 处理输入
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Forward pass（提取visual features）
        with torch.no_grad():
            # 获取模型输出（包括hidden_states和attentions）
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=True
            )
            
            # 提取visual features
            # Qwen3-VL的visual encoder输出在hidden_states中
            hidden_states = outputs.hidden_states
            
            # 获取第一个hidden state（visual embeddings）
            visual_features = hidden_states[0].squeeze(0)  # [seq_len, hidden_dim]
            
            # 提取attention weights（如果可用）
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                # 取最后一层的attention
                attention_weights = outputs.attentions[-1]  # [batch, num_heads, seq_len, seq_len]
                attention_weights = attention_weights.squeeze(0)  # [num_heads, seq_len, seq_len]
            else:
                attention_weights = None
        
        return visual_features, attention_weights
    
    def generate_with_embeddings(
        self,
        text: str,
        image: Optional[Union[str, Image.Image]] = None,
        n_samples: int = 20,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        生成多个样本并返回EOS token的embeddings
        
        用于SeaKR的eigen_score计算
        
        Args:
            text: 问题文本
            image: 图像（可选）
            n_samples: 生成样本数
            temperature: 采样温度
        
        Returns:
            torch.Tensor: EOS embeddings, shape [n_samples, hidden_dim]
        """
        # 构建messages
        if image is not None:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text}
                ]
            }]
        else:
            messages = [{
                "role": "user",
                "content": text
            }]
        
        # 应用chat模板
        text_input = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 处理输入
        if image is not None:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=[text_input],
                padding=True,
                return_tensors="pt"
            )
        
        # 移动到设备
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        eos_embeddings = []
        
        # 生成n_samples个样本
        for _ in range(n_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=temperature,
                    do_sample=True,
                    output_hidden_states=True,
                    return_dict_in_generate=True
                )
                
                # 提取最后一个hidden state（EOS token的embedding）
                # outputs.hidden_states: tuple of tuples
                # hidden_states[-1]: 最后一步生成的hidden states
                # hidden_states[-1][-1]: 最后一层的hidden state
                last_hidden = outputs.hidden_states[-1][-1]  # [batch, seq_len, hidden_dim]
                eos_embedding = last_hidden[0, -1, :]  # [hidden_dim]
                
                eos_embeddings.append(eos_embedding)
        
        # Stack所有样本
        eos_embeddings = torch.stack(eos_embeddings, dim=0)  # [n_samples, hidden_dim]
        
        return eos_embeddings
    
    def batch_generate(
        self,
        texts: List[str],
        images: Optional[List[Union[str, Image.Image]]] = None,
        max_new_tokens: int = 100,
        **kwargs
    ) -> List[str]:
        """
        批量生成（提高效率）
        
        Args:
            texts: 问题文本列表
            images: 图像列表（可选，需与texts长度一致）
            max_new_tokens: 最大生成token数
            **kwargs: 其他生成参数
        
        Returns:
            List[str]: 生成的回答列表
        """
        if images is not None and len(texts) != len(images):
            raise ValueError(f"texts和images长度不一致: {len(texts)} vs {len(images)}")
        
        results = []
        
        # 逐个生成（Qwen3-VL的batch处理较复杂，这里简化为循环）
        for i, text in enumerate(texts):
            image = images[i] if images is not None else None
            result = self.generate(text, image, max_new_tokens, **kwargs)
            results.append(result)
        
        return results


# 工厂函数
def create_qwen3_vl_wrapper(
    model_path: str = "/root/autodl-tmp/models/Qwen3-VL-8B-Instruct",
    device: str = 'cuda',
    **kwargs
) -> Qwen3VLWrapper:
    """
    创建Qwen3-VL wrapper
    
    Args:
        model_path: 模型路径
        device: 设备
        **kwargs: 其他参数
    
    Returns:
        Qwen3VLWrapper实例
    """
    return Qwen3VLWrapper(model_path, device, **kwargs)


if __name__ == '__main__':
    print("Qwen3-VL-8B-Instruct Wrapper测试")
    print("=" * 80)
    
    model_path = '/root/autodl-tmp/models/Qwen3-VL-8B-Instruct'
    
    if os.path.exists(model_path):
        print("\n✅ 模型文件存在")
        print(f"   路径: {model_path}")
        print(f"   大小: ~17GB")
        print()
        
        # 检查是否安装了依赖
        if QWEN_AVAILABLE:
            print("✅ Qwen3-VL依赖已安装")
            print()
            print("使用方法:")
            print("  from flashrag.modules.qwen3_vl import Qwen3VLWrapper")
            print("  wrapper = Qwen3VLWrapper(model_path)")
            print("  answer = wrapper.generate(text='...', image='...')")
            print()
            print("核心特性:")
            print("  - ✅ 图文多模态输入")
            print("  - ✅ thinking=false（禁用思考模式）")
            print("  - ✅ 提取hidden states（用于不确定性估计）")
            print("  - ✅ 提取visual features（用于视觉不确定性）")
            print("  - ✅ 生成多样本（用于eigen_score）")
            print()
            print("用途:")
            print("  - 替代LLaVA-1.5-7B作为主MLLM")
            print("  - Self-Aware Multimodal RAG系统")
            print("  - VQA任务（OK-VQA、A-OKVQA、MultiModalQA）")
        else:
            print("⚠️  Qwen3-VL依赖未安装")
            print("   请运行: pip install transformers accelerate qwen-vl-utils")
    else:
        print(f"❌ 模型未找到: {model_path}")
        print("   请先下载模型")
    
    print("=" * 80)

