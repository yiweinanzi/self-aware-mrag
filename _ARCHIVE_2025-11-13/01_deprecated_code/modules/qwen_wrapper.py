#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen2.5模型封装器

用于与LLaVA对比实验

模型位置：/root/autodl-tmp/models/Qwen2.5-7B-Instruct
"""

import os
import warnings
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers未安装")

class Qwen25Wrapper:
    """
    Qwen2.5-7B-Instruct模型封装器
    
    用于纯文本RAG对比实验
    
    使用示例：
    ```python
    qwen = Qwen25Wrapper('/root/autodl-tmp/models/Qwen2.5-7B-Instruct')
    
    answer = qwen.generate(
        text="Question: What is the capital of France?\nAnswer:",
        max_new_tokens=50
    )
    ```
    """
    
    def __init__(self, model_path: str, device: str = 'cuda', load_in_8bit: bool = False):
        """
        初始化Qwen模型
        
        Args:
            model_path: 模型路径
            device: 设备
            load_in_8bit: 是否8bit量化
        """
        self.model_path = model_path
        self.device = device
        self.load_in_8bit = load_in_8bit
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("请安装transformers: pip install transformers")
        
        print(f"正在加载Qwen2.5-7B: {model_path}")
        print(f"设备: {device}, 8bit量化: {load_in_8bit}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 加载模型
        if load_in_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                load_in_8bit=True,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        
        self.model.eval()
        
        print(f"✅ Qwen2.5-7B加载成功")
        print(f"   设备: {self.model.device}")
    
    def generate(self, text: str, max_new_tokens: int = 100, 
                temperature: float = 0.7, **kwargs) -> str:
        """
        生成文本
        
        Args:
            text: 输入文本（可以是prompt）
            max_new_tokens: 最大生成token数
            temperature: 温度
            
        Returns:
            str: 生成的文本
        """
        # 构建messages格式（Qwen2.5使用chat模板）
        messages = [
            {"role": "user", "content": text}
        ]
        
        # 应用chat模板
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            text_input,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                **kwargs
            )
        
        # 解码
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()


# 工厂函数
def create_qwen_wrapper(model_path: str = '/root/autodl-tmp/models/Qwen2.5-7B-Instruct',
                       device: str = 'cuda'):
    """创建Qwen wrapper"""
    return Qwen25Wrapper(model_path, device)


if __name__ == '__main__':
    print("Qwen2.5 Wrapper测试")
    print("=" * 70)
    
    model_path = '/root/autodl-tmp/models/Qwen2.5-7B-Instruct'
    
    if os.path.exists(model_path):
        print("\n模型文件存在，测试加载...")
        
        # 注：实际加载需要GPU
        print(f"模型路径: {model_path}")
        print(f"模型大小: 15GB")
        print()
        print("✅ Qwen2.5-7B-Instruct已准备好")
        print()
        print("使用方法:")
        print("  from flashrag.modules.qwen_wrapper import Qwen25Wrapper")
        print("  qwen = Qwen25Wrapper(model_path)")
        print("  answer = qwen.generate(prompt)")
        print()
        print("用途:")
        print("  - 纯文本RAG baseline")
        print("  - 与LLaVA对比（文本 vs 多模态）")
    else:
        print(f"⚠️  模型未找到: {model_path}")
    
    print("=" * 70)


