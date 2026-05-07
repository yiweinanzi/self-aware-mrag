#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import traceback

# 添加项目路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

def test_method_import():
    print("测试方法导入...")

    # 测试MuRAG
    try:
        from experiments.baselines.murag_enhanced import MuRAGEnhanced
        print("✅ MuRAGEnhanced 导入成功")
    except Exception as e:
        print(f"❌ MuRAGEnhanced 导入失败: {e}")
        traceback.print_exc()

    # 测试ViDoRAG
    try:
        from experiments.baselines.vidorag_pipeline import ViDoRAGPipeline
        print("✅ ViDoRAGPipeline 导入成功")
    except Exception as e:
        print(f"❌ ViDoRAGPipeline 导入失败: {e}")
        traceback.print_exc()

def test_methods_with_dummy():
    print("\n测试方法初始化...")

    # 加载模型
    try:
        from flashrag.pipeline.dynamic_builder import DynamicBuilder
        builder = DynamicBuilder()

        config = {
            'model_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
            'device': 'cuda',
            'torch_dtype': 'bfloat16',
            'temperature': 0.01,
            'max_new_tokens': 30,
            'load_images': False
        }

        qwen3_vl = builder.get_model("qwen3_vl", config)
        print("✅ 模型加载成功")

        # 创建虚拟检索器
        class DummyRetriever:
            def search(self, query, num=5):
                return ["dummy doc"], []

        retriever = DummyRetriever()

        # 测试MuRAG
        try:
            from experiments.baselines.murag_enhanced import MuRAGEnhanced
            murag = MuRAGEnhanced(qwen3_vl, retriever, config)
            print("✅ MuRAG 初始化成功")
        except Exception as e:
            print(f"❌ MuRAG 初始化失败: {e}")
            traceback.print_exc()

        # 测试ViDoRAG
        try:
            from experiments.baselines.vidorag_pipeline import ViDoRAGPipeline
            vidorag = ViDoRAGPipeline(qwen3_vl, retriever, config)
            print("✅ ViDoRAG 初始化成功")
        except Exception as e:
            print(f"❌ ViDoRAG 初始化失败: {e}")
            traceback.print_exc()

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_method_import()
    test_methods_with_dummy()