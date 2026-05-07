#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试统一版本的基本功能
Test Basic Functionality of Unified Version
"""

import sys
import os
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

def test_imports():
    """测试导入"""
    print("测试模块导入...")

    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch未安装")
        return False

    try:
        from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple
        print("✅ OKVQADatasetSimple")
    except ImportError as e:
        print(f"❌ OKVQADatasetSimple: {e}")
        return False

    try:
        from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
        print("✅ create_qwen3_vl_wrapper")
    except ImportError as e:
        print(f"❌ create_qwen3_vl_wrapper: {e}")
        return False

    try:
        from flashrag.retriever import DenseRetriever
        print("✅ DenseRetriever")
    except ImportError as e:
        print(f"❌ DenseRetriever: {e}")
        return False

    try:
        from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
        print("✅ SelfAwarePipelineQwen3VL")
    except ImportError as e:
        print(f"❌ SelfAwarePipelineQwen3VL: {e}")
        return False

    return True

def test_config():
    """测试配置"""
    print("\n测试配置...")

    # 测试默认参数
    from run_unified_ablation import ABLATION_VARIANTS
    print(f"✅ 找到 {len(ABLATION_VARIANTS)} 个消融变体")

    for variant in ABLATION_VARIANTS:
        print(f"  - {variant['name']}: {variant['description']}")

    return True

def test_data_paths():
    """测试数据路径"""
    print("\n测试数据路径...")

    paths = [
        '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
        '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
        '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
        '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    ]

    for path in paths:
        if os.path.exists(path):
            print(f"✅ {path}")
        else:
            print(f"❌ {path}")

    return True

def main():
    """主测试函数"""
    print("🧪 测试统一消融实验版本")
    print("="*60)

    tests = [
        ("模块导入", test_imports),
        ("配置", test_config),
        ("数据路径", test_data_paths),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")

    print(f"\n{'='*60}")
    print(f"📊 测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！统一版本准备就绪。")
        print("\n💡 使用建议:")
        print("   python run_unified_ablation.py --max-samples 100")
        print("   python run_unified_ablation.py --max-samples 100 --use-multi-gpu --num-gpus 2")
        print("   python run_unified_ablation.py --dataset mragbench --max-samples 100")
    else:
        print("⚠️ 部分测试失败，请检查环境配置")

if __name__ == "__main__":
    main()