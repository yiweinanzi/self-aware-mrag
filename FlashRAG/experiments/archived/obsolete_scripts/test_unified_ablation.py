#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一消融实验测试脚本
Test Script for Unified Ablation Study

快速验证统一版本的功能是否正常
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试统一消融实验基本功能...")
    print("="*60)

    # 测试命令行帮助
    print("1. 测试命令行参数...")
    try:
        result = subprocess.run([
            sys.executable, 'run_unified_ablation.py', '--help'
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("✅ 命令行参数正常")
        else:
            print("❌ 命令行参数错误:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ 命令行测试失败: {e}")
        return False

    # 测试导入
    print("2. 测试模块导入...")
    try:
        # 添加路径
        sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

        # 测试核心导入
        from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple
        from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
        from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
        from flashrag.retriever import DenseRetriever

        print("✅ 核心模块导入成功")
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False

    return True

def test_quick_run():
    """快速运行测试（极小样本）"""
    print("\n🧪 测试快速运行（5个样本）...")
    print("="*60)

    cmd = [
        sys.executable, 'run_unified_ablation.py',
        '--max-samples', '5',
        '--variants', 'Baseline_MuRAG',  # 只测试baseline
        '--output-dir', './test_results'
    ]

    try:
        print(f"执行命令: {' '.join(cmd)}")
        start_time = time.time()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5分钟超时
            cwd='/data0/home/zqwang/ACL/FlashRAG/experiments'
        )

        end_time = time.time()
        execution_time = end_time - start_time

        if result.returncode == 0:
            print("✅ 快速运行成功")
            print(f"   执行时间: {execution_time:.1f}秒")
            print("   输出预览:")
            print(result.stdout[-500:])  # 显示最后500字符
            return True
        else:
            print("❌ 快速运行失败")
            print(f"   返回码: {result.returncode}")
            print("   错误输出:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("❌ 快速运行超时（5分钟）")
        return False
    except Exception as e:
        print(f"❌ 快速运行异常: {e}")
        return False

def test_gpu_availability():
    """测试GPU可用性"""
    print("\n🧪 测试GPU可用性...")
    print("="*60)

    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✅ GPU可用: {device_count} 个")

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"  GPU {i}: {props.name}, {memory_gb:.1f}GB")

            return True
        else:
            print("⚠️ GPU不可用，将使用CPU模式")
            return True

    except Exception as e:
        print(f"❌ GPU检测失败: {e}")
        return False

def test_file_paths():
    """测试关键文件路径"""
    print("\n🧪 测试关键文件路径...")
    print("="*60)

    required_paths = [
        '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
        '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
        '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
        '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    ]

    all_good = True
    for path in required_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                print(f"✅ 目录存在: {path}")
            else:
                print(f"✅ 文件存在: {path}")
        else:
            print(f"❌ 路径不存在: {path}")
            all_good = False

    return all_good

def main():
    """主测试函数"""
    print("🧪 统一消融实验测试套件")
    print("="*80)
    print("测试目标: 验证run_unified_ablation.py的功能完整性")
    print("="*80)

    # 切换到实验目录
    os.chdir('/data0/home/zqwang/ACL/FlashRAG/experiments')

    tests = [
        ("GPU可用性", test_gpu_availability),
        ("文件路径", test_file_paths),
        ("基本功能", test_basic_functionality),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ 测试 '{test_name}' 异常: {e}")
            results.append((test_name, False))

    # 如果基本测试通过，进行快速运行测试
    if all(result[1] for result in results):
        print("\n" + "="*80)
        print("基本测试通过，进行快速运行测试...")
        print("="*80)

        try:
            quick_result = test_quick_run()
            results.append(("快速运行", quick_result))
        except Exception as e:
            print(f"❌ 快速运行测试异常: {e}")
            results.append(("快速运行", False))
    else:
        results.append(("快速运行", False))  # 基本测试失败，跳过快速运行

    # 生成测试报告
    print("\n" + "="*80)
    print("📊 测试报告")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<20} {status}")

    print(f"\n总体结果: {passed}/{total} 测试通过")

    if passed == total:
        print("🎉 所有测试通过！统一消融实验准备就绪。")
        print("\n📝 下一步:")
        print("1. 运行完整实验: python run_unified_ablation.py")
        print("2. 指定样本数: python run_unified_ablation.py --max-samples 100")
        print("3. 使用多GPU: python run_unified_ablation.py --use-multi-gpu --num-gpus 2")
    else:
        print("⚠️ 部分测试失败，请检查配置后再运行实验。")

if __name__ == "__main__":
    main()