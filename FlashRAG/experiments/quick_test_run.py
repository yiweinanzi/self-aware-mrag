#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试运行脚本
Quick Test Run

在使用全数据集之前，先用小样本测试所有功能
"""

import sys
import os
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

def quick_test_evaluation():
    """快速测试评估指标"""
    print("="*80)
    print("1. 快速测试评估指标")
    print("="*80)

    try:
        from experiments.test_evaluation_metrics import test_metrics, test_vqa_specific
        success1 = test_metrics()
        success2 = test_vqa_specific()

        if success1 and success2:
            print("✅ 评估指标测试通过")
            return True
        else:
            print("❌ 评估指标测试失败")
            return False
    except Exception as e:
        print(f"❌ 评估指标测试异常: {e}")
        return False

def quick_test_data_loading():
    """快速测试数据加载"""
    print("\n" + "="*80)
    print("2. 快速测试数据加载")
    print("="*80)

    try:
        from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple

        # 测试加载前50个样本
        dataset = OKVQADatasetSimple({
            'data_dir': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
            'split': 'val',
            'load_images': False,  # 暂时不加载图像以加快速度
        })

        # 限制样本数
        dataset.data = dataset.data[:50]

        print(f"✅ 数据加载测试通过: {len(dataset.data)} 样本")

        # 检查数据格式
        if dataset.data:
            sample = dataset.data[0]
            print(f"   样本格式: {list(sample.keys())}")
            print(f"   问题示例: {sample['question'][:50]}...")
            print(f"   答案示例: {sample['golden_answers'][:3]}")

        return True

    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_test_model_loading():
    """快速测试模型加载"""
    print("\n" + "="*80)
    print("3. 快速测试模型加载")
    print("="*80)

    try:
        # 检查模型路径
        model_path = '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct'

        if not os.path.exists(model_path):
            print(f"⚠️ 模型路径不存在: {model_path}")
            print("   请确认模型路径或下载模型")
            return False

        print(f"✅ 模型路径存在: {model_path}")
        print("   模型将在实际运行时加载")

        return True

    except Exception as e:
        print(f"❌ 模型路径检查失败: {e}")
        return False

def check_paths():
    """检查所有必要路径"""
    print("\n" + "="*80)
    print("4. 检查文件路径")
    print("="*80)

    paths_to_check = [
        ('数据目录', '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA'),
        ('OK-VQA问题文件', '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA/OpenEnded_mscoco_val2014_questions.json'),
        ('OK-VQA标注文件', '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA/mscoco_val2014_annotations.json'),
        ('输出目录', '/data0/home/zqwang/ACL/FlashRAG/experiments/results_ablation_okvqa'),
    ]

    all_exist = True
    for name, path in paths_to_check:
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} (不存在)")
            all_exist = False

    return all_exist

def generate_test_config():
    """生成测试用的配置文件"""
    print("\n" + "="*80)
    print("5. 生成测试配置")
    print("="*80)

    test_config = {
        'dataset_name': 'okvqa_test',
        'data_dir': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
        'split': 'val',
        'max_samples': 50,  # 测试用50样本
        'load_images': True,
        'qwen3_vl_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
        'retrieval_topk': 3,  # 减少检索数量以加快速度
        'temperature': 0.01,
        'max_new_tokens': 10,
        'uncertainty_threshold': 0.43,
    }

    # 保存测试配置
    import json
    config_file = '/data0/home/zqwang/ACL/FlashRAG/experiments/test_config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(test_config, f, indent=2, ensure_ascii=False)

    print(f"✅ 测试配置已生成: {config_file}")
    print("   可以使用此配置运行快速测试")

def main():
    """主函数"""
    print("="*80)
    print("快速测试运行")
    print("在使用全数据集运行消融实验之前进行验证")
    print("="*80)
    print()

    # 运行所有测试
    tests = [
        ("评估指标", quick_test_evaluation),
        ("数据加载", quick_test_data_loading),
        ("模型路径", quick_test_model_loading),
        ("文件路径", check_paths),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))

    # 生成测试配置
    try:
        generate_test_config()
        results.append(("测试配置", True))
    except Exception as e:
        print(f"❌ 生成测试配置失败: {e}")
        results.append(("测试配置", False))

    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)

    all_passed = all(result for _, result in results)

    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:12s}: {status}")

    if all_passed:
        print("\n🎉 所有测试通过！")
        print("🚀 现在可以运行完整的消融实验:")
        print("   python /data0/home/zqwang/ACL/FlashRAG/experiments/run_ablation_study_okvqa.py")
    else:
        print("\n⚠️ 部分测试失败，请检查问题后再运行完整实验")
        print("💡 建议:")
        print("   1. 检查文件路径是否正确")
        print("   2. 下载必要的模型和数据")
        print("   3. 确认依赖包已安装")

    print("="*80)

if __name__ == '__main__':
    main()