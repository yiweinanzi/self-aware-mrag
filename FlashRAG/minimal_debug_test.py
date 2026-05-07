#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最小化调试测试 - 直接复现格式字符串错误
"""

import os
import sys
import json

# 添加项目路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

def main():
    print("=" * 80)
    print("最小化调试测试")
    print("=" * 80)

    # 直接运行原始测试，但只处理一个样本
    print("\n运行单个样本的Self-Aware测试...")

    # 运行测试，但捕获详细错误
    import subprocess
    import tempfile

    # 创建临时的单样本测试脚本
    test_script = '''
import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

try:
    from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
    from flashrag.retriever import DenseRetriever
    from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
    from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple
    import traceback

    print("初始化模型...")
    qwen3_vl = create_qwen3_vl_wrapper(
        model_path='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
        device='cuda'
    )

    print("初始化检索器...")
    retriever_config = {
        'index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
        'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
        'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
        'max_length': 512
    }
    retriever = DenseRetriever(retriever_config)

    print("初始化Pipeline...")
    pipeline = SelfAwarePipelineQwen3VL(
        qwen3_vl_wrapper=qwen3_vl,
        retriever=retriever,
        config={
            'force_retrieval': True,
            'uncertainty_threshold': 0.43,
            'use_improved_estimator': True,
            'use_position_fusion': True,
            'use_attribution': True
        }
    )

    print("加载数据...")
    dataset = OKVQADatasetSimple({
        'data_dir': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
        'split': 'val',
        'load_images': False,
    })

    print("处理第一个样本...")
    sample = dataset[0]

    # 详细调试每个步骤
    print("\\n[1] 调用run_single...")
    try:
        result = pipeline.run_single(sample)
        print(f"✅ run_single成功")
        print(f"答案: {result.get('answer', 'N/A')!r}")
    except Exception as e:
        print(f"❌ run_single失败: {e}")
        print("错误详情:")
        traceback.print_exc()

        # 尝试更详细的调试
        print("\\n[2] 尝试单独生成答案...")
        try:
            question = sample['question']
            prompt = f"Answer with 1-3 words: {question}"
            answer = qwen3_vl.generate(text=prompt, max_new_tokens=5)
            print(f"✅ 直接生成成功: {answer!r}")
        except Exception as e2:
            print(f"❌ 直接生成失败: {e2}")
            traceback.print_exc()

except Exception as e:
    print(f"❌ 初始化失败: {e}")
    traceback.print_exc()
'''

    # 写入临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        temp_file = f.name

    try:
        # 在GPU节点上运行
        cmd = f'''
srun -p 5090 -N 1 -n 1 --gres=gpu:1 -t 00:05:00 bash -c '
eval "$(conda shell.bash hook)"
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG
export CUDA_VISIBLE_DEVICES=0
python {temp_file}
'
'''
        print(f"执行命令: {cmd[:100]}...")

        # 捕获输出
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        print("\n" + "=" * 80)
        print("输出结果:")
        print("=" * 80)
        print(result.stdout)

        if result.stderr:
            print("\n" + "=" * 80)
            print("错误信息:")
            print("=" * 80)
            print(result.stderr)

    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    main()