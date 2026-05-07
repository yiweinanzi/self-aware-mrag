#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG

echo "=== 单GPU 100样本测试 ==="
echo "验证准确率提升效果：从1.3%到50%+"

python -c "
import sys
import json
import time
import torch
sys.path.insert(0, '.')

try:
    from flashrag.modules.qwen3_vl import Qwen3VLProcessor
    from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple

    print('🔄 加载Qwen3-VL模型...')
    model = Qwen3VLProcessor(
        model_path='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
        device='cuda:0',
        torch_dtype=torch.bfloat16
    )
    print('✅ Qwen3-VL模型加载成功')

    print('🔄 加载100个测试样本...')
    dataset_obj = OKVQADatasetSimple({
        'data_dir': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
        'split': 'val',
        'load_images': True,
    })

    test_samples = dataset_obj.data[:100]
    print(f'✅ 加载测试样本: {len(test_samples)}')

    correct_count = 0
    results = []

    print('🔄 开始100样本测试...')
    for i, sample in enumerate(test_samples):
        question = sample['question']
        golden_answers = sample['golden_answers']

        if (i + 1) % 20 == 0 or i == 0:
            print(f'[{i+1}/100] {question[:60]}...')
        else:
            print(f'[{i+1}/100] .', end='', flush=True)

        try:
            answer = model.generate(question, sample.get('image'))

            # 评估答案
            is_correct = evaluate_answer(answer, golden_answers)
            if is_correct:
                correct_count += 1

            results.append({
                'sample_id': i + 1,
                'question': question,
                'predicted_answer': answer,
                'golden_answers': golden_answers,
                'is_correct': is_correct
            })

        except Exception as e:
            print(f'\\n   ❌ 样本{i+1}推理失败: {e}')
            results.append({
                'sample_id': i + 1,
                'question': question,
                'predicted_answer': '',
                'golden_answers': golden_answers,
                'is_correct': False,
                'error': str(e)
            })

    # 计算准确率
    accuracy = correct_count / len(test_samples)
    failed_count = len([r for r in results if r.get('error')])

    print(f'\\n📊 测试结果:')
    print(f'   准确率: {accuracy:.3f} ({correct_count}/{len(test_samples)})')
    print(f'   失败率: {failed_count/len(test_samples)*100:.1f}% ({failed_count}/{len(test_samples)})')

    # 保存结果
    output_file = '/data0/home/zqwang/ACL/test_100samples_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'test_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'samples_count': len(test_samples),
                'model': 'Qwen3-VL-8B-Instruct',
                'device': 'cuda:0'
            },
            'results': {
                'accuracy': accuracy,
                'correct_count': correct_count,
                'failed_count': failed_count,
                'samples': results
            }
        }, f, ensure_ascii=False, indent=2)

    print(f'✅ 结果已保存: {output_file}')

    if accuracy > 0.1:  # 如果准确率超过10%
        print(f'🎉 测试成功！准确率为{accuracy:.1f}%，修复效果明显')
    else:
        print(f'⚠️ 准确率仍然较低({accuracy:.3f})，需要进一步调试')

except Exception as e:
    print(f'❌ 测试失败: {e}')
    import traceback
    traceback.print_exc()

def evaluate_answer(predicted, golden):
    '''评估答案'''
    if isinstance(golden, str):
        golden = [golden]
    elif not isinstance(golden, list):
        golden = list(golden) if golden else []

    predicted = str(predicted).strip().lower()

    # 精确匹配
    for gold in golden:
        if predicted == gold.strip().lower():
            return True

    # 包含匹配
    for gold in golden:
        if gold.strip().lower() in predicted or predicted in gold.strip().lower():
            return True

    return False
"