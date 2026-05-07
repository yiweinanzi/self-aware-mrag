#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=test_improved_mr
#SBATCH --output=test_improved_mr_%j.out
#SBATCH --error=test_improved_mr_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "测试改进后的Self-Aware-MRAG - 3样本"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================"

# 创建输出目录
mkdir -p results_improved_mr_test

# 只运行Self-Aware-MRAG方法进行测试
python -c "
import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.retriever import DenseRetriever
import json
import gzip
from datetime import datetime

# 初始化模型
print('正在加载Qwen3-VL...')
qwen3_vl = create_qwen3_vl_wrapper(
    model_path='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    device='cuda',
    torch_dtype='bfloat16',
    max_new_tokens=50
)

# 初始化检索器
print('正在初始化检索器...')
config = {
    'index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/bge_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/corpus.jsonl',
    'retrieval_method': 'e5',
    'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
    'retrieval_query_max_length': 512,
    'retrieval_pooling_method': 'mean',
    'retrieval_use_fp16': True,
    'retrieval_batch_size': 128,
    'retrieval_topk': 5,
    'save_retrieval_cache': False,
    'use_retrieval_cache': False,
    'faiss_gpu': False
}

retriever = DenseRetriever(config)

# 初始化Self-Aware-MRAG
print('正在初始化Self-Aware-MRAG...')
pipeline = SelfAwarePipelineQwen3VL(
    qwen3_vl_wrapper=qwen3_vl,
    retriever=retriever,
    config={
        'uncertainty_threshold': 0.35,
        'use_improved_estimator': True,
        'use_position_fusion': True,
        'use_attribution': True,
        'use_dataset_docs': True
    }
)

# 加载测试样本
print('\\n加载测试样本...')
test_samples = []
with gzip.open('/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA/MMQA_dev.jsonl.gz', 'rt') as f:
    for i, line in enumerate(f):
        if i >= 3:  # 只测试3个样本
            break
        item = json.loads(line)
        test_samples.append(item)

print(f'加载了 {len(test_samples)} 个测试样本\\n')

# 测试每个样本
results = []
for i, sample in enumerate(test_samples):
    print(f'\\n========== 样本 {i+1}/3 ==========')
    print(f'问题: {sample[\"question\"]}')
    print(f'答案: {sample[\"answer\"]}')

    # 运行pipeline
    result = pipeline.process_sample(sample)

    print(f'预测答案: {result[\"answer\"]}')
    print(f'是否正确: {result[\"answer\"].lower() == sample[\"answer\"].lower()}')

    results.append({
        'question': sample['question'],
        'ground_truth': sample['answer'],
        'prediction': result['answer'],
        'correct': result['answer'].lower() == sample['answer'].lower()
    })

    print('-' * 50)

# 保存结果
output_path = 'results_improved_mr_test/test_results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print('\\n========== 测试结果 ==========')
correct_count = sum(1 for r in results if r['correct'])
print(f'正确率: {correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%)')

for i, r in enumerate(results, 1):
    status = '✅' if r['correct'] else '❌'
    print(f'{status} 样本{i}: Q={r[\"question\"][:50]}... | GT={r[\"ground_truth\"]} | Pred={r[\"prediction\"]}')

print(f'\\n结果已保存到: {output_path}')
print('\\n测试完成时间:', datetime.now())
"