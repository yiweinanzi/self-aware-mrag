#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=multimodalvqa_simple
#SBATCH --output=multimodalvqa_simple_%j.out
#SBATCH --error=multimodalvqa_simple_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================" | tee multimodalvqa_simple_${SLURM_JOB_ID}.out
echo "MultiModalQA Self-Aware-MRAG简单测试 - 3样本" | tee -a multimodalvqa_simple_${SLURM_JOB_ID}.out
echo "开始时间: $(date)" | tee -a multimodalvqa_simple_${SLURM_JOB_ID}.out
echo "============================================" | tee -a multimodalvqa_simple_${SLURM_JOB_ID}.out

# 运行MultiModalQA Self-Aware-MRAG测试（只运行3个样本）
echo -e "\n开始运行Self-Aware-MRAG..." | tee -a multimodalvqa_simple_${SLURM_JOB_ID}.out
python -c "
import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

# 简单测试代码
import json
import gzip
from datetime import datetime
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
from flashrag.retriever import DenseRetriever

print('加载数据...')
data_file = '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA/MMQA_dev.jsonl.gz'
samples = []

with gzip.open(data_file, 'rt') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        item = json.loads(line)

        # 获取图像
        metadata = item.get('metadata', {})
        image_path = None
        if 'image_doc_ids' in metadata and metadata['image_doc_ids']:
            image_id = metadata['image_doc_ids'][0]
            image_path = f'/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA/images/final_dataset_images/{image_id}.jpg'
            print(f'样本 {i+1}: 图像路径 = {image_path}')

        answers = item.get('answers', [])
        golden_answers = [ans.get('answer', '') for ans in answers]

        samples.append({
            'id': item.get('qid', f'mmqa_{i}'),
            'question': item.get('question', ''),
            'answer': golden_answers[0] if golden_answers else '',
            'golden_answers': golden_answers,
            'image': image_path,
        })

print(f'加载了 {len(samples)} 个样本')

print('初始化模型...')
qwen3_vl = create_qwen3_vl_wrapper(
    model_path='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    device='cuda',
    torch_dtype='bfloat16'
)

print('初始化检索器...')
retriever_config = {
    'index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    'retrieval_method': 'e5',
    'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
    'retrieval_topk': 5,
}
retriever = DenseRetriever(retriever_config)

print('初始化Self-Aware-MRAG...')
pipeline = SelfAwarePipelineQwen3VL(
    qwen3_vl_wrapper=qwen3_vl,
    retriever=retriever,
    config={
        'uncertainty_threshold': 0.35,
        'use_improved_estimator': True,
        'use_position_fusion': True,
        'use_attribution': True,
        'clip_model_path': '/data0/home/zqwang/ACL/models/clip-vit-large-patch14-336',
        'retrieval_topk': 5,
        'thinking': False,
        'max_images': 10,
    }
)

print('\\n运行测试...')
results = []
for i, sample in enumerate(samples):
    print(f'\\n--- 样本 {i+1} ---')
    print(f'问题: {sample[\"question\"]}')
    print(f'图像: {sample[\"image\"]}')

    try:
        result = pipeline.process(sample)
        uncertainty = result.get('uncertainty', {})
        print(f'不确定性: text={uncertainty.get(\"text\", 0):.3f}, visual={uncertainty.get(\"visual\", 0):.3f}, align={uncertainty.get(\"alignment\", 0):.3f}')
        print(f'检索: {result.get(\"retrieved\", False)}, 文档数: {len(result.get(\"retrieved_docs\", []))}')
        print(f'答案: {result.get(\"answer\", \"\")}')
        results.append(result)
    except Exception as e:
        print(f'错误: {e}')

print('\\n=== 测试完成 ===')
" 2>&1 | tee -a multimodalvqa_simple_${SLURM_JOB_ID}.out

echo -e "\n实验完成时间: $(date)" | tee -a multimodalvqa_simple_${SLURM_JOB_ID}.out