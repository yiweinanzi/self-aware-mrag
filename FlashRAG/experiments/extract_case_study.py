#!/usr/bin/env python3
"""
提取OK-VQA的case study信息
"""
import json
import os
from pathlib import Path

# 读取我们的方法结果
ours_file = '/data0/home/zqwang/ACL/FlashRAG/experiments/results_okvqa_baselines/Self-Aware-MRAG_Self_Aware_MRAG_results.json'
baseline_file = '/data0/home/zqwang/ACL/FlashRAG/experiments/results_okvqa_baselines/ViDoRAG_results.json'

with open(ours_file, 'r') as f:
    ours_data = json.load(f)

with open(baseline_file, 'r') as f:
    baseline_data = json.load(f)

# 读取原始数据集获取图片ID
questions_file = '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA/OpenEnded_mscoco_val2014_questions.json'
with open(questions_file, 'r') as f:
    questions_data = json.load(f)

# 创建image_id到问题ID的映射
imgid_to_qid = {}
if 'questions' in questions_data:
    for item in questions_data['questions'][:6000]:  # val set
        imgid_to_qid[item['image_id']] = item['question_id']

# 读取标注获取答案
annotations_file = '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA/mscoco_val2014_annotations.json'
with open(annotations_file, 'r') as f:
    annotations_data = json.load(f)

# 创建question_id到答案的映射
qid_to_answers = {}
if 'annotations' in annotations_data:
    for ann in annotations_data['annotations']:
        qid_to_answers[ann['question_id']] = ann['answers']

# 读取更多baseline
baselines = {
    'SAM-RAG': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_okvqa_baselines/SAM-RAG_results.json',
    'mR2AG': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_okvqa_baselines/mR2AG_results.json',
    'VisRAG': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_okvqa_baselines/VisRAG_results.json',
    'RagVL': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_okvqa_baselines/RagVL_results.json',
    'MuRAG': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_okvqa_baselines/MuRAG_results.json',
}

baseline_results = {}
for name, path in baselines.items():
    try:
        with open(path, 'r') as f:
            baseline_results[name] = json.load(f)
    except:
        pass

# 输出case study
print("=" * 100)
print("OK-VQA Case Study - 5个案例")
print("=" * 100)

for i in range(min(5, len(ours_data['results']))):
    ours_result = ours_data['results'][i]
    baseline_result = baseline_data['results'][i]

    question = ours_result['question']
    ours_answer = ours_result['answer']
    baseline_answer = baseline_result['answer']
    golden_answers = ours_result['golden_answers']

    # 获取uncertainty信息
    unc = ours_result.get('uncertainty', {})
    total_unc = unc.get('total', 'N/A')
    text_unc = unc.get('text', 'N/A')
    visual_unc = unc.get('visual', 'N/A')
    retrieved = ours_result.get('retrieved', False)

    # 获取检索文档
    retrieved_docs = ours_result.get('retrieved_docs', [])

    # 获取position bias信息
    pos_bias = ours_result.get('position_bias_results', {})
    pos_weights = pos_bias.get('position_weights', [])

    print(f"\n{'=' * 100}")
    print(f"Case {i+1}")
    print(f"{'=' * 100}")

    print(f"\n[Question]")
    print(f"  {question}")

    print(f"\n[Gold Answer]")
    print(f"  {golden_answers[:3]}")  # 显示前3个答案

    print(f"\n[Baseline Output (ViDoRAG)]")
    print(f"  {baseline_answer[:200]}..." if len(baseline_answer) > 200 else f"  {baseline_answer}")

    print(f"\n[Ours Output (Self-Aware-MRAG)]")
    print(f"  {ours_answer}")

    print(f"\n[Top-3 Evidence Snippets]")
    for j, doc in enumerate(retrieved_docs[:3]):
        doc_id = doc.get('id', 'N/A')
        title = doc.get('title', 'N/A')
        content = doc.get('contents', '')[:150]
        # 确定位置
        if j < len(pos_weights):
            weight = pos_weights[j]
            if weight > 0.35:
                pos = "front"
            elif weight < 0.1:
                pos = "back"
            else:
                pos = "mid"
        else:
            pos = "mid"
        print(f"  [{j+1}] DocID: {doc_id} | pos={pos}")
        print(f"      Title: {title}")
        print(f"      Snippet: {content}...")

    print(f"\n[Meta]")
    print(f"  uncertainty_total={total_unc:.4f}" if isinstance(total_unc, float) else f"  uncertainty_total={total_unc}")
    print(f"  uncertainty_text={text_unc:.4f}" if isinstance(text_unc, float) else f"  uncertainty_text={text_unc}")
    print(f"  uncertainty_visual={visual_unc:.4f}" if isinstance(visual_unc, float) else f"  uncertainty_visual={visual_unc}")
    print(f"  retrieved={retrieved}")
    print(f"  pos_aware_on=True" if ours_result.get('position_bias_stats') else "  pos_aware_on=False")

    # 查找图片路径（需要从原始数据集获取）
    # OK-VQA使用COCO图片，image_id格式需要处理
    # 由于我们没有直接的image_id映射，这里只提示

print(f"\n{'=' * 100}")
print("图片位置说明:")
print("  图片位于: /data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA/val2014/")
print("  格式: COCO_val2014_000000{image_id:06d}.jpg")
print("=" * 100)
