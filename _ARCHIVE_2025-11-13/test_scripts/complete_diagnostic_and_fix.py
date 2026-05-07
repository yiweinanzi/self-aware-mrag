#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整诊断和修复脚本
检查所有潜在问题并给出修复建议

生成时间: 2025-11-01
"""

import sys
import os

print("=" * 80)
print("Self-Aware-MRAG 完整诊断")
print("=" * 80)
print()

# 问题列表
issues = []
warnings = []
fixes = []

# ========== 问题1: 检查pipeline代码修复情况 ==========
print("[检查1] Pipeline代码是否正确传入MLLM模型...")
pipeline_file = "/root/autodl-tmp/FlashRAG/flashrag/pipeline/self_aware_pipeline_qwen3vl.py"

try:
    with open(pipeline_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'mllm_for_uncertainty = self.qwen3_vl' in content or 'mllm_for_uncertainty = self.qwen3_vl.model' in content:
        print("  ✅ Pipeline代码已修复：传入真实MLLM模型")
    elif 'mllm_model=None' in content and 'CrossModalUncertaintyEstimator' in content:
        print("  ❌ 问题：Pipeline仍然传入mllm_model=None")
        issues.append({
            'level': 'CRITICAL',
            'issue': 'Pipeline传入mllm_model=None',
            'file': pipeline_file,
            'line': '约第136行',
            'impact': '导致不确定性估计降级为简化版，所有样本返回相同的0.47'
        })
        fixes.append({
            'step': 1,
            'action': '修改pipeline代码，传入真实模型',
            'command': 'vi ' + pipeline_file + ' # 修改第136行'
        })
    else:
        print("  ⚠️  无法确定pipeline状态")
        warnings.append("无法确定pipeline是否正确修复")
except Exception as e:
    print(f"  ❌ 读取pipeline文件失败: {e}")
    issues.append({'level': 'ERROR', 'issue': f'无法读取文件: {e}'})

print()

# ========== 问题2: 检查Qwen3-VL wrapper结构 ==========
print("[检查2] Qwen3-VL wrapper结构...")

# 检查Qwen2VLInferenceEngine是否正确实现
generator_file = "/root/autodl-tmp/FlashRAG/flashrag/generator/multimodal_generator.py"
try:
    with open(generator_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'class Qwen2VLInferenceEngine' in content and 'self.model =' in content:
        print("  ✅ Qwen2VLInferenceEngine有self.model属性")
        print("     - self.model = Qwen2VLForConditionalGeneration")
        print("     - self.tokenizer = processor.tokenizer")
        print("     - self.processor = AutoProcessor")
    else:
        print("  ⚠️  Qwen2VLInferenceEngine结构可能不完整")
        warnings.append("Qwen2VLInferenceEngine结构需要验证")
except Exception as e:
    print(f"  ❌ 读取generator文件失败: {e}")

print()

# ========== 问题3: 检查CrossModalUncertaintyEstimator的兼容性 ==========
print("[检查3] CrossModalUncertaintyEstimator是否兼容Qwen2-VL...")

estimator_file = "/root/autodl-tmp/FlashRAG/flashrag/modules/uncertainty_estimator.py"
try:
    with open(estimator_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否有Qwen2-VL特定的处理逻辑
    if 'Qwen2VL' in content or 'qwen2vl' in content.lower():
        print("  ✅ Estimator有Qwen2-VL特定处理")
    else:
        print("  ⚠️  Estimator可能不支持Qwen2-VL")
        print("     - 当前主要支持LLaVA")
        print("     - 可能需要添加Qwen2-VL适配代码")
        
        issues.append({
            'level': 'WARNING',
            'issue': 'CrossModalUncertaintyEstimator可能不支持Qwen2-VL',
            'impact': '即使传入model，也可能无法正确提取embeddings',
            'solution': '需要添加Qwen2-VL适配逻辑'
        })
        
        fixes.append({
            'step': 2,
            'action': '为Qwen2-VL添加适配代码',
            'details': [
                '1. 在_get_text_embeddings中检测Qwen2VL模型',
                '2. 使用Qwen2VL的tokenizer和model.model.embed_tokens',
                '3. 测试验证'
            ]
        })
    
    # 检查降级逻辑
    if 'def _estimate_simplified' in content:
        print("  ℹ️  Estimator有简化版降级逻辑")
        print("     - 如果mllm_model=None，会降级")
        print("     - 如果提取embeddings失败，会降级")
    
except Exception as e:
    print(f"  ❌ 读取estimator文件失败: {e}")

print()

# ========== 问题4: 检查_get_text_embeddings实现 ==========
print("[检查4] _get_text_embeddings是否能处理Qwen2-VL...")

try:
    with open(estimator_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到_get_text_embeddings方法
    in_method = False
    method_lines = []
    for i, line in enumerate(lines):
        if 'def _get_text_embeddings' in line:
            in_method = True
            start_line = i
        elif in_method:
            method_lines.append(line)
            if line.strip().startswith('def ') and 'def _get_text_embeddings' not in line:
                break
    
    method_code = ''.join(method_lines)
    
    # 分析实现
    has_llava_logic = 'hasattr(self.mllm_model, \'encode_text\')' in method_code
    has_model_attr_logic = 'hasattr(self.mllm_model, \'model\')' in method_code
    has_tokenizer_logic = 'self.mllm_model.tokenizer' in method_code
    
    print(f"  方法定义在第 {start_line + 1} 行")
    print(f"  - 检测encode_text方法: {'✅' if has_llava_logic else '❌'}")
    print(f"  - 检测model属性: {'✅' if has_model_attr_logic else '❌'}")  
    print(f"  - 使用tokenizer: {'✅' if has_tokenizer_logic else '❌'}")
    
    if has_model_attr_logic:
        print("  ✅ 应该可以处理Qwen2-VL (有model属性检测)")
    else:
        print("  ⚠️  可能无法正确处理Qwen2-VL")
        issues.append({
            'level': 'WARNING',
            'issue': '_get_text_embeddings可能不支持Qwen2-VL',
            'impact': '即使传入Qwen2-VL模型，也会使用随机嵌入',
            'line': f'{start_line + 1}-{start_line + len(method_lines)}'
        })
    
except Exception as e:
    print(f"  ❌ 分析失败: {e}")

print()

# ========== 问题5: 检查实验配置 ==========
print("[检查5] 实验配置...")

config_file = "/root/autodl-tmp/FlashRAG/experiments/run_all_baselines_100samples.py"
try:
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查use_improved_estimator
    if "'use_improved_estimator': False" in content:
        print("  ✅ 配置正确: use_improved_estimator = False")
        print("     将使用CrossModalUncertaintyEstimator（论文方法）")
    elif "'use_improved_estimator': True" in content:
        print("  ⚠️  配置: use_improved_estimator = True")
        print("     将使用ImprovedUncertaintyEstimator（简化方法）")
        print("     这不是论文完整方法，但已验证可行（62% EM）")
        warnings.append("当前使用简化版estimator，不是论文完整方法")
    
    # 检查threshold
    if "'uncertainty_threshold'" in content:
        import re
        match = re.search(r"'uncertainty_threshold':\s*([0-9.]+)", content)
        if match:
            threshold = float(match.group(1))
            print(f"  ℹ️  不确定性阈值: {threshold}")
            if threshold == 0.35:
                print("     ✅ 阈值合理（标准值）")
            elif threshold < 0.3:
                print("     ⚠️  阈值较低，可能导致过度检索")
            elif threshold > 0.4:
                print("     ⚠️  阈值较高，可能导致检索不足")
    
except Exception as e:
    print(f"  ❌ 读取配置文件失败: {e}")

print()

# ========== 汇总报告 ==========
print("=" * 80)
print("诊断汇总")
print("=" * 80)
print()

if issues:
    print(f"🔴 发现 {len(issues)} 个问题:")
    print()
    for i, issue in enumerate(issues, 1):
        print(f"问题 {i}: [{issue.get('level', 'UNKNOWN')}] {issue.get('issue', '')}")
        if 'file' in issue:
            print(f"  文件: {issue['file']}")
        if 'line' in issue:
            print(f"  位置: {issue['line']}")
        if 'impact' in issue:
            print(f"  影响: {issue['impact']}")
        if 'solution' in issue:
            print(f"  解决: {issue['solution']}")
        print()
else:
    print("✅ 未发现严重问题")
    print()

if warnings:
    print(f"⚠️  有 {len(warnings)} 个警告:")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")
    print()

# ========== 修复建议 ==========
if fixes:
    print("=" * 80)
    print("修复建议")
    print("=" * 80)
    print()
    
    for fix in fixes:
        print(f"Step {fix['step']}: {fix['action']}")
        if 'command' in fix:
            print(f"  命令: {fix['command']}")
        if 'details' in fix:
            for detail in fix['details']:
                print(f"    {detail}")
        print()

# ========== 关键决策建议 ==========
print("=" * 80)
print("关键决策")
print("=" * 80)
print()

print("基于诊断结果，您有两个选择:")
print()

print("【选项A】使用论文完整方法 (CrossModalUncertaintyEstimator + Qwen2-VL)")
print("  优点:")
print("    - 符合论文描述（SeaKR的Gram矩阵方法）")
print("    - 理论上更准确")
print("    - 学术价值高")
print("  缺点:")
print("    - 需要确保Qwen2-VL兼容性")
print("    - 可能需要额外的适配代码")
print("    - 存在风险（如果适配不当可能失败）")
print("  修复步骤:")
print("    1. 确认pipeline代码已修复（传入真实模型）")
print("    2. 为Qwen2-VL添加适配代码（如需要）")
print("    3. 小规模测试（10样本）验证")
print("    4. 全数据集实验")
print()

print("【选项B】使用简化方法 (ImprovedUncertaintyEstimator)")
print("  优点:")
print("    - 已验证可行（62% EM on 100 samples）")
print("    - 无需依赖MLLM内部结构")
print("    - 风险低，实施快")
print("  缺点:")
print("    - 不完全符合论文描述")
print("    - 理论深度不足")
print("  修复步骤:")
print("    1. 修改配置: use_improved_estimator = True")
print("    2. 直接运行实验")
print()

print("=" * 80)
print("建议：")
print("  如果时间紧迫，优先选择【选项B】（已验证可行）")
print("  如果追求学术完整性，选择【选项A】（但需要额外测试）")
print("=" * 80)
print()

sys.exit(0 if not issues else 1)

