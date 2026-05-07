#!/bin/bash
# 实验结果分析脚本

LOG_FILE="/root/autodl-tmp/optimized_100samples.log"

echo "================================================"
echo "🎯 Self-Aware-MRAG 优化实验结果分析"
echo "================================================"
echo ""

# 检查实验是否完成
if tail -50 "$LOG_FILE" | grep -q "Self-Aware-MRAG.*100/100"; then
    echo "✅ 实验已完成！"
else
    echo "⏳ 实验进行中..."
    tail -100 "$LOG_FILE" | grep "运行 Self-Aware" | tail -1
    echo ""
fi

echo ""
echo "----------------------------------------"
echo "📊 Self-Aware-MRAG 性能指标"
echo "----------------------------------------"

# 提取Self-Aware-MRAG的结果
if grep -q "Self-Aware-MRAG" "$LOG_FILE"; then
    echo ""
    tail -200 "$LOG_FILE" | grep -A 10 "Self-Aware-MRAG" | grep -E "(EM:|F1:|VQA-Score:|Recall@5:|平均|样本)" | head -10
fi

echo ""
echo "----------------------------------------"
echo "🔍 Adaptive Retrieval 统计"
echo "----------------------------------------"

# 统计检索决策
TOTAL_RETRIEVE=$(grep -c "should_retrieve=True" "$LOG_FILE" || echo "0")
TOTAL_SKIP=$(grep -c "should_retrieve=False" "$LOG_FILE" || echo "0")
TOTAL=$((TOTAL_RETRIEVE + TOTAL_SKIP))

if [ $TOTAL -gt 0 ]; then
    RETRIEVE_PERCENT=$(awk "BEGIN {printf \"%.1f\", $TOTAL_RETRIEVE * 100 / $TOTAL}")
    SKIP_PERCENT=$(awk "BEGIN {printf \"%.1f\", $TOTAL_SKIP * 100 / $TOTAL}")
    
    echo "总样本数: $TOTAL"
    echo "检索样本: $TOTAL_RETRIEVE (${RETRIEVE_PERCENT}%)"
    echo "跳过检索: $TOTAL_SKIP (${SKIP_PERCENT}%)"
fi

# 不确定性统计
echo ""
echo "----------------------------------------"
echo "📈 不确定性分布"
echo "----------------------------------------"

if grep -q "uncertainty=" "$LOG_FILE"; then
    grep "uncertainty=" "$LOG_FILE" | grep -oP 'uncertainty=\K[0-9.]+' | \
        awk '{
            sum+=$1; count++; 
            if($1<0.35) low++;
            else high++;
        } 
        END {
            if(count>0) {
                printf "平均不确定性: %.4f\n", sum/count;
                printf "低于阈值(0.35): %d (%.1f%%)\n", low, low*100/count;
                printf "高于阈值(0.35): %d (%.1f%%)\n", high, high*100/count;
            }
        }'
fi

echo ""
echo "================================================"
echo "💡 与基线版本(EM 62.0%)对比将在实验完成后生成"
echo "================================================"

