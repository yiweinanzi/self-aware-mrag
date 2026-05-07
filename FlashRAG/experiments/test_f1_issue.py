#!/usr/bin/env python3
"""Test F1 calculation for multiple choice questions"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')
from flashrag.evaluator.metrics import F1_Score

# Test data: multiple choice
prediction = "Yorkshire_terrier"
golden_answers = ["silky_terrier"]  # This is what we currently pass (wrong)

# What we should pass for multiple choice:
# Option 1: Pass the choice text directly
golden_answers_mc = ["silky_terrier"]  # Same as prediction text format

# Create mock data
class MockData:
    def __init__(self, pred, golden_answers, choices=None):
        self.pred = [pred]
        self.golden_answers = [golden_answers]
        self.choices = choices if choices else [[]]  # Empty list means not multiple choice

# Create F1 calculator
f1_calc = F1_Score({"dataset_name": "test"})

# Test 1: Current approach (treating as open-ended)
data1 = MockData(prediction, golden_answers)
score1, _ = f1_calc.calculate_metric(data1)
print(f"Test 1 - Current approach: F1 = {score1['f1']:.4f}")

# Test 2: Proper multiple choice (should still have low F1 since tokens are different)
data2 = MockData(prediction, golden_answers_mc)
score2, _ = f1_calc.calculate_metric(data2)
print(f"Test 2 - Same format: F1 = {score2['f1']:.4f}")

# Test 3: What if prediction was correct
data3 = MockData("silky_terrier", golden_answers_mc)
score3, _ = f1_calc.calculate_metric(data3)
print(f"Test 3 - Correct answer: F1 = {score3['f1']:.4f}")

# Test 4: What if tokens overlap
data4 = MockData("terrier_silky", golden_answers_mc)
score4, _ = f1_calc.calculate_metric(data4)
print(f"Test 4 - Partial overlap: F1 = {score4['f1']:.4f}")

# Test 5: Test with actual overlap
data5 = MockData("silk_terrier", golden_answers_mc)
score5, _ = f1_calc.calculate_metric(data5)
print(f"Test 5 - Better overlap: F1 = {score5['f1']:.4f}")

# Test 6: Test with common tokens
data6 = MockData("terrier", golden_answers_mc)
score6, _ = f1_calc.calculate_metric(data6)
print(f"Test 6 - Common token: F1 = {score6['f1']:.4f}")

# Test 7: Test with spaced format (should work)
data7 = MockData("silk terrier", golden_answers_mc)
score7, _ = f1_calc.calculate_metric(data7)
print(f"Test 7 - Spaced format: F1 = {score7['f1']:.4f}")

# Analyze tokens
from flashrag.evaluator.utils import normalize_answer

p_tokens = normalize_answer(prediction).split()
gt_tokens = normalize_answer(golden_answers_mc[0]).split()

print(f"\nToken analysis:")
print(f"Prediction tokens: {p_tokens}")
print(f"Ground truth tokens: {gt_tokens}")
print(f"Common tokens: {set(p_tokens) & set(gt_tokens)}")

# Additional analysis
p5_tokens = normalize_answer("silk_terrier").split()
p6_tokens = normalize_answer("terrier").split()
p7_tokens = normalize_answer("silk terrier").split()
print(f"\nTest 5 tokens: {p5_tokens}, Common: {set(p5_tokens) & set(gt_tokens)}")
print(f"Test 6 tokens: {p6_tokens}, Common: {set(p6_tokens) & set(gt_tokens)}")
print(f"Test 7 tokens: {p7_tokens}, Common: {set(p7_tokens) & set(gt_tokens)}")