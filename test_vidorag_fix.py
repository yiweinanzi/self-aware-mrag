#!/usr/bin/env python3
"""
Test ViDoRAG fix
"""

import sys
import os
sys.path.insert(0, '/data0/home/zqwang/ACL')
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')
sys.path.insert(0, '/data0/home/zqwang/ACL/ViDoRAG-main')

from experiments.baselines.vidorag_pipeline import create_vidorag_pipeline

# Test ViDoRAG import and pipeline creation
print("Testing ViDoRAG import...")
try:
    from agent.agent_prompt import answer_prompt
    print("✅ ViDoRAG imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")

# Create a mock config
config = {
    'retrieval_topk': 5,
}

# Test pipeline creation
print("\nTesting pipeline creation...")
try:
    # Create mock qwen3vl
    qwen3vl = type('MockQwen3VL', (), {
        'generate': lambda self, prompt, images=None, **kwargs: "test answer"
    })()

    pipeline = create_vidorag_pipeline(qwen3vl, None, config)
    print("✅ Pipeline created successfully")
except Exception as e:
    print(f"❌ Pipeline creation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Test complete")