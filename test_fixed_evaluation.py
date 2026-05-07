#!/usr/bin/env python
"""Test the fixed evaluation logic"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from experiments.run_okvqa_baselines import main

# Quick 10-sample test
config = {
    'max_samples': 10,
    'output_dir': '/data0/home/zqwang/ACL/test_fix_output'
}

print("Testing fixed evaluation with 10 samples...")
print("=" * 60)

if __name__ == "__main__":
    import sys
    sys.argv = ['run_okvqa_baselines.py', '--max_samples', '10']
    main()