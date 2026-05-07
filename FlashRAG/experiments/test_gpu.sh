#!/bin/bash
echo "GPU申请测试成功"
nvidia-smi
echo "当前工作目录: $(pwd)"
echo "Python路径: $(which python)"
echo "时间: $(date)"