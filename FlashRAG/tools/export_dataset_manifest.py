#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
✅ P1-3: 数据集清单导出工具

导出数据集清单CSV文件，包含：
- 数据集名称
- 版本
- 样本数
- 文件路径
- 校验和（MD5/SHA256）
- 数据集描述

输出文件：datasets_manifest.csv
"""

import os
import sys
import hashlib
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flashrag.config.dataset_registry import get_dataset_registry


def calculate_file_hash(file_path: str, algorithm: str = 'md5') -> Optional[str]:
    """
    计算文件的哈希值
    
    Args:
        file_path: 文件路径
        algorithm: 哈希算法（'md5' 或 'sha256'）
    
    Returns:
        str: 哈希值（十六进制字符串）
    """
    if not os.path.exists(file_path):
        return None
    
    try:
        if algorithm == 'md5':
            hasher = hashlib.md5()
        elif algorithm == 'sha256':
            hasher = hashlib.sha256()
        else:
            raise ValueError(f"不支持的哈希算法: {algorithm}")
        
        # 分块读取以处理大文件
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    except Exception as e:
        print(f"  ⚠️  计算哈希失败 ({file_path}): {e}")
        return None


def calculate_dir_hash(dir_path: str, algorithm: str = 'md5') -> Optional[str]:
    """
    计算目录下所有文件的综合哈希值
    
    Args:
        dir_path: 目录路径
        algorithm: 哈希算法
    
    Returns:
        str: 综合哈希值
    """
    if not os.path.exists(dir_path):
        return None
    
    try:
        if algorithm == 'md5':
            hasher = hashlib.md5()
        elif algorithm == 'sha256':
            hasher = hashlib.sha256()
        else:
            raise ValueError(f"不支持的哈希算法: {algorithm}")
        
        # 按文件名排序以保证一致性
        file_paths = []
        for root, dirs, files in os.walk(dir_path):
            for filename in sorted(files):
                file_path = os.path.join(root, filename)
                file_paths.append(file_path)
        
        # 计算每个文件的哈希并合并
        for file_path in sorted(file_paths):
            try:
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b''):
                        hasher.update(chunk)
            except Exception as e:
                print(f"  ⚠️  跳过文件 {file_path}: {e}")
                continue
        
        return hasher.hexdigest()
    except Exception as e:
        print(f"  ⚠️  计算目录哈希失败 ({dir_path}): {e}")
        return None


def get_dataset_info(dataset_id: str, dataset_config) -> Dict[str, any]:
    """
    获取数据集详细信息
    
    Args:
        dataset_id: 数据集ID
        dataset_config: 数据集配置对象
    
    Returns:
        dict: 数据集信息
    """
    info = {
        'dataset_id': dataset_id,
        'name': dataset_config.name,
        'enabled': dataset_config.enabled,
        'description': dataset_config.description,
        'version': dataset_config.version or 'N/A',
        'path': dataset_config.path or 'N/A',
        'loader': dataset_config.loader or 'N/A',
        'num_samples': dataset_config.num_samples or 0,
        'path_exists': False,
        'path_type': 'N/A',
        'size_bytes': 0,
        'size_human': 'N/A',
        'md5_checksum': 'N/A',
        'sha256_checksum': 'N/A',
        'files_count': 0,
        'last_modified': 'N/A',
    }
    
    # 检查路径
    if dataset_config.path and os.path.exists(dataset_config.path):
        info['path_exists'] = True
        path_obj = Path(dataset_config.path)
        
        if path_obj.is_file():
            # 单个文件
            info['path_type'] = 'file'
            info['size_bytes'] = path_obj.stat().st_size
            info['size_human'] = format_bytes(info['size_bytes'])
            info['files_count'] = 1
            info['last_modified'] = datetime.fromtimestamp(
                path_obj.stat().st_mtime
            ).strftime('%Y-%m-%d %H:%M:%S')
            
            # 计算校验和（文件较小时）
            if info['size_bytes'] < 100 * 1024 * 1024:  # < 100MB
                print(f"  计算校验和: {dataset_id}")
                info['md5_checksum'] = calculate_file_hash(dataset_config.path, 'md5')
                info['sha256_checksum'] = calculate_file_hash(dataset_config.path, 'sha256')
            else:
                print(f"  跳过校验和（文件过大）: {dataset_id}")
        
        elif path_obj.is_dir():
            # 目录
            info['path_type'] = 'directory'
            
            # 计算目录大小和文件数
            total_size = 0
            file_count = 0
            latest_mtime = 0
            
            for root, dirs, files in os.walk(dataset_config.path):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    try:
                        stat = os.stat(file_path)
                        total_size += stat.st_size
                        file_count += 1
                        latest_mtime = max(latest_mtime, stat.st_mtime)
                    except:
                        pass
            
            info['size_bytes'] = total_size
            info['size_human'] = format_bytes(total_size)
            info['files_count'] = file_count
            info['last_modified'] = datetime.fromtimestamp(
                latest_mtime
            ).strftime('%Y-%m-%d %H:%M:%S') if latest_mtime > 0 else 'N/A'
            
            # 计算目录校验和（目录较小时）
            if total_size < 50 * 1024 * 1024:  # < 50MB
                print(f"  计算目录校验和: {dataset_id}")
                info['md5_checksum'] = calculate_dir_hash(dataset_config.path, 'md5')
            else:
                print(f"  跳过目录校验和（目录过大）: {dataset_id}")
    
    return info


def format_bytes(bytes_size: int) -> str:
    """格式化字节大小为人类可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def export_to_csv(dataset_infos: List[Dict], output_path: str):
    """
    导出到CSV文件
    
    Args:
        dataset_infos: 数据集信息列表
        output_path: 输出文件路径
    """
    fieldnames = [
        'dataset_id',
        'name',
        'enabled',
        'version',
        'num_samples',
        'description',
        'path',
        'path_exists',
        'path_type',
        'size_bytes',
        'size_human',
        'files_count',
        'md5_checksum',
        'sha256_checksum',
        'loader',
        'last_modified',
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for info in dataset_infos:
            # 只写入指定的字段
            row = {k: info.get(k, 'N/A') for k in fieldnames}
            writer.writerow(row)
    
    print(f"\n✅ CSV文件已导出: {output_path}")


def export_to_json(dataset_infos: List[Dict], output_path: str):
    """
    导出到JSON文件
    
    Args:
        dataset_infos: 数据集信息列表
        output_path: 输出文件路径
    """
    output = {
        'export_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_datasets': len(dataset_infos),
        'enabled_datasets': sum(1 for d in dataset_infos if d['enabled']),
        'datasets': dataset_infos
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON文件已导出: {output_path}")


def print_summary(dataset_infos: List[Dict]):
    """打印汇总信息"""
    print("\n" + "=" * 80)
    print("📊 数据集清单汇总")
    print("=" * 80)
    
    enabled = [d for d in dataset_infos if d['enabled']]
    disabled = [d for d in dataset_infos if not d['enabled']]
    
    print(f"\n总数据集数: {len(dataset_infos)}")
    print(f"  ✅ 启用: {len(enabled)}")
    print(f"  ❌ 禁用: {len(disabled)}")
    
    print(f"\n启用的数据集:")
    total_samples = 0
    total_size = 0
    
    for d in enabled:
        status = "✅" if d['path_exists'] else "⚠️ "
        print(f"  {status} {d['name']:20s} - {d['num_samples']:6d} samples, {d['size_human']:>10s}")
        total_samples += d['num_samples']
        total_size += d['size_bytes']
    
    print(f"\n总样本数: {total_samples:,}")
    print(f"总大小: {format_bytes(total_size)}")
    
    if disabled:
        print(f"\n禁用的数据集:")
        for d in disabled:
            print(f"  ❌ {d['name']:20s} - {d['description']}")
    
    print("=" * 80)


def main():
    """主函数"""
    print("=" * 80)
    print("📋 数据集清单导出工具")
    print("=" * 80)
    
    # 获取数据集注册表
    registry = get_dataset_registry()
    
    # 获取所有数据集配置
    all_datasets = {**registry.get_enabled_datasets(), **registry.get_disabled_datasets()}
    
    print(f"\n发现 {len(all_datasets)} 个数据集")
    print("正在收集详细信息...")
    print()
    
    # 收集数据集信息
    dataset_infos = []
    for dataset_id, config in all_datasets.items():
        print(f"处理: {config.name} ({dataset_id})")
        info = get_dataset_info(dataset_id, config)
        dataset_infos.append(info)
    
    # 按启用状态和名称排序
    dataset_infos.sort(key=lambda x: (not x['enabled'], x['name']))
    
    # 导出到CSV
    csv_path = '/root/autodl-tmp/FlashRAG/datasets_manifest.csv'
    export_to_csv(dataset_infos, csv_path)
    
    # 导出到JSON（附加详细信息）
    json_path = '/root/autodl-tmp/FlashRAG/datasets_manifest.json'
    export_to_json(dataset_infos, json_path)
    
    # 打印汇总
    print_summary(dataset_infos)
    
    print(f"\n✅ 导出完成！")
    print(f"  - CSV: {csv_path}")
    print(f"  - JSON: {json_path}")
    print()


if __name__ == '__main__':
    main()

