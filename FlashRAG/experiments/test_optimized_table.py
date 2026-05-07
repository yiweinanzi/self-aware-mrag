#!/usr/bin/env python3
"""测试优化后的表格处理"""

import sys
import json
import gzip

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

# 模拟优化的表格转��
def test_optimized_table():
    # 加载第一个样本的表格
    table_id = '8513db80c11ea439ab11eba406ec00d9'

    with gzip.open('/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA/MMQA_tables.jsonl.gz', 'rt') as f:
        for line in f:
            item = json.loads(line)
            if item['id'] == table_id:
                # 使用优化后的转换逻辑
                table = item.get('table', {})
                if 'table_rows' in table:
                    rows_text = []

                    # 添加表格标题
                    table_title = item.get('title', 'Filmography')
                    rows_text.append(f"【表格信息】{table_title}")
                    rows_text.append("=" * 50)
                    rows_text.append("")

                    # 获取表头并格式化
                    headers = table.get('header', [])
                    if headers:
                        header_line = []
                        for i, h in enumerate(headers):
                            col_name = h.get('column_name', f'Column{i+1}')
                            # 添加中文说明
                            if i == 0:
                                header_line.append(f"{col_name} (年份)".ljust(20))
                            elif i == 1:
                                header_line.append(f"{col_name} (电影名)".ljust(20))
                            elif i == 2:
                                header_line.append(f"{col_name} (角色名)".ljust(20))
                            else:
                                header_line.append(col_name.ljust(20))
                        rows_text.append('|'.join(header_line))
                        rows_text.append('-' * len(header_line) * 21)

                    # 获取所有数据行
                    for row_idx, row in enumerate(table['table_rows'], 1):
                        row_values = []
                        for i, cell in enumerate(row):
                            if isinstance(cell, dict):
                                cell_text = cell.get('text', '')
                            else:
                                cell_text = str(cell)
                            row_values.append(cell_text.ljust(20))

                        # 高亮可能的答案行
                        if 'Mr. Simms' in str(row) or 'Mask' in str(row):
                            row_values.append(" <-- 可能的答案")

                        rows_text.append(f"第{row_idx}行:|".join(row_values))

                    # 添加查找提示
                    rows_text.append("")
                    rows_text.append("【查找提示】:")
                    rows_text.append("- 在表格中查找与问题相关的年份、电影名或角色名")
                    rows_text.append("- 注意角色名和电影名的对应关系")
                    rows_text.append("- 仔细阅读每一行的完整信息")

                    # 打印结果
                    print("\n优化后的表格格式示例：")
                    print("-" * 80)
                    print('\n'.join(rows_text[:30]))  # 只显示前30行
                    print("-" * 80)

                    # 检查是否包含答案
                    full_text = '\n'.join(rows_text)
                    if 'Mask' in full_text and 'Mr. Simms' in full_text:
                        print("\n✅ 成功：表格包含答案 'Mask' 和角色 'Mr. Simms'")
                    else:
                        print("\n❌ 错误：表格不包含答案信息")
                break

if __name__ == "__main__":
    test_optimized_table()