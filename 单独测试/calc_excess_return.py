# -*- coding: utf-8 -*-
"""
手动计算超额收益率和超额年化收益率
包含两种计算方式：差值法和净值法
"""
import pandas as pd
import numpy as np

# 读取数据
file_path = r'C:\Users\MECHREVO\001_TEMP\Quant\行业动量\factor分析\positive_bubble_backtest_alltime_015040.xlsx'
df = pd.read_excel(file_path, sheet_name='净值序列')

# 重命名列（根据数据结构）
df.columns = ['日期', '策略净值', '基准净值', '超额净值']
df['日期'] = pd.to_datetime(df['日期'])

print('数据预览:')
print(df.head(10))
print(f'\n数据范围: {df["日期"].iloc[0]} 至 {df["日期"].iloc[-1]}')
print(f'总行数: {len(df)}')

# 计算超额收益率（每期）
df['策略收益率'] = df['策略净值'].pct_change()
df['基准收益率'] = df['基准净值'].pct_change()
df['超额收益率_差值法'] = df['策略收益率'] - df['基准收益率']
df['超额收益率_净值法'] = df['超额净值'].pct_change()

# 计算累计超额收益（两种方式）
df['累计超额收益_差值法'] = (df['策略净值'] - df['基准净值']) / df['基准净值'].iloc[0]  # 简单差值
df['累计超额收益_净值法'] = df['策略净值'] / df['基准净值'] - 1  # 净值相除

# 计算年化指标
start_date = df['日期'].iloc[0]
end_date = df['日期'].iloc[-1]
n_years = (end_date - start_date).days / 365.25

# 策略
strategy_total_return = df['策略净值'].iloc[-1] / df['策略净值'].iloc[0] - 1
strategy_annual_return = (1 + strategy_total_return) ** (1/n_years) - 1

# 基准
benchmark_total_return = df['基准净值'].iloc[-1] / df['基准净值'].iloc[0] - 1
benchmark_annual_return = (1 + benchmark_total_return) ** (1/n_years) - 1

# ========== 超额收益（两种方式）==========
# 方式1：差值法 - 简单相减
excess_total_diff = strategy_total_return - benchmark_total_return
excess_annual_diff = strategy_annual_return - benchmark_annual_return

# 方式2：净值法 - 用超额净值计算
excess_total_nav = df['超额净值'].iloc[-1] / df['超额净值'].iloc[0] - 1
excess_annual_nav = (1 + excess_total_nav) ** (1/n_years) - 1

print(f'\n{"="*50}')
print(f'收益统计')
print(f'{"="*50}')
print(f'回测年数: {n_years:.2f} 年')
print(f'\n【策略】')
print(f'  总收益: {strategy_total_return*100:.2f}%')
print(f'  年化收益: {strategy_annual_return*100:.2f}%')
print(f'\n【基准】')
print(f'  总收益: {benchmark_total_return*100:.2f}%')
print(f'  年化收益: {benchmark_annual_return*100:.2f}%')
print(f'\n【超额收益 - 差值法】(策略-基准)')
print(f'  超额总收益: {excess_total_diff*100:.2f}%')
print(f'  超额年化收益: {excess_annual_diff*100:.2f}%')
print(f'\n【超额收益 - 净值法】(策略/基准-1)')
print(f'  超额总收益: {excess_total_nav*100:.2f}%')
print(f'  超额年化收益: {excess_annual_nav*100:.2f}%')

# 保存到新文件
output_path = r'C:\Users\MECHREVO\001_TEMP\Quant\行业动量\factor分析\positive_bubble_backtest_alltime_015040_手动计算.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # Sheet1: 净值序列和收益率
    df.to_excel(writer, sheet_name='净值与收益率', index=False)

    # Sheet2: 汇总统计（包含两种方式）
    summary = pd.DataFrame({
        '指标': [
            '回测年数',
            '---策略---',
            '策略总收益',
            '策略年化收益',
            '---基准---',
            '基准总收益',
            '基准年化收益',
            '---超额(差值法: 策略-基准)---',
            '超额总收益(差值法)',
            '超额年化收益(差值法)',
            '---超额(净值法: 策略/基准-1)---',
            '超额总收益(净值法)',
            '超额年化收益(净值法)',
        ],
        '数值': [
            n_years,
            '',
            strategy_total_return,
            strategy_annual_return,
            '',
            benchmark_total_return,
            benchmark_annual_return,
            '',
            excess_total_diff,
            excess_annual_diff,
            '',
            excess_total_nav,
            excess_annual_nav,
        ],
        '百分比显示': [
            f'{n_years:.2f}年',
            '',
            f'{strategy_total_return*100:.2f}%',
            f'{strategy_annual_return*100:.2f}%',
            '',
            f'{benchmark_total_return*100:.2f}%',
            f'{benchmark_annual_return*100:.2f}%',
            '',
            f'{excess_total_diff*100:.2f}%',
            f'{excess_annual_diff*100:.2f}%',
            '',
            f'{excess_total_nav*100:.2f}%',
            f'{excess_annual_nav*100:.2f}%',
        ]
    })
    summary.to_excel(writer, sheet_name='汇总统计', index=False)

print(f'\n结果已保存到: {output_path}')

