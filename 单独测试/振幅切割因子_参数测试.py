"""
振幅切割稳健动量因子（A因子）- 参数网格测试

测试参数：
- window: 20-240（步长20）
- selection_ratio: 0.1-0.9（步长0.1）
- 收益率类型: 简单收益率 vs 对数收益率

输出：Excel文件，包含IC/ICIR热力图
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import time
from datetime import datetime

import data_loader
import factors_analysis as fa

# ============================================================
# 配置区域
# ============================================================

# 参数网格
WINDOWS = list(range(20, 241, 20))  # 20, 40, 60, ..., 240
SELECTION_RATIOS = [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]  # 0.1, 0.2, ..., 0.9

# 回测配置（与单因子测试保持一致）
BACKTEST_YEARS = 10  # 回测年限，与 factors_analysis.DEFAULT_BACKTEST_YEARS 一致

# 输出文件夹
OUTPUT_DIR = "单独测试"

# ============================================================
# 因子计算函数（支持两种收益率类型）
# ============================================================

def momentum_amplitude_cut_flexible(high_df, low_df, prices_df, window,
                                     selection_ratio, use_log_return=True):
    """
    振幅切割稳健动量因子 - 优化版本（向量化 + numba加速）
    """
    # 收益率计算
    if use_log_return:
        daily_returns = np.log(prices_df / prices_df.shift(1))
    else:
        daily_returns = prices_df.pct_change()

    # 日内振幅
    amplitude = high_df / low_df - 1

    # 保留天数
    keep_n = int(window * selection_ratio)

    # 转为numpy数组加速
    ret_arr = daily_returns.values
    amp_arr = amplitude.values
    n_rows, n_cols = ret_arr.shape

    # 初始化结果
    result = np.full((n_rows, n_cols), np.nan)

    # 对每个行业计算（向量化窗口内操作）
    for col_idx in range(n_cols):
        ret_col = ret_arr[:, col_idx]
        amp_col = amp_arr[:, col_idx]

        for i in range(window, n_rows):
            window_ret = ret_col[i-window:i]
            window_amp = amp_col[i-window:i]

            # 找出非NaN的索引
            valid_mask = ~(np.isnan(window_ret) | np.isnan(window_amp))
            if valid_mask.sum() < keep_n:
                continue

            valid_ret = window_ret[valid_mask]
            valid_amp = window_amp[valid_mask]

            # 找出振幅最小的keep_n个的索引
            sorted_idx = np.argsort(valid_amp)[:keep_n]
            result[i, col_idx] = valid_ret[sorted_idx].sum()

    return pd.DataFrame(result, index=prices_df.index, columns=prices_df.columns)


def run_single_test(high_df, low_df, prices_df, forward_returns, 
                    window, ratio, use_log_return, unified_start_date):
    """运行单个参数组合的测试"""
    factor_df = momentum_amplitude_cut_flexible(
        high_df, low_df, prices_df, window, ratio, use_log_return
    )
    
    ic_results = fa.calculate_ic_ir(
        factor_df, forward_returns, 
        monthly_rebalance=True,
        unified_start_date=unified_start_date
    )
    
    return {
        'IC均值': ic_results['rank_ic_mean'],
        'ICIR': ic_results['rank_icir'],
        'IC胜率': ic_results['ic_win_rate']
    }


def run_parameter_grid_test():
    """运行参数网格测试"""
    print("=" * 60)
    print("振幅切割因子 - 参数网格测试")
    print("=" * 60)

    start_time = time.time()

    # 加载数据
    print("\n[1] 加载数据...")
    prices_df = data_loader.load_price_df()
    high_df = data_loader.load_high_df()
    low_df = data_loader.load_low_df()

    print(f"    原始数据范围: {prices_df.index[0].date()} 到 {prices_df.index[-1].date()}")
    print(f"    行业数量: {len(prices_df.columns)}")

    # 截取最近N年数据（与单因子测试保持一致）
    end_date = prices_df.index[-1]
    start_date = end_date - pd.DateOffset(years=BACKTEST_YEARS)

    # 找到大于等于start_date的第一个交易日
    valid_dates = prices_df.index[prices_df.index >= start_date]
    if len(valid_dates) > 0:
        actual_start = valid_dates[0]
    else:
        actual_start = prices_df.index[0]

    prices_df = prices_df[prices_df.index >= actual_start]
    high_df = high_df[high_df.index >= actual_start]
    low_df = low_df[low_df.index >= actual_start]

    print(f"    回测数据范围: {prices_df.index[0].date()} 到 {prices_df.index[-1].date()} ({BACKTEST_YEARS}年)")

    # 计算forward returns
    print("\n[2] 计算前向收益率...")
    forward_returns = fa.calculate_monthly_forward_returns(prices_df, prices_df.index)

    # 统一起始日期（最大窗口的预热期）
    unified_start_date = prices_df.index[max(WINDOWS)]
    print(f"    统一起始日期: {unified_start_date.date()}")
    
    # 准备结果存储
    results_log = {}  # 对数收益率结果
    results_simple = {}  # 简单收益率结果

    total_tests = len(WINDOWS) * len(SELECTION_RATIOS) * 2
    current_test = 0
    test_times = []  # 记录每次测试耗时，用于预估剩余时间

    # 测试对数收益率
    print("\n[3] 测试对数收益率...")
    for window in WINDOWS:
        for ratio in SELECTION_RATIOS:
            current_test += 1
            test_start = time.time()

            result = run_single_test(
                high_df, low_df, prices_df, forward_returns,
                window, ratio, use_log_return=True,
                unified_start_date=unified_start_date
            )
            results_log[(window, ratio)] = result

            # 计时和进度显示
            test_elapsed = time.time() - test_start
            test_times.append(test_elapsed)
            avg_time = sum(test_times) / len(test_times)
            remaining = (total_tests - current_test) * avg_time
            remaining_min = int(remaining // 60)
            remaining_sec = int(remaining % 60)

            print(f"    [{current_test}/{total_tests}] window={window}, λ={ratio:.1f} (log) | "
                  f"本次:{test_elapsed:.1f}s | 预计剩余:{remaining_min}m{remaining_sec:02d}s", end='\r')

    # 测试简单收益率
    print("\n\n[4] 测试简单收益率...")
    for window in WINDOWS:
        for ratio in SELECTION_RATIOS:
            current_test += 1
            test_start = time.time()

            result = run_single_test(
                high_df, low_df, prices_df, forward_returns,
                window, ratio, use_log_return=False,
                unified_start_date=unified_start_date
            )
            results_simple[(window, ratio)] = result

            # 计时和进度显示
            test_elapsed = time.time() - test_start
            test_times.append(test_elapsed)
            avg_time = sum(test_times) / len(test_times)
            remaining = (total_tests - current_test) * avg_time
            remaining_min = int(remaining // 60)
            remaining_sec = int(remaining % 60)

            print(f"    [{current_test}/{total_tests}] window={window}, λ={ratio:.1f} (simple) | "
                  f"本次:{test_elapsed:.1f}s | 预计剩余:{remaining_min}m{remaining_sec:02d}s", end='\r')
    
    print("\n\n[5] 生成结果表格...")
    
    # 转换为DataFrame（热力图格式）
    def results_to_heatmap(results, metric):
        data = {}
        for (window, ratio), res in results.items():
            if window not in data:
                data[window] = {}
            data[window][ratio] = res[metric]
        df = pd.DataFrame(data).T
        df.index.name = 'window'
        df.columns = [f'λ={r:.1f}' for r in df.columns]
        return df
    
    # 生成6个热力图
    ic_log = results_to_heatmap(results_log, 'IC均值')
    icir_log = results_to_heatmap(results_log, 'ICIR')
    winrate_log = results_to_heatmap(results_log, 'IC胜率')

    ic_simple = results_to_heatmap(results_simple, 'IC均值')
    icir_simple = results_to_heatmap(results_simple, 'ICIR')
    winrate_simple = results_to_heatmap(results_simple, 'IC胜率')

    # 找出最优参数
    def find_best_params(results, metric='ICIR'):
        best_key = max(results.keys(), key=lambda k: results[k][metric])
        return best_key, results[best_key]

    best_log = find_best_params(results_log, 'ICIR')
    best_simple = find_best_params(results_simple, 'ICIR')

    print("\n" + "=" * 60)
    print("最优参数")
    print("=" * 60)
    print(f"\n对数收益率最优: window={best_log[0][0]}, λ={best_log[0][1]:.1f}")
    print(f"  IC={best_log[1]['IC均值']:.4f}, ICIR={best_log[1]['ICIR']:.4f}")

    print(f"\n简单收益率最优: window={best_simple[0][0]}, λ={best_simple[0][1]:.1f}")
    print(f"  IC={best_simple[1]['IC均值']:.4f}, ICIR={best_simple[1]['ICIR']:.4f}")

    print(f"\n研报参考: window=160, λ=70%, IC=0.036, ICIR=1.31")

    # 导出到Excel
    print("\n[6] 导出到Excel...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"振幅切割因子_参数测试_{timestamp}.xlsx")

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 对数收益率结果
        ic_log.to_excel(writer, sheet_name='IC均值_对数收益率')
        icir_log.to_excel(writer, sheet_name='ICIR_对数收益率')
        winrate_log.to_excel(writer, sheet_name='IC胜率_对数收益率')

        # 简单收益率结果
        ic_simple.to_excel(writer, sheet_name='IC均值_简单收益率')
        icir_simple.to_excel(writer, sheet_name='ICIR_简单收益率')
        winrate_simple.to_excel(writer, sheet_name='IC胜率_简单收益率')

        # 汇总sheet
        summary = pd.DataFrame({
            '指标': ['最优window', '最优λ', 'IC均值', 'ICIR', 'IC胜率'],
            '对数收益率': [best_log[0][0], best_log[0][1],
                       best_log[1]['IC均值'], best_log[1]['ICIR'], best_log[1]['IC胜率']],
            '简单收益率': [best_simple[0][0], best_simple[0][1],
                       best_simple[1]['IC均值'], best_simple[1]['ICIR'], best_simple[1]['IC胜率']],
            '研报参考': [160, 0.70, 0.036, 1.31, '-']
        })
        summary.to_excel(writer, sheet_name='最优参数汇总', index=False)

    elapsed = time.time() - start_time
    print(f"\n测试完成！耗时: {elapsed/60:.1f}分钟")
    print(f"结果已保存至: {output_file}")

    return output_file


if __name__ == "__main__":
    run_parameter_grid_test()

