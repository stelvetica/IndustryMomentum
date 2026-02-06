"""
龙头领先修正动量因子 - 分割参数测试脚本

测试不同的 split_ratio 参数对因子表现的影响
固定使用 window=20（1个月），与原文一致

通过临时修改 FACTOR_CONFIG 的 split_ratios，复用 analyze_single_factor 逻辑
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_loader
import factors_analysis as fa
import pandas as pd

# ============================================================
# 配置区域
# ============================================================

FACTOR_NAME = 'momentum_lead_lag_enhanced'
SPLIT_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 要测试的分割参数列表
WINDOWS = [20, 60, 120, 240]  # 要测试的窗口列表（None 表示使用因子默认配置 [20]，例如: [20, 60, 120]）
END_DATE = None  # None 表示使用最新数据

# ============================================================
# 主程序
# ============================================================

def run_split_ratio_test():
    """运行分割参数测试"""
    import time
    start_time = time.time()
    
    print("=" * 70)
    print("龙头领先修正动量因子 - 分割参数测试")
    print("=" * 70)
    print(f"分割参数: {SPLIT_RATIOS}")
    print()
    
    # 临时修改因子配置
    original_config = fa.FACTOR_CONFIG[FACTOR_NAME].copy()
    fa.FACTOR_CONFIG[FACTOR_NAME]['split_ratios'] = SPLIT_RATIOS
    if WINDOWS is not None:
        fa.FACTOR_CONFIG[FACTOR_NAME]['lookback_windows'] = WINDOWS
    
    try:
        # 加载数据
        print("正在加载数据...")
        data = fa.DataContainer(
            data_loader.DEFAULT_CACHE_FILE,
            end_date=END_DATE,
            backtest_years=fa.DEFAULT_BACKTEST_YEARS,
            load_constituent=True
        )
        
        # 获取窗口列表
        windows = fa.get_factor_windows(FACTOR_NAME)
        
        # 直接调用 analyze_single_factor（会自动处理 unified_start_date）
        all_results = fa.analyze_single_factor(FACTOR_NAME, data, windows=windows)
        
        # 提取结果并汇总
        factor_results = all_results.get(FACTOR_NAME, {})
        
        # 汇总结果
        print("\n" + "=" * 70)
        print("【因子汇总指标】- 不同分割参数对比")
        print("=" * 70)
        
        summary_data = []
        for key, result in factor_results.items():
            if result is None:
                continue
            
            # key 格式为 (window, split_ratio)
            window, split_ratio = key
            
            row = {'window': window, 'split_ratio': split_ratio}
            row['IC均值'] = result['ic_mean']
            row['ICIR'] = result['icir']
            row['IC胜率'] = result['ic_win_rate']
            
            if 'G5' in result['excess_metrics']:
                g5 = result['excess_metrics']['G5']
                row['G5_累计收益率(%)'] = g5['累计收益率(%)']
                row['G5_年化收益率(%)'] = g5['年化收益率(%)']
                row['G5_超额累计收益率(%)'] = g5['超额累计收益率(%)']
                row['G5_超额年化收益率(%)'] = g5['超额年化收益率(%)']
                row['G5_年化波动率(%)'] = g5['年化波动率(%)']
                row['G5_夏普比率'] = g5['夏普比率']
                row['G5_最大回撤(%)'] = g5['最大回撤(%)']
                row['G5_多头胜率(%)'] = g5['多头胜率(%)']
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        if not summary_df.empty:
            summary_df = summary_df.set_index(['window', 'split_ratio'])
            print(summary_df.round(4).to_string())
            
            # 找出最优参数
            best_ic_idx = summary_df['IC均值'].idxmax()
            best_icir_idx = summary_df['ICIR'].idxmax()
            best_excess_idx = summary_df['G5_超额年化收益率(%)'].idxmax()
            
            print(f"\n【最优参数】")
            print(f"  最优IC: {best_ic_idx} (IC={summary_df.loc[best_ic_idx, 'IC均值']:.4f})")
            print(f"  最优ICIR: {best_icir_idx} (ICIR={summary_df.loc[best_icir_idx, 'ICIR']:.4f})")
            print(f"  最优超额收益: {best_excess_idx} (超额年化={summary_df.loc[best_excess_idx, 'G5_超额年化收益率(%)']:.2f}%)")
            
            # 导出到Excel
            output_file = f"factor分析/龙头领先因子_分割参数测试_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            summary_df.to_excel(output_file)
            print(f"\n结果已导出到: {output_file}")
        
    finally:
        # 恢复原始配置
        fa.FACTOR_CONFIG[FACTOR_NAME] = original_config
    
    # 计算耗时
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"\n总耗时: {minutes}分{seconds}秒")
    
    return summary_df if 'summary_df' in dir() else None, all_results


if __name__ == "__main__":
    summary_df, results = run_split_ratio_test()

