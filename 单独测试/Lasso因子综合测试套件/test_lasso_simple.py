"""
Lasso因子简单测试脚本
测试train_periods=12的效果
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import data_loader
import factor_
import factors_analysis as fa
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def test_lasso_factor(end_date=None, years=10):
    """测试Lasso因子"""
    print("=" * 60)
    print("Lasso因子测试 (train_periods=12)")
    print("=" * 60)
    
    # 加载数据
    print("\n正在加载数据...")
    prices_df = data_loader.load_price_df()
    print(f"数据范围: {prices_df.index[0].date()} 到 {prices_df.index[-1].date()}")
    print(f"行业数量: {len(prices_df.columns)}")
    
    # 计算因子
    print("\n正在计算Lasso因子...")
    factor_df = factor_.momentum_cross_industry_lasso(
        prices_df, 
        window=20, 
        rebalance_freq=20
    )
    print("因子计算完成")
    
    # 计算forward_returns
    forward_returns_df = fa.calculate_monthly_forward_returns(prices_df, prices_df.index)
    
    # 获取月末日期
    monthly_dates = fa.get_monthly_rebalance_dates(prices_df.index)
    
    # 确定测试范围
    if end_date is not None:
        end_dt = pd.Timestamp(end_date)
    else:
        end_dt = monthly_dates[-2]  # 最后一个完整月
    
    start_dt = end_dt - pd.DateOffset(years=years)
    
    test_dates = [d for d in monthly_dates if start_dt <= d <= end_dt]
    print(f"\n测试范围: {test_dates[0].date()} 到 {test_dates[-1].date()}")
    print(f"测试月数: {len(test_dates)}")
    
    # 计算IC
    ic_list = []
    for date in test_dates:
        if date not in factor_df.index or date not in forward_returns_df.index:
            continue
        
        fac = factor_df.loc[date]
        ret = forward_returns_df.loc[date]
        
        valid = fac.notna() & ret.notna()
        if valid.sum() < 5:
            continue
        
        ic, _ = stats.spearmanr(fac[valid], ret[valid])
        if not np.isnan(ic):
            ic_list.append({'date': date, 'ic': ic})
    
    ic_df = pd.DataFrame(ic_list)
    
    print("\n" + "=" * 60)
    print("IC统计")
    print("=" * 60)
    print(f"IC均值: {ic_df['ic'].mean():.4f}")
    print(f"IC标准差: {ic_df['ic'].std():.4f}")
    print(f"ICIR: {ic_df['ic'].mean() / ic_df['ic'].std():.4f}")
    print(f"IC胜率: {(ic_df['ic'] > 0).mean():.2%}")
    print(f"样本数: {len(ic_df)}")
    
    # 计算分层收益
    print("\n" + "=" * 60)
    print("分层回测 (G5=因子值最高组)")
    print("=" * 60)
    
    g5_returns = []
    bench_returns = []
    
    for i, date in enumerate(test_dates[:-1]):
        next_date = test_dates[i + 1]
        
        if date not in factor_df.index:
            continue
        
        fac = factor_df.loc[date]
        valid = fac.notna()
        if valid.sum() < 10:
            continue
        
        fac_valid = fac[valid]
        
        # 分5组，取最高组
        n_per_group = len(fac_valid) // 5
        sorted_idx = fac_valid.sort_values(ascending=False).index
        g5_industries = sorted_idx[:n_per_group]
        
        # 计算收益
        if date in prices_df.index and next_date in prices_df.index:
            ret = (prices_df.loc[next_date] / prices_df.loc[date]) - 1
            g5_ret = ret[g5_industries].mean()
            bench_ret = ret[valid].mean()
            g5_returns.append(g5_ret)
            bench_returns.append(bench_ret)
    
    g5_excess = np.array(g5_returns) - np.array(bench_returns)
    
    print(f"G5月均收益: {np.mean(g5_returns)*100:.2f}%")
    print(f"基准月均收益: {np.mean(bench_returns)*100:.2f}%")
    print(f"G5月均超额: {np.mean(g5_excess)*100:.2f}%")
    print(f"G5年化超额: {np.mean(g5_excess)*12*100:.2f}%")
    print(f"G5超额IR: {np.mean(g5_excess)/np.std(g5_excess)*np.sqrt(12):.2f}")
    
    return ic_df, g5_excess


if __name__ == "__main__":
    # 测试研报截止日期
    print("\n>>> 测试1: 研报截止日期 (2022-11-30)")
    test_lasso_factor('2022-11-30', years=10)
    
    print("\n\n>>> 测试2: 最新数据")
    test_lasso_factor(None, years=10)

