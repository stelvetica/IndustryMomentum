"""
Lasso因子差异排查脚本
逐步对比 test_lasso_simple.py 和 factors_analysis.py 的计算逻辑
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

def main():
    print("=" * 70)
    print("Lasso因子差异排查")
    print("=" * 70)
    
    # 1. 检查factor_函数的train_periods默认值
    import inspect
    sig = inspect.signature(factor_.momentum_cross_industry_lasso)
    train_periods_default = sig.parameters['train_periods'].default
    print(f"\n[1] factor_.momentum_cross_industry_lasso 的 train_periods 默认值: {train_periods_default}")
    
    # 2. 加载数据
    print("\n[2] 加载数据...")
    prices_df = data_loader.load_price_df()
    print(f"    数据范围: {prices_df.index[0].date()} 到 {prices_df.index[-1].date()}")
    
    # 3. 计算因子
    print("\n[3] 计算Lasso因子...")
    factor_df = factor_.momentum_cross_industry_lasso(
        prices_df, 
        window=20, 
        rebalance_freq=20
    )
    print(f"    因子形状: {factor_df.shape}")
    print(f"    因子非空值: {factor_df.notna().sum().sum()}")
    
    # 4. 计算forward_returns（两种方式）
    print("\n[4] 计算forward_returns...")
    
    # 方式A: test_lasso_simple.py 的方式
    forward_returns_A = fa.calculate_monthly_forward_returns(prices_df, prices_df.index)
    print(f"    方式A (calculate_monthly_forward_returns): 形状={forward_returns_A.shape}")
    
    # 5. 获取调仓日期
    print("\n[5] 获取调仓日期...")
    monthly_dates = fa.get_monthly_rebalance_dates(prices_df.index)
    print(f"    总月末日期数: {len(monthly_dates)}")
    
    # 6. 确定测试范围
    end_date = pd.Timestamp('2022-11-30')
    
    # 方式A: test_lasso_simple.py 的方式
    start_date_A = end_date - pd.DateOffset(years=10)
    test_dates_A = [d for d in monthly_dates if start_date_A <= d <= end_date]
    
    # 方式B: factors_analysis.py 的方式 (从12月底开始)
    start_date_B = pd.Timestamp('2012-12-31')
    test_dates_B = [d for d in monthly_dates if start_date_B <= d <= end_date]
    
    print(f"\n[6] 测试日期范围:")
    print(f"    方式A (test_lasso_simple): {test_dates_A[0].date()} 到 {test_dates_A[-1].date()}, 共{len(test_dates_A)}个月")
    print(f"    方式B (factors_analysis): {test_dates_B[0].date()} 到 {test_dates_B[-1].date()}, 共{len(test_dates_B)}个月")
    
    # 7. 计算IC（两种方式）
    print("\n[7] 计算IC...")
    
    def calc_ic_for_dates(test_dates, factor_df, forward_returns_df, name):
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
        
        if ic_list:
            ic_df = pd.DataFrame(ic_list)
            print(f"    {name}:")
            print(f"      IC均值: {ic_df['ic'].mean():.4f}")
            print(f"      ICIR: {ic_df['ic'].mean() / ic_df['ic'].std():.4f}")
            print(f"      IC胜率: {(ic_df['ic'] > 0).mean():.2%}")
            print(f"      样本数: {len(ic_df)}")
            return ic_df
        return None
    
    ic_A = calc_ic_for_dates(test_dates_A, factor_df, forward_returns_A, "方式A (test_lasso_simple)")
    ic_B = calc_ic_for_dates(test_dates_B, factor_df, forward_returns_A, "方式B (factors_analysis日期)")
    
    # 8. 检查因子值和forward_returns的对齐
    print("\n[8] 检查因子值和forward_returns的对齐...")
    sample_date = test_dates_B[0]
    print(f"    样本日期: {sample_date.date()}")
    
    fac_sample = factor_df.loc[sample_date]
    ret_sample = forward_returns_A.loc[sample_date]
    
    print(f"    因子值非空数: {fac_sample.notna().sum()}")
    print(f"    forward_returns非空数: {ret_sample.notna().sum()}")
    print(f"    因子值范围: {fac_sample.min():.4f} 到 {fac_sample.max():.4f}")
    print(f"    forward_returns范围: {ret_sample.min():.4f} 到 {ret_sample.max():.4f}")
    
    # 9. 模拟factors_analysis的完整流程
    print("\n[9] 模拟factors_analysis的完整流程...")
    
    # 使用DataContainer加载数据
    data = fa.DataContainer(
        data_loader.DEFAULT_CACHE_FILE,
        end_date='2022-11-30',
        backtest_years=10,
        load_constituent=False
    )
    
    print(f"    DataContainer.first_holding_date: {data.first_holding_date}")
    print(f"    DataContainer.last_holding_date: {data.last_holding_date}")
    
    # 计算因子（通过factors_analysis的方式）
    factor_df_fa = fa.calculate_factor(
        'momentum_cross_industry_lasso',
        data,
        window=20,
        rebalance_freq=20
    )
    print(f"    factors_analysis计算的因子形状: {factor_df_fa.shape}")
    
    # 检查两种方式计算的因子是否相同
    common_dates = factor_df.index.intersection(factor_df_fa.index)
    if len(common_dates) > 0:
        sample_date = common_dates[-10]  # 取一个靠后的日期
        fac1 = factor_df.loc[sample_date]
        fac2 = factor_df_fa.loc[sample_date]
        
        print(f"\n    对比日期: {sample_date.date()}")
        print(f"    直接计算的因子值（前5个）: {fac1.head().values}")
        print(f"    factors_analysis计算的因子值（前5个）: {fac2.head().values}")
        
        # 检查是否相同
        diff = (fac1 - fac2).abs()
        print(f"    最大差异: {diff.max():.6f}")
    
    # 10. 使用factors_analysis的IC计算函数
    print("\n[10] 使用factors_analysis的IC计算函数...")
    
    forward_returns_df_fa = fa.calculate_monthly_forward_returns(data.prices_df, data.prices_df.index)
    
    ic_results = fa.calculate_ic_ir(
        factor_df_fa, 
        forward_returns_df_fa, 
        monthly_rebalance=True,
        unified_start_date=data.first_holding_date
    )
    
    print(f"    factors_analysis计算的IC均值: {ic_results['rank_ic_mean']:.4f}")
    print(f"    factors_analysis计算的ICIR: {ic_results['rank_icir']:.4f}")
    print(f"    factors_analysis计算的IC胜率: {ic_results['ic_win_rate']:.2%}")
    print(f"    IC序列长度: {len(ic_results['rank_ic_series'])}")
    
    # 11. 对比IC序列
    print("\n[11] 对比IC序列...")
    if ic_B is not None:
        ic_series_fa = ic_results['rank_ic_series']
        ic_series_simple = ic_B.set_index('date')['ic']
        
        common_ic_dates = ic_series_fa.index.intersection(ic_series_simple.index)
        print(f"    共同日期数: {len(common_ic_dates)}")
        
        if len(common_ic_dates) > 0:
            ic_diff = (ic_series_fa.loc[common_ic_dates] - ic_series_simple.loc[common_ic_dates]).abs()
            print(f"    IC差异最大值: {ic_diff.max():.6f}")
            print(f"    IC差异均值: {ic_diff.mean():.6f}")
            
            # 如果有差异，打印几个例子
            if ic_diff.max() > 0.001:
                print("\n    IC差异较大的日期:")
                for date in ic_diff.nlargest(3).index:
                    print(f"      {date.date()}: factors_analysis={ic_series_fa.loc[date]:.4f}, simple={ic_series_simple.loc[date]:.4f}")


if __name__ == "__main__":
    main()

