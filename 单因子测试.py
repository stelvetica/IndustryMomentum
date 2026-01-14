import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 导入数据加载模块和因子模块
from wind_data_loader import load_price_df
from factor_ import momentum


def calc_rank_ic(factor_series, return_series):
    """
    计算Rank IC (Spearman相关系数)
    
    参数:
        factor_series: pd.Series, 因子值序列
        return_series: pd.Series, 收益率序列
    
    返回:
        float, Rank IC值
    """
    valid_mask = factor_series.notna() & return_series.notna()
    if valid_mask.sum() < 3:
        return np.nan
    
    factor_valid = factor_series[valid_mask]
    return_valid = return_series[valid_mask]
    
    ic, _ = stats.spearmanr(factor_valid, return_valid)
    return ic


def calc_ic_ir(factor_df, forward_returns_df, rebalance_freq=20):
    """
    计算因子的IC和ICIR指标
    
    参数:
        factor_df: pd.DataFrame, 因子值 (index=日期, columns=行业)
        forward_returns_df: pd.DataFrame, 未来收益率 (index=日期, columns=行业)
        rebalance_freq: int, 调仓频率（交易日），默认20
    
    返回:
        dict: 包含以下指标的字典
            - 'ic_series': pd.Series, IC时间序列
            - 'ic_mean': float, IC均值
            - 'ic_std': float, IC标准差
            - 'icir': float, ICIR (IC均值/IC标准差)
            - 'ic_win_rate': float, IC胜率（正IC占比）
            - 'ic_abs_mean': float, |IC|均值
    """
    # 获取调仓日期
    all_dates = factor_df.index
    rebalance_indices = list(range(0, len(all_dates), rebalance_freq))
    rebalance_dates = all_dates[rebalance_indices]
    
    # 计算每个调仓日的IC
    ic_list = []
    ic_dates = []
    
    for date in rebalance_dates:
        if date not in factor_df.index or date not in forward_returns_df.index:
            continue
        
        ic = calc_rank_ic(factor_df.loc[date], forward_returns_df.loc[date])
        
        if not np.isnan(ic):
            ic_list.append(ic)
            ic_dates.append(date)
    
    # 构建IC时间序列
    ic_series = pd.Series(ic_list, index=ic_dates)
    
    # 计算IC累积序列
    ic_cumsum = ic_series.cumsum()
    
    # 计算统计指标
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    icir = ic_mean / ic_std if ic_std > 0 else np.nan
    ic_win_rate = (ic_series > 0).sum() / len(ic_series) if len(ic_series) > 0 else np.nan
    ic_abs_mean = ic_series.abs().mean()
    
    return {
        'ic_series': ic_series,
        'ic_cumsum': ic_cumsum,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'ic_win_rate': ic_win_rate,
        'ic_abs_mean': ic_abs_mean
    }


def calc_factor_ic_ir_summary(factor_dict, prices_df, rebalance_freq=20):
    """
    批量计算多个因子的IC和ICIR指标
    
    参数:
        factor_dict: dict, 因子字典 {因子名称: 因子DataFrame}
        prices_df: pd.DataFrame, 价格数据 (index=日期, columns=行业)
        rebalance_freq: int, 调仓频率（交易日），默认20
    
    返回:
        pd.DataFrame, 汇总表格，包含所有因子的IC和ICIR指标
    """
    # 计算未来收益率
    forward_returns = prices_df.pct_change(rebalance_freq).shift(-rebalance_freq)
    
    # 结果容器
    results = []
    ic_series_dict = {}
    ic_cumsum_dict = {}
    
    for factor_name, factor_df in factor_dict.items():
        ic_ir_dict = calc_ic_ir(factor_df, forward_returns, rebalance_freq)
        
        # 保存IC序列和累积序列
        ic_series_dict[factor_name] = ic_ir_dict['ic_series']
        ic_cumsum_dict[factor_name] = ic_ir_dict['ic_cumsum']
        
        results.append({
            '因子名称': factor_name,
            'IC均值': ic_ir_dict['ic_mean'],
            'ICIR': ic_ir_dict['icir'],
            'IC胜率': ic_ir_dict['ic_win_rate']
        })
    
    # 转换为DataFrame
    summary_df = pd.DataFrame(results)
    
    # 按ICIR降序排列
    summary_df = summary_df.sort_values('ICIR', ascending=False)
    
    # 构建IC序列和累积序列的DataFrame
    ic_series_df = pd.DataFrame(ic_series_dict)
    ic_cumsum_df = pd.DataFrame(ic_cumsum_dict)
    
    return summary_df, ic_series_df, ic_cumsum_df



# 使用示例
if __name__ == "__main__":
    print("开始传统动量因子IC/ICIR分析...")
    
    # 1. 从loader加载价格数据
    prices_df = load_price_df()
    
    # 2. 计算不同窗口的传统动量因子
    windows = [20, 60, 120, 240]
    factor_dict = {}
    
    for window in windows:
        factor_name = f'传统动量_{window}天'
        factor_dict[factor_name] = momentum(prices_df, window=window)
    
    # 3. 计算IC和ICIR指标
    summary, ic_series_df, ic_cumsum_df = calc_factor_ic_ir_summary(factor_dict, prices_df, rebalance_freq=20)
    
    # 4. 保存结果到Excel（多个sheet）
    output_file = '单因子测试.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        summary.to_excel(writer, sheet_name='IC_ICIR汇总', index=False)
        ic_series_df.to_excel(writer, sheet_name='IC序列', index=True)
        ic_cumsum_df.to_excel(writer, sheet_name='IC累积序列', index=True)
    
    print(f"\n分析完成！结果已保存至: {output_file}")
    print("\n汇总结果:")
    print(summary.to_string(index=False))