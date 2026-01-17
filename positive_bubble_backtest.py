"""
正向泡沫因子（momentum_positive_bubble）专用回测模块

该因子与其他因子不同：
1. 返回0/1信号，而非连续因子值
2. 周频数据（每周五）
3. 信号=1表示持有，信号=0表示不持有

回测逻辑：
- 每周五根据信号选择持仓行业（信号=1的行业）
- 等权持有所有信号=1的行业
- 计算策略收益、基准收益、超额收益
- 输出历史持仓记录和策略结果
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 导入数据加载模块和因子模块
from wind_data_loader import (
    load_price_df, load_volume_df, DEFAULT_CACHE_FILE
)
from factor_ import momentum_positive_bubble


def clean_industry_name(name):
    """
    清理行业名称，删除"（申万）"后缀
    """
    if isinstance(name, str):
        return name.replace('（申万）', '').replace('(申万)', '')
    return name


def run_positive_bubble_backtest(prices_df, amount_df, 
                                  bsadf_min_window=62,
                                  bsadf_compare_window=40,
                                  bocd_hazard=0.01,
                                  bocd_run_length_threshold=5,
                                  regime_lookback=4,
                                  similarity_lookback=52,
                                  similarity_top_n=8,
                                  similarity_threshold=4):
    """
    运行正向泡沫因子回测
    
    参数:
        prices_df: pd.DataFrame, 日频收盘价数据 (index=日期, columns=行业)
        amount_df: pd.DataFrame, 日频成交额数据 (index=日期, columns=行业)
        其他参数: 因子计算参数
    
    返回:
        dict: 包含回测结果的字典
            - 'signal_df': 周频信号DataFrame (0/1)
            - 'holdings_history': 历史持仓记录
            - 'strategy_nav': 策略净值序列
            - 'benchmark_nav': 基准净值序列
            - 'excess_nav': 超额净值序列
            - 'performance_metrics': 绩效指标
            - 'yearly_returns': 每年收益统计
    """
    print("正在计算正向泡沫因子信号...")
    
    # 计算因子信号（周频0/1信号）
    signal_df = momentum_positive_bubble(
        prices_df, amount_df,
        bsadf_min_window=bsadf_min_window,
        bsadf_compare_window=bsadf_compare_window,
        bocd_hazard=bocd_hazard,
        bocd_run_length_threshold=bocd_run_length_threshold,
        regime_lookback=regime_lookback,
        similarity_lookback=similarity_lookback,
        similarity_top_n=similarity_top_n,
        similarity_threshold=similarity_threshold
    )
    
    print(f"信号计算完成，周频数据点数: {len(signal_df)}")
    
    # 获取周频价格数据（与信号对齐）
    prices_df_copy = prices_df.copy()
    prices_df_copy.index = pd.to_datetime(prices_df_copy.index)
    weekly_prices = prices_df_copy.resample('W-FRI').last()
    
    # 对齐信号和价格的索引
    common_dates = signal_df.index.intersection(weekly_prices.index)
    signal_df = signal_df.loc[common_dates]
    weekly_prices = weekly_prices.loc[common_dates]
    
    print(f"有效周数: {len(common_dates)}")
    
    # ========== 回测计算 ==========
    
    # 初始化净值序列
    strategy_nav = pd.Series(index=common_dates, dtype=float)
    benchmark_nav = pd.Series(index=common_dates, dtype=float)
    strategy_nav.iloc[0] = 1.0
    benchmark_nav.iloc[0] = 1.0
    
    # 历史持仓记录
    holdings_history = {}
    
    # 计算周收益率
    weekly_returns = weekly_prices.pct_change()
    
    # 逐周回测
    total_weeks = len(common_dates)
    for i in range(1, total_weeks):
        if (i + 1) % 100 == 0 or i == total_weeks - 1:
            print(f"    回测进度: {i+1}/{total_weeks} 周 ({((i+1)/total_weeks)*100:.2f}%)")

        prev_date = common_dates[i-1]
        curr_date = common_dates[i]
        
        # 获取上周的信号（用于本周持仓）
        prev_signal = signal_df.loc[prev_date]
        
        # 选择信号=1的行业
        selected_industries = prev_signal[prev_signal == 1].index.tolist()
        
        # 记录持仓
        holdings_history[prev_date] = selected_industries
        
        # 计算本周收益
        curr_returns = weekly_returns.loc[curr_date]
        
        # 策略收益：等权持有信号=1的行业
        if len(selected_industries) > 0:
            strategy_return = curr_returns[selected_industries].mean()
        else:
            strategy_return = 0  # 无持仓时收益为0
        
        # 基准收益：全市场等权
        benchmark_return = curr_returns.mean()
        
        # 更新净值
        strategy_nav.iloc[i] = strategy_nav.iloc[i-1] * (1 + strategy_return)
        benchmark_nav.iloc[i] = benchmark_nav.iloc[i-1] * (1 + benchmark_return)
    
    # 记录最后一期持仓
    last_date = common_dates[-1]
    last_signal = signal_df.loc[last_date]
    holdings_history[last_date] = last_signal[last_signal == 1].index.tolist()
    
    # 计算超额净值
    excess_nav = strategy_nav / benchmark_nav
    
    # ========== 计算绩效指标 ==========
    
    performance_metrics = calculate_performance_metrics(
        strategy_nav, benchmark_nav, excess_nav
    )
    
    # ========== 计算每年收益统计 ==========
    
    yearly_returns = calculate_yearly_returns(
        strategy_nav, benchmark_nav, start_year=2017
    )
    
    return {
        'signal_df': signal_df,
        'holdings_history': holdings_history,
        'strategy_nav': strategy_nav,
        'benchmark_nav': benchmark_nav,
        'excess_nav': excess_nav,
        'performance_metrics': performance_metrics,
        'yearly_returns': yearly_returns,
    }


def calculate_performance_metrics(strategy_nav, benchmark_nav, excess_nav):
    """
    计算绩效指标
    
    参数:
        strategy_nav: pd.Series, 策略净值
        benchmark_nav: pd.Series, 基准净值
        excess_nav: pd.Series, 超额净值
    
    返回:
        dict: 绩效指标字典
    """
    # 计算时间跨度
    start_date = strategy_nav.index[0]
    end_date = strategy_nav.index[-1]
    years = (end_date - start_date).days / 365.25
    
    # 周频，假设一年约52周
    periods_per_year = 52
    
    # 策略收益率序列
    strategy_returns = strategy_nav.pct_change().dropna()
    benchmark_returns = benchmark_nav.pct_change().dropna()
    excess_returns = excess_nav.pct_change().dropna()
    
    # ========== 策略指标 ==========
    
    # 累计收益率
    total_return = (strategy_nav.iloc[-1] / strategy_nav.iloc[0] - 1) * 100
    
    # 年化收益率
    annual_return = ((strategy_nav.iloc[-1] / strategy_nav.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # 年化波动率
    volatility = strategy_returns.std() * np.sqrt(periods_per_year) * 100
    
    # 夏普比率
    sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(periods_per_year)) if strategy_returns.std() > 0 else 0
    
    # 最大回撤
    cummax = strategy_nav.cummax()
    drawdown = (strategy_nav - cummax) / cummax
    max_drawdown = drawdown.min() * 100
    
    # Calmar比率
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # ========== 基准指标 ==========
    
    benchmark_total_return = (benchmark_nav.iloc[-1] / benchmark_nav.iloc[0] - 1) * 100
    benchmark_annual_return = ((benchmark_nav.iloc[-1] / benchmark_nav.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # ========== 超额指标 ==========
    
    # 超额累计收益率
    excess_total_return = (excess_nav.iloc[-1] / excess_nav.iloc[0] - 1) * 100
    
    # 超额年化收益率
    excess_annual_return = ((excess_nav.iloc[-1] / excess_nav.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # 超额最大回撤
    excess_cummax = excess_nav.cummax()
    excess_drawdown = (excess_nav - excess_cummax) / excess_cummax
    excess_max_drawdown = excess_drawdown.min() * 100
    
    # 信息比率 (IR)
    tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
    information_ratio = (excess_returns.mean() * periods_per_year) / tracking_error if tracking_error > 0 else 0
    
    # 胜率：策略收益 > 基准收益的周数占比
    win_count = (strategy_returns > benchmark_returns).sum()
    total_periods = len(strategy_returns)
    win_rate = (win_count / total_periods * 100) if total_periods > 0 else 0
    
    return {
        '策略累计收益率(%)': round(total_return, 2),
        '策略年化收益率(%)': round(annual_return, 2),
        '策略年化波动率(%)': round(volatility, 2),
        '策略夏普比率': round(sharpe, 2),
        '策略最大回撤(%)': round(max_drawdown, 2),
        '策略Calmar比率': round(calmar, 2),
        '基准累计收益率(%)': round(benchmark_total_return, 2),
        '基准年化收益率(%)': round(benchmark_annual_return, 2),
        '超额累计收益率(%)': round(excess_total_return, 2),
        '超额年化收益率(%)': round(excess_annual_return, 2),
        '超额最大回撤(%)': round(excess_max_drawdown, 2),
        '信息比率(IR)': round(information_ratio, 2),
        '周胜率(%)': round(win_rate, 2),
    }


def calculate_yearly_returns(strategy_nav, benchmark_nav, start_year=2017):
    """
    计算每年的收益统计
    
    参数:
        strategy_nav: pd.Series, 策略净值
        benchmark_nav: pd.Series, 基准净值
        start_year: int, 起始年份
    
    返回:
        pd.DataFrame: 每年的收益统计
    """
    # 获取所有年份
    years = sorted(set(strategy_nav.index.year))
    years = [y for y in years if y >= start_year]
    
    yearly_data = []
    
    for year in years:
        # 获取该年的数据
        year_mask = strategy_nav.index.year == year
        year_dates = strategy_nav.index[year_mask]
        
        if len(year_dates) < 2:
            continue
        
        start_date = year_dates[0]
        end_date = year_dates[-1]
        
        # 策略收益
        strategy_return = (strategy_nav.loc[end_date] / strategy_nav.loc[start_date] - 1) * 100
        
        # 基准收益
        bench_return = (benchmark_nav.loc[end_date] / benchmark_nav.loc[start_date] - 1) * 100
        
        # 超额收益
        excess_return = strategy_return - bench_return
        
        yearly_data.append({
            '年份': year,
            '策略收益(%)': round(strategy_return, 2),
            '基准收益(%)': round(bench_return, 2),
            '超额收益(%)': round(excess_return, 2),
        })
    
    # 添加全样本统计
    if len(strategy_nav) >= 2:
        total_strategy_return = (strategy_nav.iloc[-1] / strategy_nav.iloc[0] - 1) * 100
        total_bench_return = (benchmark_nav.iloc[-1] / benchmark_nav.iloc[0] - 1) * 100
        total_excess_return = total_strategy_return - total_bench_return
        
        yearly_data.append({
            '年份': '全样本',
            '策略收益(%)': round(total_strategy_return, 2),
            '基准收益(%)': round(total_bench_return, 2),
            '超额收益(%)': round(total_excess_return, 2),
        })
    
    return pd.DataFrame(yearly_data)


def create_holdings_df(holdings_history):
    """
    创建历史持仓DataFrame
    
    参数:
        holdings_history: dict, {日期: [行业列表]}
    
    返回:
        pd.DataFrame: 历史持仓记录
    """
    # 按日期降序排列
    sorted_dates = sorted(holdings_history.keys(), reverse=True)
    
    data = []
    for date in sorted_dates:
        industries = holdings_history[date]
        # 清理行业名称
        cleaned_industries = [clean_industry_name(ind) for ind in industries]
        
        data.append({
            '日期': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
            '持仓数量': len(cleaned_industries),
            '持仓行业': ', '.join(cleaned_industries) if cleaned_industries else '空仓',
        })
    
    return pd.DataFrame(data)


def create_nav_df(strategy_nav, benchmark_nav, excess_nav):
    """
    创建净值DataFrame
    
    参数:
        strategy_nav: pd.Series, 策略净值
        benchmark_nav: pd.Series, 基准净值
        excess_nav: pd.Series, 超额净值
    
    返回:
        pd.DataFrame: 净值记录
    """
    nav_df = pd.DataFrame({
        '策略净值': strategy_nav,
        '基准净值': benchmark_nav,
        '超额净值': excess_nav,
    })
    
    # 格式化日期索引
    nav_df.index = nav_df.index.strftime('%Y-%m-%d')
    nav_df.index.name = '日期'
    
    # 保留4位小数
    nav_df = nav_df.round(4)
    
    return nav_df


def export_to_excel(backtest_result, writer, sheet_name='正向泡沫因子回测'):
    """
    将回测结果导出到Excel
    
    参数:
        backtest_result: dict, 回测结果
        writer: pd.ExcelWriter, Excel写入器对象
        sheet_name: str, 要写入的sheet名称
    """
    print(f"\n正在导出回测结果到: {writer.path} 的 [{sheet_name}] sheet页")
    
    # ========== Sheet 1: 策略概览 ==========
    start_row = 0
    
    # 因子说明
    factor_doc = """
正向泡沫行业轮动因子（momentum_positive_bubble）

出处：兴业证券《如何结合行业轮动的长短信号？》

理念：利用"理性泡沫"理论，通过 BSADF（量价爆炸检测）和 BOCD（结构突变检测）
      捕捉行业的起涨点，并结合市场环境进行风控。

构造：
    1. 数据预处理：日频转周频（W-FRI）
    2. BSADF信号：价格和成交额的泡沫检测，双重确认
    3. BOCD信号：收益率变点检测，捕捉趋势起点
    4. 市场环境：根据牛熊市状态调整信号
    5. 信号合成：牛市扩散、熊市清仓、震荡维持

特点：
    - 返回0/1信号，而非连续因子值
    - 周频调仓（每周五）
    - 信号=1表示持有，信号=0表示不持有
"""
    doc_df = pd.DataFrame({'【因子说明】': [factor_doc]})
    doc_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
    start_row += 20
    
    # 绩效指标
    metrics = backtest_result['performance_metrics']
    metrics_df = pd.DataFrame([metrics]).T
    metrics_df.columns = ['数值']
    metrics_df.index.name = '指标'
    
    header_df = pd.DataFrame({'【绩效指标】': ['']})
    header_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
    start_row += 1
    metrics_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=True)
    start_row += len(metrics_df) + 3
    
    # 每年收益统计
    yearly_df = backtest_result['yearly_returns']
    if not yearly_df.empty:
        header_df = pd.DataFrame({'【每年收益统计】': ['']})
        header_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
        start_row += 1
        yearly_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
        start_row += len(yearly_df) + 3
    
    # ========== Sheet 2: 净值序列 ==========
    nav_df = create_nav_df(
        backtest_result['strategy_nav'],
        backtest_result['benchmark_nav'],
        backtest_result['excess_nav']
    )
    nav_df.to_excel(writer, sheet_name='净值序列', index=True)
    
    # ========== Sheet 3: 历史持仓 ==========
    holdings_df = create_holdings_df(backtest_result['holdings_history'])
    holdings_df.to_excel(writer, sheet_name='历史持仓', index=False)
    
    # ========== Sheet 4: 原始信号 ==========
    signal_df = backtest_result['signal_df'].copy()
    # 清理列名
    signal_df.columns = [clean_industry_name(col) for col in signal_df.columns]
    # 格式化日期索引
    signal_df.index = signal_df.index.strftime('%Y-%m-%d')
    signal_df.index.name = '日期'
    signal_df.to_excel(writer, sheet_name='原始信号', index=True)
    
    print(f"导出完成！")


def main():
    """
    主函数
    """
    print("=" * 60)
    print("正向泡沫因子（momentum_positive_bubble）回测")
    print("=" * 60)
    
    # 加载数据
    print("\n正在加载数据...")
    prices_df = load_price_df(DEFAULT_CACHE_FILE)
    amount_df = load_volume_df(DEFAULT_CACHE_FILE)  # 使用成交量作为成交额代理
    
    print(f"价格数据: {prices_df.shape}")
    print(f"成交额数据: {amount_df.shape}")
    print(f"日期范围: {prices_df.index[0].date()} 至 {prices_df.index[-1].date()}")
    
    # 运行回测
    print("\n开始回测...")
    backtest_result = run_positive_bubble_backtest(
        prices_df, amount_df,
        bsadf_min_window=62,
        bsadf_compare_window=40,
        bocd_hazard=0.01,
        bocd_run_length_threshold=5,
        regime_lookback=4,
        similarity_lookback=52,
        similarity_top_n=8,
        similarity_threshold=4
    )
    
    # 打印绩效指标
    print("\n" + "=" * 60)
    print("绩效指标")
    print("=" * 60)
    for key, value in backtest_result['performance_metrics'].items():
        print(f"  {key}: {value}")
    
    # 打印每年收益
    print("\n" + "=" * 60)
    print("每年收益统计")
    print("=" * 60)
    print(backtest_result['yearly_returns'].to_string(index=False))
    
    # 导出到Excel
    output_file = 'factors_analysis_report.xlsx'
    sheet_name = '正向泡沫因子回测'
    
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        export_to_excel(backtest_result, writer, sheet_name)
    
    print("\n" + "=" * 60)
    print(f"回测完成！结果已保存至: {output_file} 的 [{sheet_name}] sheet页")
    print("=" * 60)


if __name__ == "__main__":
    main()