"""
因子分析模块
负责批量因子分析、IC/IR 计算、分层回测、统计指标计算
输出到Excel，每个因子一个sheet页，包含因子说明

支持多周期回测：20日、60日、120日、240日
超额收益基准：申万一级行业等权收益
"""
import pandas as pd
import numpy as np
import inspect

# 导入数据加载模块和因子模块
import wind_data_loader
from wind_data_loader import (
    load_price_df, load_high_df, load_low_df,
    load_turnover_df, load_volume_df, DEFAULT_CACHE_FILE
)
import factor_

# 默认参数
DEFAULT_REBALANCE_FREQ = 20   # 调仓频率/预测窗口 (天)
N_LAYERS = 5                  # 分层回测层数
WINDOWS = [20, 60, 120, 240]  # 多周期回测窗口

# 因子注册表（模块级别，方便扩展）
# 新增因子只需在此添加一行
FACTOR_REGISTRY = {
    'momentum': factor_.momentum,  # 传统动量因子
    'momentum_sharpe': factor_.momentum_sharpe,  # 夏普动量因子
    'momentum_std': factor_.momentum_std,  # 标准化动量因子（Z-score，不排名）
    'momentum_rank_std': factor_.momentum_rank_std,  # Rank标准化动量因子
    'momentum_turnover_adj': factor_.momentum_turnover_adj,  # 换手率调整动量因子
    'momentum_calmar_ratio': factor_.momentum_calmar_ratio,  # Calmar比率因子
    'momentum_pure_liquidity_stripped': factor_.momentum_pure_liquidity_stripped,  # 剥离流动性提纯动量因子
    'momentum_cross_industry_lasso': factor_.momentum_cross_industry_lasso,  # Lasso因子（计算较慢，按需启用）
    #'momentum_positive_bubble': factor_.momentum_positive_bubble,  # 正向泡沫行业轮动因子
}

class DataContainer:
    """
    数据容器类，统一管理所有数据
    一次性加载所有数据，避免重复读取文件
    """
    def __init__(self, file_path=DEFAULT_CACHE_FILE, start_date=None, end_date=None):
        """
        初始化数据容器，加载所有数据
        
        参数:
        file_path: str, 数据文件路径
        start_date: str, 数据开始日期 (YYYY-MM-DD)
        end_date: str, 数据结束日期 (YYYY-MM-DD)
        """
        print("正在加载所有数据...")
        self.file_path = file_path
        
        # 加载所有数据
        self.prices_df = load_price_df(file_path)
        self.high_df = load_high_df(file_path)
        self.low_df = load_low_df(file_path)
        self.turnover_df = load_turnover_df(file_path)
        self.volume_df = load_volume_df(file_path)
        
        # amount_df 使用 volume_df 作为代理（成交量可近似代表成交额趋势）
        self.amount_df = self.volume_df

        # 根据日期范围筛选数据
        if start_date:
            self.prices_df = self.prices_df[self.prices_df.index >= start_date]
            self.high_df = self.high_df[self.high_df.index >= start_date]
            self.low_df = self.low_df[self.low_df.index >= start_date]
            self.turnover_df = self.turnover_df[self.turnover_df.index >= start_date]
            self.volume_df = self.volume_df[self.volume_df.index >= start_date]
            self.amount_df = self.amount_df[self.amount_df.index >= start_date]
        if end_date:
            self.prices_df = self.prices_df[self.prices_df.index <= end_date]
            self.high_df = self.high_df[self.high_df.index <= end_date]
            self.low_df = self.low_df[self.low_df.index <= end_date]
            self.turnover_df = self.turnover_df[self.turnover_df.index <= end_date]
            self.volume_df = self.volume_df[self.volume_df.index <= end_date]
            self.amount_df = self.amount_df[self.amount_df.index <= end_date]
        
        print(f"数据加载完成:")
        print(f"  - 价格数据: {self.prices_df.shape}")
        print(f"  - 最高价数据: {self.high_df.shape}")
        print(f"  - 最低价数据: {self.low_df.shape}")
        print(f"  - 换手率数据: {self.turnover_df.shape}")
        print(f"  - 成交量数据: {self.volume_df.shape}")
        print(f"  - 日期范围: {self.prices_df.index[0].date()} 至 {self.prices_df.index[-1].date()}")


def get_factor_docstring(factor_name):
    """
    获取因子函数的docstring作为因子说明
    
    参数:
    factor_name: str, 因子名称
    
    返回:
    str: 因子说明文档
    """
    if factor_name not in FACTOR_REGISTRY:
        return "未找到因子说明"
    
    func = FACTOR_REGISTRY[factor_name]
    docstring = inspect.getdoc(func)
    return docstring if docstring else "无说明文档"


def compute_factor(factor_name, data: DataContainer, window, rebalance_freq=DEFAULT_REBALANCE_FREQ):
    """
    计算单个因子值
    
    参数:
    factor_name: str, 因子名称
    data: DataContainer, 数据容器（包含所有需要的数据）
    window: int, 回溯窗口
    rebalance_freq: int, 调仓频率
    
    返回:
    pd.DataFrame: 因子值
    """
    if factor_name not in FACTOR_REGISTRY:
        raise ValueError(f"未知因子名称: {factor_name}. 可选: {list(FACTOR_REGISTRY.keys())}")
    
    factor_func = FACTOR_REGISTRY[factor_name]
    sig = inspect.signature(factor_func)
    param_names = list(sig.parameters.keys())
    
    # 构建参数字典 - 包含所有可能需要的参数
    available_params = {
        # 数据参数
        'prices_df': data.prices_df,
        'high_df': data.high_df,
        'low_df': data.low_df,
        'turnover_df': data.turnover_df,
        'volume_df': data.volume_df,
        'amount_df': data.amount_df,  # 使用volume_df作为代理
        # 窗口参数（所有因子统一使用 window）
        'window': window,
        'rebalance_freq': rebalance_freq,
        # 其他参数
        'zscore_window': 250,
        'smooth_window': 3,
        'min_industries': 15,
        'train_periods': None,
        'benchmark_returns': None,
    }
    
    # 根据函数签名自动选择参数
    call_kwargs = {}
    for param_name in param_names:
        if param_name in available_params:
            call_kwargs[param_name] = available_params[param_name]
        else:
            param = sig.parameters[param_name]
            if param.default is inspect.Parameter.empty:
                raise ValueError(f"因子 '{factor_name}' 需要参数 '{param_name}'，但未定义。")
    
    return factor_func(**call_kwargs)


def calculate_benchmark_returns(prices_df, rebalance_freq):
    """
    计算基准收益率（申万一级行业等权）
    
    参数:
    prices_df: pd.DataFrame, 价格数据
    rebalance_freq: int, 调仓频率
    
    返回:
    pd.Series: 基准收益率序列（每期）
    """
    # 计算每个行业的期收益率
    period_returns = prices_df.pct_change(rebalance_freq)
    # 等权平均作为基准
    benchmark_returns = period_returns.mean(axis=1)
    return benchmark_returns


def calculate_ic_ir(factor_df, forward_returns_df, window, rebalance_freq=DEFAULT_REBALANCE_FREQ, n_layers=N_LAYERS):
    """
    计算因子的 IC 和 IR
    
    返回:
    tuple: (ic_series, ic_cumsum, ic_mean, ic_std, icir)
    """
    valid_dates = factor_df.index[window::rebalance_freq]
    valid_dates = [d for d in valid_dates if d in forward_returns_df.index]
    
    ic_series = []
    ic_dates = []
    for date in valid_dates:
        fac = factor_df.loc[date]
        ret = forward_returns_df.loc[date]
        
        valid_mask = ~(fac.isna() | ret.isna())
        fac = fac[valid_mask]
        ret = ret[valid_mask]
        
        if len(fac) < n_layers * 2:
            continue
        
        ic = fac.corr(ret, method='spearman')
        ic_series.append(ic)
        ic_dates.append(date)
    
    ic_series = pd.Series(ic_series, index=ic_dates)
    ic_cumsum = ic_series.cumsum()  # IC累积序列
    ic_mean = ic_series.mean() if len(ic_series) > 0 else 0
    ic_std = ic_series.std() if len(ic_series) > 0 else 0
    icir = ic_mean / ic_std if ic_std != 0 else 0
    
    return ic_series, ic_cumsum, ic_mean, ic_std, icir


def stratified_backtest(factor_df, prices_df, window, rebalance_freq=DEFAULT_REBALANCE_FREQ, n_layers=N_LAYERS):
    """
    分层回测：根据因子值将资产分成n层，计算每层的收益率累计净值
    同时返回每期各层选中的行业
    
    返回:
    tuple: (nav_df, layer_returns, layer_holdings_history)
    """
    daily_returns = prices_df.pct_change()
    
    valid_dates = factor_df.index[window::rebalance_freq]
    valid_dates = [d for d in valid_dates if d in prices_df.index]
    
    layer_nav = {i: [1.0] for i in range(n_layers)}
    nav_dates = [valid_dates[0]] if valid_dates else []
    layer_holdings_history = {i: {} for i in range(n_layers)}  # 记录每期每层持仓
    
    for i, date in enumerate(valid_dates[:-1]):
        next_date = valid_dates[i + 1]
        
        fac = factor_df.loc[date]
        valid_mask = ~fac.isna()
        fac = fac[valid_mask]
        
        if len(fac) < n_layers * 2:
            for layer in range(n_layers):
                layer_nav[layer].append(layer_nav[layer][-1])
            nav_dates.append(next_date)
            continue
        
        # 按因子值排序并分层（升序：G1=因子值最小，G5=因子值最大）
        sorted_assets = fac.sort_values(ascending=True)
        n_assets = len(sorted_assets)
        layer_size = n_assets // n_layers
        
        holding_period = daily_returns.loc[date:next_date].iloc[1:]
        for layer in range(n_layers):
            start_idx = layer * layer_size
            end_idx = start_idx + layer_size if layer < n_layers - 1 else n_assets
            layer_assets = sorted_assets.index[start_idx:end_idx].tolist()
            
            # 记录持仓
            layer_holdings_history[layer][date] = layer_assets
            
            if layer_assets and len(holding_period) > 0:
                layer_ret = holding_period[layer_assets].mean(axis=1)
                cumulative_ret = (1 + layer_ret).prod() - 1
                layer_nav[layer].append(layer_nav[layer][-1] * (1 + cumulative_ret))
            else:
                layer_nav[layer].append(layer_nav[layer][-1])
        
        nav_dates.append(next_date)
    
    nav_df = pd.DataFrame(layer_nav, index=nav_dates)
    nav_df.columns = [f'G{i+1}' for i in range(n_layers)]
    layer_returns = nav_df.pct_change().dropna()
    
    return nav_df, layer_returns, layer_holdings_history


def calculate_excess_metrics(nav_df, benchmark_nav, rebalance_freq=DEFAULT_REBALANCE_FREQ):
    """
    计算超额收益指标
    
    参数:
    nav_df: pd.DataFrame, 各层净值
    benchmark_nav: pd.Series, 基准净值
    rebalance_freq: int, 调仓频率
    
    返回:
    dict: 各层的超额收益指标
    """
    results = {}
    
    start_date = nav_df.index[0]
    end_date = nav_df.index[-1]
    years = (end_date - start_date).days / 365.25
    periods_per_year = 252 / rebalance_freq
    
    # 计算基准收益率序列
    benchmark_returns = benchmark_nav.pct_change().dropna()
    
    for col in nav_df.columns:
        nav = nav_df[col]
        
        # 超额净值 = 策略净值 / 基准净值
        excess_nav = nav / benchmark_nav
        excess_returns = excess_nav.pct_change().dropna()
        
        # 超额累计收益率
        excess_total_return = (excess_nav.iloc[-1] / excess_nav.iloc[0] - 1) * 100
        
        # 超额年化收益率
        excess_total_ratio = excess_nav.iloc[-1] / excess_nav.iloc[0]
        excess_annual_return = (excess_total_ratio ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # 绝对收益指标
        total_return = (nav.iloc[-1] / nav.iloc[0] - 1) * 100
        total_ratio = nav.iloc[-1] / nav.iloc[0]
        annual_return = (total_ratio ** (1 / years) - 1) * 100 if years > 0 else 0
        
        returns = nav.pct_change().dropna()
        volatility = returns.std() * np.sqrt(periods_per_year) * 100
        
        # 夏普比率
        sharpe = (returns.mean() / returns.std() * np.sqrt(periods_per_year)) if returns.std() > 0 else 0
        
        # 最大回撤
        cummax = nav.cummax()
        drawdown = (nav - cummax) / cummax
        max_dd = drawdown.min() * 100
        
        # 多头胜率：策略收益 > 基准收益的期数占比
        aligned_returns = returns.align(benchmark_returns, join='inner')
        strategy_ret = aligned_returns[0]
        bench_ret = aligned_returns[1]
        win_count = (strategy_ret > bench_ret).sum()
        total_periods = len(strategy_ret)
        win_rate = (win_count / total_periods * 100) if total_periods > 0 else 0
        
        results[col] = {
            '累计收益率(%)': total_return,
            '年化收益率(%)': annual_return,
            '超额累计收益率(%)': excess_total_return,
            '超额年化收益率(%)': excess_annual_return,
            '年化波动率(%)': volatility,
            '夏普比率': sharpe,
            '最大回撤(%)': max_dd,
            '多头胜率(%)': win_rate,
        }
    
    return results


def calculate_yearly_returns(nav_df, benchmark_nav, start_year=2017):
    """
    计算每年的收益统计（仅针对G5）
    
    参数:
    nav_df: pd.DataFrame, 各层净值（包含G5列）
    benchmark_nav: pd.Series, 基准净值
    start_year: int, 起始年份
    
    返回:
    pd.DataFrame: 每年的多头收益、超额收益、基准收益
    """
    if 'G5' not in nav_df.columns:
        return pd.DataFrame()
    
    g5_nav = nav_df['G5']
    
    # 获取所有年份
    years = sorted(set(nav_df.index.year))
    years = [y for y in years if y >= start_year]
    
    yearly_data = []
    
    for year in years:
        # 获取该年的数据
        year_mask = nav_df.index.year == year
        year_dates = nav_df.index[year_mask]
        
        if len(year_dates) < 2:
            continue
        
        start_date = year_dates[0]
        end_date = year_dates[-1]
        
        # G5多头收益
        g5_return = (g5_nav.loc[end_date] / g5_nav.loc[start_date] - 1) * 100
        
        # 基准收益
        bench_return = (benchmark_nav.loc[end_date] / benchmark_nav.loc[start_date] - 1) * 100
        
        # 超额收益
        excess_return = g5_return - bench_return
        
        yearly_data.append({
            '年份': year,
            'G5多头收益(%)': round(g5_return, 2),
            '超额收益(%)': round(excess_return, 2),
            '基准收益(%)': round(bench_return, 2),
        })
    
    # 添加全样本统计
    if len(nav_df) >= 2:
        total_g5_return = (g5_nav.iloc[-1] / g5_nav.iloc[0] - 1) * 100
        total_bench_return = (benchmark_nav.iloc[-1] / benchmark_nav.iloc[0] - 1) * 100
        total_excess_return = total_g5_return - total_bench_return
        
        yearly_data.append({
            '年份': '全样本',
            'G5多头收益(%)': round(total_g5_return, 2),
            '超额收益(%)': round(total_excess_return, 2),
            '基准收益(%)': round(total_bench_return, 2),
        })
    
    return pd.DataFrame(yearly_data)


def analyze_single_factor_window(factor_name, data: DataContainer, window, rebalance_freq=DEFAULT_REBALANCE_FREQ):
    """
    分析单个因子在单个窗口下的表现
    
    返回:
    dict: 包含IC/IR、分层指标、持仓等信息
    """
    # 计算因子值
    factor_df = compute_factor(factor_name, data, window, rebalance_freq)
    
    # 计算未来收益率
    forward_returns_df = data.prices_df.pct_change(rebalance_freq).shift(-rebalance_freq)
    
    # 计算IC/IR（包含IC累积序列）
    ic_series, ic_cumsum, ic_mean, ic_std, icir = calculate_ic_ir(factor_df, forward_returns_df, window, rebalance_freq)
    
    # 分层回测
    nav_df, layer_returns, layer_holdings = stratified_backtest(factor_df, data.prices_df, window, rebalance_freq)
    
    # 计算基准净值
    benchmark_returns = calculate_benchmark_returns(data.prices_df, rebalance_freq)
    # 对齐到调仓日期
    benchmark_nav = pd.Series(index=nav_df.index, dtype=float)
    benchmark_nav.iloc[0] = 1.0
    for i in range(1, len(nav_df.index)):
        prev_date = nav_df.index[i-1]
        curr_date = nav_df.index[i]
        # 计算期间基准收益
        period_ret = (data.prices_df.loc[curr_date] / data.prices_df.loc[prev_date] - 1).mean()
        benchmark_nav.iloc[i] = benchmark_nav.iloc[i-1] * (1 + period_ret)
    
    # 计算超额指标
    excess_metrics = calculate_excess_metrics(nav_df, benchmark_nav, rebalance_freq)
    
    # 计算每年收益统计（仅针对G5）
    yearly_returns = calculate_yearly_returns(nav_df, benchmark_nav, start_year=2017)
    
    return {
        'ic_mean': ic_mean,
        'icir': icir,
        'ic_series': ic_series,
        'ic_cumsum': ic_cumsum,
        'nav_df': nav_df,
        'benchmark_nav': benchmark_nav,
        'excess_metrics': excess_metrics,
        'layer_holdings': layer_holdings,
        'yearly_returns': yearly_returns,
    }


def analyze_all_factors(data: DataContainer, windows=WINDOWS, rebalance_freq=DEFAULT_REBALANCE_FREQ):
    """
    分析所有因子在所有窗口下的表现
    
    参数:
    data: DataContainer, 数据容器
    windows: list, 窗口列表
    rebalance_freq: int, 调仓频率
    
    返回:
    dict: {factor_name: {window: analysis_result}}
    """
    all_results = {}
    
    for factor_name in FACTOR_REGISTRY.keys():
        print(f"\n正在分析因子: {factor_name}")
        factor_results = {}
        
        for window in windows:
            print(f"  窗口: {window}日...")
            try:
                result = analyze_single_factor_window(
                    factor_name, data, window, rebalance_freq
                )
                factor_results[window] = result
            except Exception as e:
                print(f"    错误: {e}")
                factor_results[window] = None
        
        all_results[factor_name] = factor_results
    
    return all_results


def create_factor_summary_df(factor_name, factor_results, windows=WINDOWS):
    """
    创建单个因子的汇总DataFrame
    
    格式：
    - 列名为窗口周期（20, 60, 120, 240）
    - 行名为指标名称
    - G5放在最前面，然后是G4, G3, G2, G1
    - IC和ICIR保留4位小数，其他指标保留2位小数
    
    返回:
    pd.DataFrame: 汇总表格（行为指标，列为窗口）
    """
    # 构建数据字典：{窗口: {指标: 值}}
    data_dict = {}
    
    for window in windows:
        result = factor_results.get(window)
        if result is None:
            continue
        
        window_data = {}
        # IC和ICIR保留4位小数
        window_data['IC均值'] = round(result['ic_mean'], 4)
        window_data['ICIR'] = round(result['icir'], 4)
        
        # 按G5, G4, G3, G2, G1顺序添加各层指标
        layer_order = ['G5', 'G4', 'G3', 'G2', 'G1']
        for layer_name in layer_order:
            if layer_name in result['excess_metrics']:
                metrics = result['excess_metrics'][layer_name]
                for metric_name, value in metrics.items():
                    row_name = f'{layer_name}_{metric_name}'
                    window_data[row_name] = round(value, 2)
        
        data_dict[window] = window_data
    
    # 转换为DataFrame并转置（行为指标，列为窗口）
    df = pd.DataFrame(data_dict)
    
    # 确保列按窗口顺序排列
    df = df.reindex(columns=windows)
    
    return df


def clean_industry_name(name):
    """
    清理行业名称，删除"（申万）"后缀
    
    参数:
    name: str, 行业名称
    
    返回:
    str: 清理后的行业名称
    """
    if isinstance(name, str):
        return name.replace('（申万）', '').replace('(申万)', '')
    return name


def format_date_index(df):
    """
    格式化DataFrame的日期索引，只保留日期部分（不含时间）
    
    参数:
    df: pd.DataFrame, 带有日期索引的DataFrame
    
    返回:
    pd.DataFrame: 格式化后的DataFrame
    """
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.strftime('%Y-%m-%d')
    return df


def create_g5_holdings_df(factor_results, windows=WINDOWS):
    """
    创建G5持仓记录DataFrame（按列输出每个窗口的G5持仓）
    
    格式：
    - 列名为窗口周期（20, 60, 120, 240）
    - 行为日期（降序，最新日期在最前面）
    - 值为持仓行业（逗号分隔，不含"（申万）"后缀）
    
    参数:
    factor_results: dict, 因子分析结果 {window: result}
    windows: list, 窗口列表
    
    返回:
    pd.DataFrame: G5持仓记录（行为日期，列为窗口）
    """
    # 收集所有窗口的G5持仓
    holdings_dict = {}
    all_dates = set()
    
    for window in windows:
        result = factor_results.get(window)
        if result is None or result.get('layer_holdings') is None:
            continue
        
        # G5 是 layer_idx=4
        g5_holdings = result['layer_holdings'].get(4, {})
        
        # 转换为 {日期: 持仓行业字符串}，同时清理行业名称
        window_holdings = {}
        for date, industries in g5_holdings.items():
            # 清理行业名称，删除"（申万）"后缀
            cleaned_industries = [clean_industry_name(ind) for ind in industries] if industries else []
            window_holdings[date] = ', '.join(cleaned_industries) if cleaned_industries else ''
            all_dates.add(date)
        
        holdings_dict[window] = window_holdings
    
    if not all_dates:
        return pd.DataFrame()
    
    # 按日期降序排列（最新日期在最前面）
    sorted_dates = sorted(all_dates, reverse=True)
    
    # 构建DataFrame
    data = {}
    for window in windows:
        if window in holdings_dict:
            data[window] = [holdings_dict[window].get(date, '') for date in sorted_dates]
        else:
            data[window] = [''] * len(sorted_dates)
    
    # 格式化日期索引（只保留日期部分）
    formatted_dates = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in sorted_dates]
    
    df = pd.DataFrame(data, index=formatted_dates)
    df.index.name = '日期'
    
    return df


def create_ic_cumsum_df(factor_results, windows=WINDOWS):
    """
    创建IC累积序列DataFrame
    
    参数:
    factor_results: dict, 因子分析结果 {window: result}
    windows: list, 窗口列表
    
    返回:
    pd.DataFrame: IC累积序列（行为日期，列为窗口）
    """
    ic_cumsum_dict = {}
    all_dates = set()
    
    for window in windows:
        result = factor_results.get(window)
        if result is None or result.get('ic_cumsum') is None:
            continue
        
        ic_cumsum = result['ic_cumsum']
        ic_cumsum_dict[window] = ic_cumsum
        all_dates.update(ic_cumsum.index)
    
    if not all_dates:
        return pd.DataFrame()
    
    # 按日期升序排列
    sorted_dates = sorted(all_dates)
    
    # 构建DataFrame
    data = {}
    for window in windows:
        if window in ic_cumsum_dict:
            data[f'{window}日'] = [ic_cumsum_dict[window].get(date, np.nan) for date in sorted_dates]
        else:
            data[f'{window}日'] = [np.nan] * len(sorted_dates)
    
    # 格式化日期索引（只保留日期部分）
    formatted_dates = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in sorted_dates]
    
    df = pd.DataFrame(data, index=formatted_dates)
    df.index.name = '日期'
    
    # 保留四位小数
    df = df.round(4)
    
    return df


def create_layer_nav_df(factor_results, windows=WINDOWS):
    """
    创建分层累积净值DataFrame（每个窗口的G1-G5净值）
    
    参数:
    factor_results: dict, 因子分析结果 {window: result}
    windows: list, 窗口列表
    
    返回:
    dict: {window: nav_df} 每个窗口的分层净值DataFrame
    """
    layer_nav_dict = {}
    
    for window in windows:
        result = factor_results.get(window)
        if result is None or result.get('nav_df') is None:
            continue
        
        nav_df = result['nav_df'].copy()
        # 添加基准净值列
        if result.get('benchmark_nav') is not None:
            nav_df['基准'] = result['benchmark_nav']
        
        # 保留四位小数
        nav_df = nav_df.round(4)
        
        # 格式化日期索引（只保留日期部分）
        nav_df = format_date_index(nav_df)
        
        layer_nav_dict[window] = nav_df
    
    return layer_nav_dict


def create_g5_yearly_returns_df(factor_results, windows=WINDOWS):
    """
    创建G5每年收益统计DataFrame（仅针对指定窗口，如G5对应的窗口）
    
    参数:
    factor_results: dict, 因子分析结果 {window: result}
    windows: list, 窗口列表
    
    返回:
    dict: {window: yearly_returns_df}
    """
    yearly_dict = {}
    
    for window in windows:
        result = factor_results.get(window)
        if result is None or result.get('yearly_returns') is None:
            continue
        
        yearly_df = result['yearly_returns']
        if not yearly_df.empty:
            yearly_dict[window] = yearly_df
    
    return yearly_dict


def get_data_date_range(factor_results, windows=WINDOWS):
    """
    获取数据日期范围
    
    参数:
    factor_results: dict, 因子分析结果 {window: result}
    windows: list, 窗口列表
    
    返回:
    tuple: (start_date_str, end_date_str) 日期范围字符串
    """
    all_dates = []
    
    for window in windows:
        result = factor_results.get(window)
        if result is None or result.get('nav_df') is None:
            continue
        
        nav_df = result['nav_df']
        if len(nav_df) > 0:
            all_dates.extend(nav_df.index.tolist())
    
    if not all_dates:
        return None, None
    
    start_date = min(all_dates)
    end_date = max(all_dates)
    
    # 格式化日期
    start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
    end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
    
    return start_str, end_str


def export_to_excel(all_results, output_file='factors_analysis_report.xlsx', windows=WINDOWS):
    """
    将所有因子分析结果导出到Excel
    每个因子一个sheet页
    
    参数:
    all_results: dict, 所有因子的分析结果
    output_file: str, 输出文件名
    windows: list, 窗口列表
    """
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for factor_name, factor_results in all_results.items():
            print(f"正在导出因子: {factor_name}")
            
            # 获取因子说明
            docstring = get_factor_docstring(factor_name)
            
            # 获取数据日期范围
            start_date, end_date = get_data_date_range(factor_results, windows)
            
            # 创建汇总表
            summary_df = create_factor_summary_df(factor_name, factor_results, windows)
            
            # 创建G5持仓记录（按列输出每个窗口）
            g5_holdings_df = create_g5_holdings_df(factor_results, windows)
            
            # 创建IC累积序列
            ic_cumsum_df = create_ic_cumsum_df(factor_results, windows)
            
            # 创建分层累积净值
            layer_nav_dict = create_layer_nav_df(factor_results, windows)
            
            # 创建G5每年收益统计
            g5_yearly_dict = create_g5_yearly_returns_df(factor_results, windows)
            
            # 写入sheet
            sheet_name = factor_name[:31]  # Excel sheet名最长31字符
            
            # 写入因子说明标题
            start_row = 0
            title_df = pd.DataFrame({f'【因子说明】': [docstring]})
            title_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
            start_row += 3
            
            # 写入数据日期范围
            if start_date and end_date:
                date_range_df = pd.DataFrame({f'【数据日期范围】': [f'{start_date} 至 {end_date}']})
                date_range_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
                start_row += 3
            else:
                start_row += 1
            
            # 写入汇总表（行为指标，列为窗口，需要写入index）
            if not summary_df.empty:
                # 写入汇总表标题
                header_df = pd.DataFrame({f'【因子汇总指标】': ['']})
                header_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
                start_row += 1
                summary_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=True)
                start_row += len(summary_df) + 3
            
            # 写入G5每年收益统计（仅针对G5，各窗口）
            for window in windows:
                if window in g5_yearly_dict:
                    yearly_df = g5_yearly_dict[window]
                    # 写入标题行
                    header_df = pd.DataFrame({f'【G5每年收益统计 - {window}日窗口】': ['']})
                    header_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
                    start_row += 1
                    yearly_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
                    start_row += len(yearly_df) + 3
            
            # 写入IC累积序列
            if not ic_cumsum_df.empty:
                # 写入标题行
                header_df = pd.DataFrame({f'【IC累积序列】': ['']})
                header_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
                start_row += 1
                ic_cumsum_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=True)
                start_row += len(ic_cumsum_df) + 3
            
            # 写入分层累积净值（每个窗口单独输出）
            for window in windows:
                if window in layer_nav_dict:
                    nav_df = layer_nav_dict[window]
                    # 写入标题行
                    header_df = pd.DataFrame({f'【分层累积净值 - {window}日窗口】': ['']})
                    header_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
                    start_row += 1
                    nav_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=True)
                    start_row += len(nav_df) + 3
            
            # 写入G5持仓记录（按列输出每个窗口，日期降序）
            if not g5_holdings_df.empty:
                # 写入标题行
                header_df = pd.DataFrame({f'【G5持仓行业 - 各窗口】': ['']})
                header_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
                start_row += 1
                g5_holdings_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=True)
                start_row += len(g5_holdings_df) + 3
    
    print(f"\n分析报告已导出到: {output_file}")


def list_factors():
    """列出所有可用因子"""
    return list(FACTOR_REGISTRY.keys())


if __name__ == "__main__":
    print("=" * 60)
    print("因子批量分析")
    print("=" * 60)
    
    # 配置参数
    REBALANCE_FREQ = 20  # 调仓频率（天）
    WINDOWS_TO_TEST = [20, 60, 120, 240]  # 测试窗口
    OUTPUT_FILE = 'factors_analysis_report.xlsx'

    # 固定日期范围
    start_date = "2020-01-01"
    end_date = None # 最新日期
    
    # 加载所有数据（一次性加载）
    print(f"\n正在加载 {start_date} 至 最新 日期的数据...")
    data = DataContainer(DEFAULT_CACHE_FILE, start_date=start_date, end_date=end_date)
    
    # 显示可用因子
    print(f"\n可用因子列表: {list_factors()}")
    
    # 分析所有因子
    print("\n开始分析所有因子...")
    all_results = analyze_all_factors(
        data,
        windows=WINDOWS_TO_TEST,
        rebalance_freq=REBALANCE_FREQ
    )
    
    # 导出到Excel
    print("\n正在导出到Excel...")
    export_to_excel(all_results, OUTPUT_FILE, WINDOWS_TO_TEST)
    
    print("\n分析完成！")
