# 因子构建模块
# 新增因子注册在factor_analysis.py中
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoLarsIC
from scipy import stats
import warnings
import time
import sys


class ProgressBar:
    """
    进度条工具类，支持同一行刷新显示进度和时间

    用法:
        progress = ProgressBar(total=1000, desc='计算中')
        for i in range(1000):
            # 做一些计算
            progress.update(i + 1)
        progress.finish()
    """
    def __init__(self, total, desc='进度', update_interval=30.0):
        """
        初始化进度条

        参数:
            total: int, 总任务数
            desc: str, 描述文字
            update_interval: float, 更新间隔（秒），默认30秒
        """
        self.total = total
        self.desc = desc
        self.update_interval = update_interval
        self.start_time = time.time()
        self.last_update_time = 0
        self.current = 0

        # 在Windows上启用ANSI转义序列支持
        if sys.platform == 'win32':
            import os
            os.system('')  # 启用Windows终端的ANSI支持

    def update(self, current):
        """更新进度"""
        self.current = current
        now = time.time()

        # 控制更新频率
        if now - self.last_update_time < self.update_interval and current < self.total:
            return

        self.last_update_time = now
        elapsed = now - self.start_time

        # 计算进度百分比
        pct = current / self.total * 100

        # 计算预估剩余时间
        if current > 0:
            eta = elapsed / current * (self.total - current)
            eta_str = self._format_time(eta)
        else:
            eta_str = '--:--'

        elapsed_str = self._format_time(elapsed)

        # 构建消息
        msg = f'  {self.desc}: {current}/{self.total} ({pct:.1f}%) | 已用: {elapsed_str} | 剩余: {eta_str}'

        # 使用ANSI转义序列清除当前行并输出
        # \033[2K 清除整行, \r 回到行首
        sys.stdout.write('\033[2K\r' + msg)
        sys.stdout.flush()

    def finish(self):
        """完成进度条"""
        elapsed = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed)
        msg = f'  {self.desc}: {self.total}/{self.total} (100.0%) | 总用时: {elapsed_str}'
        # 清除行并输出完成消息，然后换行
        sys.stdout.write('\033[2K\r' + msg + '\n')
        sys.stdout.flush()

    def _format_time(self, seconds):
        """格式化时间"""
        if seconds < 60:
            return f'{seconds:.0f}秒'
        elif seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f'{m}分{s}秒'
        else:
            h, remainder = divmod(int(seconds), 3600)
            m, s = divmod(remainder, 60)
            return f'{h}时{m}分'


def momentum(prices_df, window):
    """
    传统动量因子（区间收益率）
    
    理念：基于价格惯性效应，过去表现好的资产未来倾向于继续表现好。
    构造：Momentum = P_t / P_{t-window} - 1
    
    参数:
        prices_df: pd.DataFrame, 价格数据 (index=日期, columns=行业)
        window: int, 回溯窗口（交易日），常用20/60/120/240
    
    返回:
        pd.DataFrame, 动量因子值，值越大动量越强
    """
    return prices_df.pct_change(window)


def momentum_zscore(prices_df, window):
    """
    标准化动量因子（横截面Z-score标准化）
    
    理念：消除不同行业间波动率差异，衡量相对强弱的"程度"。
    构造：每日收益率横截面Z-score标准化后，取窗口期均值。
          z = (r - mean) / std，Factor = mean(z_{t-window+1:t})
    
    参数:
        prices_df: pd.DataFrame, 价格数据 (index=日期, columns=行业)
        window: int, 回溯窗口（交易日），常用20/60/120
    
    返回:
        pd.DataFrame, 标准化动量因子值，值越大相对表现越强
    """
    daily_returns = prices_df.pct_change()
    
    def zscore_row(row):
        valid = row.dropna()
        if len(valid) <= 1:
            return row * np.nan
        mean_val = valid.mean()
        std_val = valid.std()
        if std_val == 0:
            return row * 0
        return (row - mean_val) / std_val
    
    standardized_returns = daily_returns.apply(zscore_row, axis=1)
    momentum_std_factor = standardized_returns.rolling(window=window).mean()
    
    return momentum_std_factor


def momentum_sharpe(prices_df, window):
    """
    夏普动量因子（风险调整后的动量）
    
    理念：偏好"稳健上涨"而非"大起大落"，惩罚高波动。
    构造：Sharpe = mean(r) / std(r)，窗口期内日收益率的均值除以标准差。
    
    参数:
        prices_df: pd.DataFrame, 价格数据 (index=日期, columns=行业)
        window: int, 回溯窗口（交易日），常用20/60
    
    返回:
        pd.DataFrame, 夏普动量因子值，值越大表示稳健上涨
    """
    daily_ret = prices_df.pct_change()
    rolling_mean = daily_ret.rolling(window).mean()
    rolling_std = daily_ret.rolling(window).std()
    sharpe = rolling_mean.div(rolling_std)
    
    return sharpe


def momentum_calmar_ratio(prices_df, window):
    """
    Calmar比率因子（最大回撤调整后的动量）
    
    出处：20180411-海通证券-行业轮动系列研究7：行业间动量和趋势因子的应用分析

    理念：关注"最坏情况下的损失"，偏好持续创新高、回撤小的行业。空头效果强。
    构造：Calmar = 年化收益率 / |最大回撤|
          年化收益率 = (1 + 区间收益率)^(240/window) - 1
    
    参数:
        prices_df: pd.DataFrame, 价格数据 (index=日期, columns=行业)
        window: int, 回溯窗口（交易日），推荐63，可测试20/40/60/120
    
    返回:
        pd.DataFrame, Calmar比率因子值，值越大风险调整后收益越高
    """
    # 假设一年约240个交易日
    trading_days_per_year = 240
    # 计算窗口期内的收益率
    period_return = prices_df.pct_change(window)
    # 年化收益率 = (1 + 区间收益率)^(240/window) - 1
    annualized_return = (1 + period_return) ** (trading_days_per_year / window) - 1
    
    def calc_max_drawdown(price_series, win):
        """
        计算滚动窗口内的最大回撤
        
        最大回撤 = (峰值 - 谷值) / 峰值
        """
        result = pd.Series(index=price_series.index, dtype=float)
        
        for i in range(win, len(price_series)):
            # 获取窗口内的价格序列
            window_prices = price_series.iloc[i-win:i+1]
            # 计算累计最大值（峰值）
            cummax = window_prices.cummax()
            # 计算回撤序列
            drawdown = (cummax - window_prices) / cummax
            # 最大回撤
            max_dd = drawdown.max()
            result.iloc[i] = max_dd
        
        return result
    
    # 计算每个行业的滚动最大回撤
    max_drawdown_df = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)
    for col in prices_df.columns:
        max_drawdown_df[col] = calc_max_drawdown(prices_df[col], window)
    
    # 计算Calmar比率
    # 避免除以零，对最大回撤取绝对值并设置最小值
    max_drawdown_abs = max_drawdown_df.abs()
    max_drawdown_abs = max_drawdown_abs.replace(0, np.nan)  # 避免除以零
    
    calmar = annualized_return / max_drawdown_abs
    
    # 处理无穷大值
    calmar = calmar.replace([np.inf, -np.inf], np.nan)
    
    return calmar


def momentum_rank_zscore(prices_df, window):
    """
    Rank标准化动量因子（横截面排名标准化）
    
    出处：20221201-东方证券-《量化策略研究之六》：行业动量的刻画
    
    理念：只关注"谁比谁强"的顺序信息，消除波动率差异，对极端值不敏感。
    构造：每日收益率横截面排名 → 排名标准化 → 窗口期均值
          z = (Rank - (N+1)/2) / sqrt((N+1)(N-1)/12)
    
    参数:
        prices_df: pd.DataFrame, 价格数据 (index=日期, columns=行业)
        window: int, 回溯窗口（交易日），常用20/60/120
    
    返回:
        pd.DataFrame, Rank标准化动量因子值，值越大排名持续越靠前
    """
    daily_returns = prices_df.pct_change()
    daily_rank = daily_returns.rank(axis=1, ascending=True)
    
    def standardize_rank(rank_row):
        N = rank_row.count()
        if N <= 1:
            return rank_row * np.nan
        mean_rank = (N + 1) / 2
        std_rank = np.sqrt((N + 1) * (N - 1) / 12)
        return (rank_row - mean_rank) / std_rank
    
    standardized_rank = daily_rank.apply(standardize_rank, axis=1)
    momentum_rank_std_factor = standardized_rank.rolling(window=window).mean()
    
    return momentum_rank_std_factor

"""
平稳动量因子
"""
def momentum_volume_return_corr(prices_df, amount_df, window):
    """
    量益相关性动量因子（成交额-收益率相关性）
    
    出处：20220914-长江证券-行业轮动系列(六)：风险篇
    
    理念：量益相关性本质为成交量调整的收益率，属于动量因子。
          该值越大，在行业成交活跃时收益能力越强，风险溢价越高。
          正相关说明增量资金持续推升价格，动量可持续。
    构造：corr(行业成交额, 行业收益率)，计算窗口内的Pearson相关系数
    
    最优参数（研报表14）：
        - window=10日: IC=4.55%, ICIR=17.79%（最优）
        - 可配合分位数优化: 量益相关性_10_分位_720 IC=5.32%, ICIR=21.93%
    
    参数:
        prices_df: pd.DataFrame, 价格数据 (index=日期, columns=行业)
        amount_df: pd.DataFrame, 成交额数据 (index=日期, columns=行业)
        window: int, 回溯窗口（交易日），研报最优10日
    
    返回:
        pd.DataFrame, 量益相关性因子值，值越大风险溢价越高（正向因子）
    """
    # 计算日收益率
    daily_returns = prices_df.pct_change()
    
    # 计算滚动窗口内成交额与收益率的Pearson相关系数
    factor_df = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)
    
    for col in prices_df.columns:
        ret_series = daily_returns[col]
        amt_series = amount_df[col]
        
        # 滚动计算相关性
        factor_df[col] = ret_series.rolling(window=window).corr(amt_series)
    
    return factor_df


def momentum_turnover_adj(prices_df, turnover_df, window):
    """
    换手率调整动量因子（量价背离动量）
    
    出处：20221201-东方证券-《量化策略研究之六》：行业动量的刻画
    
    理念："缩量上涨"优于"放量上涨"。换手率低时信息传播慢，后续上涨概率大。
    构造：收益换手比 = r / Turnover，再做夏普处理 Factor = Mean / Std
    
    参数:
        prices_df: pd.DataFrame, 价格数据 (index=日期, columns=行业)
        turnover_df: pd.DataFrame, 换手率数据 (index=日期, columns=行业)
        window: int, 回溯窗口（交易日），常用20/60
    
    返回:
        pd.DataFrame, 换手率调整动量因子值，值越大"缩量上涨"特征越明显
    """
    daily_returns = prices_df.pct_change()
    return_turnover_ratio = daily_returns / turnover_df
    return_turnover_ratio = return_turnover_ratio.replace([np.inf, -np.inf], np.nan)
    
    rolling_mean = return_turnover_ratio.rolling(window=window).mean()
    rolling_std = return_turnover_ratio.rolling(window=window).std()
    momentum_turnover_adj = rolling_mean / rolling_std
    
    return momentum_turnover_adj


def momentum_price_volume_icir(prices_df, amount_df, window=20,
                                rebalance_freq=20, lookback_num_for_icir=12):
    """
    量价清洗ICIR加权动量因子

    出处：20220406-长江证券-行业轮动系列(五)：动量篇

    理念：从量（成交额）和价（波动率）两个维度"清洗"动量，剔除情绪噪音。
          通过ICIR动态加权将短期和长期动量合成为复合因子。
    构造：
        - 量维度：剔除成交额最高10%交易日，累加剩余日对数收益率
        - 价维度：路径平滑度 = 区间总涨幅(对数) / Σ|日对数收益率|
        - 合成：M = z(Factor_Amt) + z(Factor_Pric)
        - 加权：IR = Mean(IC)/Std(IC)，按IR归一化加权短期(10日)和长期(240日)因子

    参数:
        prices_df: pd.DataFrame, 价格数据 (index=日期, columns=行业)
        amount_df: pd.DataFrame, 成交额数据 (index=日期, columns=行业)
        window: int, 该参数不使用，保留仅为兼容性。因子固定使用短期10天、长期240天
        rebalance_freq: int, 调仓频率，默认20
        lookback_num_for_icir: int or None, 计算ICIR的回溯IC数量，默认12（滚动12个月）

    返回:
        pd.DataFrame, 量价清洗ICIR加权动量因子值，值越大动量越强
    """
    # 固定窗口设置（长江证券研报：短期10天，长期240天）
    short_window = 10
    long_window = 240
    
    def calc_log_returns(prices):
        """计算对数收益率: r_t = ln(Close_t) - ln(Close_{t-1})"""
        return np.log(prices).diff()
    
    def factor_amt_single_window(log_returns_df, amt_df, window):
        """
        量维度改进因子 (Factor_Amt)
        剔除成交额>=90%分位数的K线，累加剩余对数收益率
        """
        result = pd.DataFrame(index=log_returns_df.index, columns=log_returns_df.columns, dtype=float)
        
        for col in log_returns_df.columns:
            ret_series = log_returns_df[col]
            amt_series = amt_df[col]
            
            factor_values = []
            for i in range(len(ret_series)):
                if i < window:
                    factor_values.append(np.nan)
                    continue
                
                ret_window = ret_series.iloc[i-window+1:i+1]
                amt_window = amt_series.iloc[i-window+1:i+1]
                
                valid_amt = amt_window.dropna()
                if len(valid_amt) < 2:
                    factor_values.append(np.nan)
                    continue
                
                p90 = valid_amt.quantile(0.9)
                mask = amt_window < p90
                filtered_ret = ret_window[mask]
                factor_value = filtered_ret.sum() if len(filtered_ret) > 0 else np.nan
                factor_values.append(factor_value)
            
            result[col] = factor_values
        
        return result
    
    def factor_pric_single_window(log_returns_df, prices, window):
        """
        价维度改进因子 (Factor_Pric)
        路径平滑度 = 区间总涨幅(对数) / Σ|日收益率|
        """
        log_prices = np.log(prices)
        total_return = log_prices.diff(window)
        abs_returns = log_returns_df.abs()
        sum_abs_returns = abs_returns.rolling(window=window).sum()
        factor_pric = total_return / sum_abs_returns
        factor_pric = factor_pric.replace([np.inf, -np.inf], np.nan)
        return factor_pric
    
    def zscore_cross_sectional(df):
        """
        横截面Z-Score标准化: z(x) = (x - mean(x)) / std(x)
        """
        def zscore_row(row):
            valid = row.dropna()
            if len(valid) <= 1:
                return row * np.nan
            mean_val = valid.mean()
            std_val = valid.std()
            if std_val == 0 or np.isnan(std_val):
                return row * 0
            return (row - mean_val) / std_val
        return df.apply(zscore_row, axis=1)
    
    def calc_rank_ic(factor_series, return_series):
        """
        计算Rank IC (Spearman相关系数)
        """
        valid_mask = factor_series.notna() & return_series.notna()
        if valid_mask.sum() < 3:
            return np.nan
        factor_valid = factor_series[valid_mask]
        return_valid = return_series[valid_mask]
        ic, _ = stats.spearmanr(factor_valid, return_valid)
        return ic
    
    def calc_dynamic_weights(factor_short_df, factor_long_df, forward_returns_df,
                             rebal_dates, lb_num=None):
        """
        计算动态权重 (基于IC的IR权重)
        
        参数:
        factor_short_df: pd.DataFrame, 短期因子值（调仓日期）
        factor_long_df: pd.DataFrame, 长期因子值（调仓日期）
        forward_returns_df: pd.DataFrame, 未来收益率（调仓日期）
        rebal_dates: DatetimeIndex, 调仓日期列表
        lb_num: int or None, 计算ICIR需要多少个IC
                - None: 使用全部可用的IC
                - 正整数: 只使用最近的N个IC
        
        返回:
        tuple: (weight_short, weight_long) 权重序列
        """
        weight_short = pd.Series(index=rebal_dates, dtype=float)
        weight_long = pd.Series(index=rebal_dates, dtype=float)
        
        # 最少需要3个IC才能计算ICIR
        min_ic_count = 3
        
        for i, date in enumerate(rebal_dates):
            # 获取当前日期之前的所有调仓日期
            past_dates = rebal_dates[:i]
            
            # 如果指定了lb_num，只取最近的N个
            if lb_num is not None and len(past_dates) > lb_num:
                past_dates = past_dates[-lb_num:]
            
            # 回溯期不足，使用等权
            if len(past_dates) < min_ic_count:
                weight_short[date] = 0.5
                weight_long[date] = 0.5
                continue
            
            ic_short_list = []
            ic_long_list = []
            
            for past_date in past_dates:
                if past_date not in factor_short_df.index or past_date not in forward_returns_df.index:
                    continue
                
                ic_short = calc_rank_ic(factor_short_df.loc[past_date], forward_returns_df.loc[past_date])
                ic_long = calc_rank_ic(factor_long_df.loc[past_date], forward_returns_df.loc[past_date])
                
                if not np.isnan(ic_short):
                    ic_short_list.append(ic_short)
                if not np.isnan(ic_long):
                    ic_long_list.append(ic_long)
            
            # 计算IR
            if len(ic_short_list) >= min_ic_count:
                mean_ic_short = np.mean(ic_short_list)
                std_ic_short = np.std(ic_short_list)
                ir_short = mean_ic_short / std_ic_short if std_ic_short > 0 else 0
            else:
                ir_short = 0
            
            if len(ic_long_list) >= min_ic_count:
                mean_ic_long = np.mean(ic_long_list)
                std_ic_long = np.std(ic_long_list)
                ir_long = mean_ic_long / std_ic_long if std_ic_long > 0 else 0
            else:
                ir_long = 0
            
            # 极值处理：负IR设为0
            ir_short = max(0, ir_short)
            ir_long = max(0, ir_long)
            
            # 归一化权重
            total_ir = ir_short + ir_long
            if total_ir > 0:
                weight_short[date] = ir_short / total_ir
                weight_long[date] = ir_long / total_ir
            else:
                weight_short[date] = 0.5
                weight_long[date] = 0.5
        
        return weight_short, weight_long
    
    
    # ========== 第一阶段：数据准备 ==========
    log_returns = calc_log_returns(prices_df)
    
    # ========== 第二阶段：单一窗口因子计算 ==========
    # 短期因子
    factor_amt_short = factor_amt_single_window(log_returns, amount_df, short_window)
    factor_pric_short = factor_pric_single_window(log_returns, prices_df, short_window)
    
    # 长期因子
    factor_amt_long = factor_amt_single_window(log_returns, amount_df, long_window)
    factor_pric_long = factor_pric_single_window(log_returns, prices_df, long_window)
    
    # ========== 第三阶段：截面合成 ==========
    # Z-Score标准化
    z_amt_short = zscore_cross_sectional(factor_amt_short)
    z_pric_short = zscore_cross_sectional(factor_pric_short)
    z_amt_long = zscore_cross_sectional(factor_amt_long)
    z_pric_long = zscore_cross_sectional(factor_pric_long)
    
    # 等权加总
    m_short = z_amt_short + z_pric_short
    m_long = z_amt_long + z_pric_long
    
    # ========== 第四阶段：动态加权 ==========
    # 根据调仓频率生成调仓日期
    all_dates = prices_df.index
    rebalance_indices = list(range(long_window, len(all_dates), rebalance_freq))
    rebalance_dates = all_dates[rebalance_indices]
    
    # 计算未来收益率（用于IC计算）
    # 未来rebalance_freq天的收益率
    forward_returns = prices_df.pct_change(rebalance_freq).shift(-rebalance_freq)
    
    # 提取调仓日期的因子值和未来收益率
    m_short_rebal = m_short.loc[rebalance_dates]
    m_long_rebal = m_long.loc[rebalance_dates]
    forward_returns_rebal = forward_returns.loc[rebalance_dates]
    
    # 计算动态权重
    weight_short, weight_long = calc_dynamic_weights(
        m_short_rebal, m_long_rebal, forward_returns_rebal,
        rebal_dates=rebalance_dates,
        lb_num=lookback_num_for_icir
    )
    
    # 将调仓日权重扩展到日度
    weight_short_daily = weight_short.reindex(prices_df.index, method='ffill')
    weight_long_daily = weight_long.reindex(prices_df.index, method='ffill')
    
    # 计算最终因子
    final_factor = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)
    for col in prices_df.columns:
        final_factor[col] = weight_short_daily * m_short[col] + weight_long_daily * m_long[col]
    return final_factor


def momentum_rebound_with_crowding_filter(prices_df, amount_df, window=240,
                                          crowding_ma_window=5,
                                          crowding_threshold_pct=0.15):
    """
    反弹动量因子（综合动量 + 拥挤度过滤）
    
    出处：20240624-华鑫证券-动量、拥挤度、轮动速率的统一："电风扇"行情下的行业轮动
    
    理念：纯动量策略容易追高过热行业，需用拥挤度过滤规避均值回归风险。
          反弹动量捕捉"V型反转"信号，结合传统动量形成综合动量。
    构造：
        - 传统动量：P_t / P_{t-window} - 1（起点到终点的涨幅）
        - 反弹动量：P_t / min(P_{t-window:t}) - 1（从最低点的反弹幅度）
        - 综合动量：z(传统动量) + z(反弹动量)
        - 拥挤度过滤：成交额MA在历史窗口的分位数，最高15%进入黑名单
    
    参数:
        prices_df: pd.DataFrame, 价格数据 (index=日期, columns=行业)
        amount_df: pd.DataFrame, 成交额数据 (index=日期, columns=行业)
        window: int, 回溯窗口，默认240
        crowding_ma_window: int, 成交额MA窗口，默认5
        crowding_threshold_pct: float, 黑名单阈值，默认0.15
    
    返回:
        pd.DataFrame, 调整后的综合动量因子值，黑名单行业为NaN
    """
    # 使用统一的window参数
    lookback_window = window
    
    def zscore_cross_sectional(df):
        """
        横截面Z-Score标准化: z(x) = (x - mean(x)) / std(x)
        """
        def zscore_row(row):
            valid = row.dropna()
            if len(valid) <= 1:
                return row * np.nan
            mean_val = valid.mean()
            std_val = valid.std()
            if std_val == 0 or np.isnan(std_val):
                return row * 0
            return (row - mean_val) / std_val
        return df.apply(zscore_row, axis=1)
    
    def calc_rolling_percentile(series, lookback):
        """
        计算滚动分位数排名
        当前值在过去lookback天中的分位数位置 (0-1之间)
        """
        result = pd.Series(index=series.index, dtype=float)
        
        for i in range(lookback, len(series)):
            current_val = series.iloc[i]
            if pd.isna(current_val):
                result.iloc[i] = np.nan
                continue
            
            # 获取过去lookback天的历史数据（不包含当天）
            historical = series.iloc[i-lookback:i]
            valid_historical = historical.dropna()
            
            if len(valid_historical) < 10:  # 至少需要10个有效数据点
                result.iloc[i] = np.nan
                continue
            
            # 计算当前值在历史数据中的分位数
            # 分位数 = 小于当前值的历史数据占比
            percentile = (valid_historical < current_val).sum() / len(valid_historical)
            result.iloc[i] = percentile
        
        return result
    
    # ========== 第一步：计算综合动量 ==========
    
    # 1.1 传统动量：过去lookback_window天的累计涨跌幅
    # Ret = P_t / P_{t-lookback_window} - 1
    traditional_momentum = prices_df.pct_change(lookback_window)
    
    # 1.2 反弹动量：距lookback_window天内最低点的涨幅
    # Ret_low = P_t / min(P_{t-lookback_window...t}) - 1
    rolling_min = prices_df.rolling(window=lookback_window, min_periods=1).min()
    rebound_momentum = prices_df / rolling_min - 1
    
    # 1.3 Z-score标准化
    z_traditional = zscore_cross_sectional(traditional_momentum)
    z_rebound = zscore_cross_sectional(rebound_momentum)
    
    # 1.4 等权合成综合动量
    composite_momentum = z_traditional + z_rebound
    
    # ========== 第二步：计算拥挤度指标 ==========
    
    # 2.1 计算成交额的移动平均（平滑处理）
    amount_ma = amount_df.rolling(window=crowding_ma_window, min_periods=1).mean()
    
    # 2.2 计算每个行业的成交额分位数（使用相同的lookback_window）
    crowding_percentile = pd.DataFrame(index=amount_df.index, columns=amount_df.columns, dtype=float)
    
    for col in amount_df.columns:
        crowding_percentile[col] = calc_rolling_percentile(amount_ma[col], lookback_window)
    
    # ========== 第三步：生成拥挤度黑名单 ==========
    
    # 3.1 计算每日需要进入黑名单的行业数量（基于百分比阈值）
    # 黑名单阈值：拥挤度分位数最高的前crowding_threshold_pct比例的行业
    
    def generate_blacklist_row(row, threshold_pct):
        """
        对单行（单日）数据生成黑名单
        选出拥挤度分位数最高的前threshold_pct比例的行业
        """
        valid = row.dropna()
        if len(valid) == 0:
            return pd.Series(False, index=row.index)
        
        # 计算需要进入黑名单的行业数量
        n_blacklist = max(1, int(np.ceil(len(valid) * threshold_pct)))
        
        # 获取拥挤度最高的n_blacklist个行业
        top_crowded = valid.nlargest(n_blacklist).index
        
        # 生成黑名单标记
        blacklist = pd.Series(False, index=row.index)
        blacklist[top_crowded] = True
        
        return blacklist
    
    crowding_blacklist = crowding_percentile.apply(
        lambda row: generate_blacklist_row(row, crowding_threshold_pct),
        axis=1
    )
    
    # ========== 第四步：将黑名单行业的因子值设为NaN ==========
    
    # 创建调整后的因子值副本
    adjusted_momentum = composite_momentum.copy()
    
    # 将黑名单行业的因子值设为NaN
    # 这样在分层时这些行业会被自动排除
    adjusted_momentum[crowding_blacklist] = np.nan
    
    # ========== 返回结果 ==========
    
    return adjusted_momentum


def momentum_amplitude_cut(high_df, low_df, prices_df, window=120,
                           selection_ratio=0.50):
    """
    振幅切割稳健动量因子（Amplitude-Cut Momentum / A因子）

    出处：20200721-开源证券-开源量化评论（3）：A股市场中如何构造动量因子？

    研报核心逻辑：
    - A股市场呈现显著反转效应，传统动量因子失效
    - 反转效应主要来自高振幅（交易活跃）的日子
    - 低振幅日代表"风平浪静"的真实趋势，呈现动量效应
    - 高振幅日代表"过度反应"的噪音，呈现反转效应

    构造步骤（表1）：
    1. 对选定股票，回溯取其最近N个交易日的数据
    2. 计算股票每日的振幅（最高价/最低价-1）
    3. 选择振幅较低的λ比例交易日，涨跌幅加总，记为A因子

    研报最优参数：
    - N = 160个交易日（约8个月）
    - λ = 70%（保留振幅最低的70%交易日）
    - IC均值 = 0.036，ICIR = 1.31

    参数:
        high_df: pd.DataFrame, 最高价数据（后复权）
        low_df: pd.DataFrame, 最低价数据（后复权）
        prices_df: pd.DataFrame, 收盘价数据（后复权）
        window: int, 回溯窗口（交易日），研报最优160
        selection_ratio: float, 低振幅日保留比例λ，研报最优0.70

    返回:
        pd.DataFrame, 振幅切割动量因子值（A因子），值越大动量越强
    """

    # ========== 步骤1：基础特征计算 ==========

    # 日收益率（简单收益率，与研报一致）
    # r_t = Close_t / Close_{t-1} - 1
    daily_returns = prices_df.pct_change()

    # 日内振幅：Amp_t = High_t / Low_t - 1
    # 研报原文步骤2
    amplitude = high_df / low_df - 1

    # 计算需要保留的天数（振幅最低的λ比例）
    # 研报步骤3：选择振幅较低的λ比例交易日
    keep_n = int(window * selection_ratio)

    # ========== 步骤2：初始化结果DataFrame ==========
    factor_df = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)

    # ========== 步骤3：滚动窗口计算因子值 ==========

    # 对每个行业分别计算
    for col in prices_df.columns:
        ret_series = daily_returns[col]
        amp_series = amplitude[col]

        factor_values = [np.nan] * len(prices_df)

        # 从 window 开始遍历（需要完整的回溯窗口）
        for i in range(window, len(prices_df)):

            # 取过去N个交易日的数据（T-N到T-1，不包含当天T）
            # 研报步骤1：回溯取其最近N个交易日的数据
            window_ret = ret_series.iloc[i-window:i]
            window_amp = amp_series.iloc[i-window:i]

            # 构建临时DataFrame用于排序
            window_data = pd.DataFrame({
                'ret': window_ret.values,
                'amp': window_amp.values
            })

            # 剔除NaN值
            window_data = window_data.dropna()

            if len(window_data) < keep_n:
                # 数据不足，跳过
                continue

            # 核心逻辑（研报步骤3）：
            # 1. 按振幅从小到大排序
            # 2. 取前λ比例（低振幅日）
            # 3. 涨跌幅加总
            selected_subset = window_data.nsmallest(keep_n, columns='amp')

            # A因子：低振幅日的涨跌幅之和
            # 研报原文："选择振幅较低的λ比例交易日，涨跌幅加总，记为A因子"
            factor_value = selected_subset['ret'].sum()

            factor_values[i] = factor_value

        factor_df[col] = factor_values

    return factor_df


"""
特质收益动量因子
"""
def momentum_pure_liquidity_stripped(prices_df, turnover_df, window=20,
                                      zscore_window=240, smooth_window=3,
                                      min_industries=15):
    """
    剥离流动性提纯动量因子（Liquidity-Stripped Pure Momentum）
    
    出处：2019年6月15日 光大证券 《再论动量因子——多因子系列报告之二十二》 周萧潇、刘均伟

    理念：区分"非理性繁荣"（换手率异常、情绪过热）和"基本面推动的稳健上涨"，
          通过截面回归剔除流动性和情绪成分，提取"特质动量"（残差）。
    构造：
        - 计算换手率异常度和波动率异常度（时间序列Z-Score）
        - 截面OLS回归：R = α + β1*Z_Turnover + β2*Z_Vol + ε
        - 残差ε即为提纯动量，代表无法被流动性和情绪解释的涨幅
        - 可选平滑处理提高稳健性
    
    参数:
        prices_df: pd.DataFrame, 价格数据 (index=日期, columns=行业)
        turnover_df: pd.DataFrame, 换手率数据 (index=日期, columns=行业)
        window: int, 动量计算窗口，默认20
        zscore_window: int, Z-Score标准化回溯窗口，默认250（约1年）
        smooth_window: int, 残差平滑窗口，默认3
        min_industries: int, 截面回归最少行业数量，默认15
    
    返回:
        pd.DataFrame, 提纯动量因子值，值越大表示"真实上涨"越强
    """
    import statsmodels.api as sm
    
    # ========== Phase 1: 数据准备 ==========
    
    # 1.1 原始动量：过去window天的涨跌幅
    # R_i = P_t / P_{t-window} - 1
    raw_momentum = prices_df.pct_change(window)
    
    # 1.2 平均换手率：过去window天的日均换手率
    avg_turnover = turnover_df.rolling(window=window).mean()
    
    # 1.3 波动率：过去window天日收益率的标准差
    daily_returns = prices_df.pct_change()
    volatility = daily_returns.rolling(window=window).std()
    
    # ========== Phase 2: 纵向标准化（计算异常度）==========
    
    def time_series_zscore(df, window):
        """
        计算时间序列Z-Score
        Z = (当前值 - 过去window天均值) / 过去window天标准差
        
        含义：当前值相对于自身历史水平的异常程度
        """
        rolling_mean = df.rolling(window=window).mean()
        rolling_std = df.rolling(window=window).std()
        # 加上1e-8防止除以0
        z_score = (df - rolling_mean) / (rolling_std + 1e-8)
        return z_score
    
    # 换手率异常度：当前换手率相对于过去1年的Z-Score
    z_turnover = time_series_zscore(avg_turnover, zscore_window)
    
    # 波动率异常度：当前波动率相对于过去1年的Z-Score
    z_volatility = time_series_zscore(volatility, zscore_window)
    
    # ========== Phase 3 & 4: 截面回归与残差提取 ==========
    
    # 初始化结果容器
    pure_momentum_factor = pd.DataFrame(index=prices_df.index,
                                        columns=prices_df.columns,
                                        dtype=float)
    
    # 逐日进行截面回归
    for date in raw_momentum.index:
        # 获取当日横截面数据
        y_data = raw_momentum.loc[date]      # 原始动量
        x1_data = z_turnover.loc[date]       # 换手率异常度
        x2_data = z_volatility.loc[date]     # 波动率异常度
        
        # 拼接数据面板
        current_slice = pd.concat([y_data, x1_data, x2_data], axis=1)
        current_slice.columns = ['Return', 'Turnover_Z', 'Vol_Z']
        
        # 清洗数据：去除任何包含NaN的行业
        current_slice = current_slice.dropna()
        
        # 行业数量太少时跳过
        if len(current_slice) < min_industries:
            continue
        
        # ========== OLS回归 ==========
        # Y = α + β1*X1 + β2*X2 + ε
        # 其中：
        #   Y = 原始动量
        #   X1 = 换手率异常度
        #   X2 = 波动率异常度
        #   ε = 残差（提纯动量）
        
        X = sm.add_constant(current_slice[['Turnover_Z', 'Vol_Z']])
        Y = current_slice['Return']
        
        try:
            model = sm.OLS(Y, X).fit()
            # 获取残差，"提纯动量"
            # 残差 = 实际值 - 预测值
            # 代表无法被换手率和波动率解释的"特质动量"
            pure_momentum_factor.loc[date, current_slice.index] = model.resid.values
        except Exception:
            # 回归失败时跳过
            continue
    
    # ========== Phase 5: 信号平滑 ==========
    
    # 转换为float类型
    pure_momentum_factor = pure_momentum_factor.astype(float)
    
    # 对残差进行移动平均，提高稳健性
    if smooth_window > 1:
        pure_momentum_factor = pure_momentum_factor.rolling(
            window=smooth_window, min_periods=1
        ).mean()
    
    return pure_momentum_factor


def momentum_residual(industry_prices_df, barra_factor_returns_df, window):
    """
    行业残差动量因子 (Industry Residual Momentum) - 严格按兴业证券研报实现

    出处：兴业证券《基于盈余惊喜(基本面)、残差动量(技术面)、北向资金(资金流)的行业轮动模型》2022年3月27日
    核心引用：Blitz, Huij, and Martens (2011) - Residual Momentum

    理念：
    - 传统动量存在"动量崩溃"风险，对策略构造期表现较好的因子有较高风险敞口
    - 如果这些因子在持有期出现反转，普通的动量策略就有可能失效
    - 残差动量通过回归先剥离风险因素的影响，再使用残差构造动量策略

    构造（严格按研报原文）：
    1. 将每个月的各行业指数收益率对 Barra 因子进行滚动回归（回看 12 个月）
    2. 将回归后得到的**当月残差**作为该行业当月的残差动量

    注意：研报使用月度数据进行回归，回看窗口固定为12个月

    参数:
        industry_prices_df: pd.DataFrame, 行业日频价格 (index=日期, columns=行业)
        barra_factor_returns_df: pd.DataFrame, Barra风格因子日频收益率
            - index: 日期
            - columns: 因子名称 (市场, Size, Beta, Momentum, ResidualVolatility,
                       NonlinearSize, BookToPrice, Liquidity, EarningsYield, Growth, Leverage)
        window: int, 回溯窗口（交易日数），会转换为月数（window/20约等于月数）
                研报固定使用12个月回看窗口

    返回:
        pd.DataFrame, 残差动量因子值（日频，月末更新），值越大表示特质动量越强
    """
    import statsmodels.api as sm

    # ========== Phase 1: 数据准备（月度数据） ==========
    # 研报原文："将每个月的各行业指数收益率对 Barra 因子进行滚动回归（回看 12 个月）"
    # 使用月度数据进行回归，回看12个月，不加常数项
    # 因子值 = 当月残差（实际月度收益 - 风格因子解释的收益）
    #
    # 注意：研报使用中信一级行业（28个），本代码使用申万一级行业（30个）

    # 获取每月最后一个交易日的日期
    trade_dates = industry_prices_df.index
    monthly_last = trade_dates.to_series().groupby(
        [trade_dates.year, trade_dates.month]
    ).apply(lambda x: x.iloc[-1])
    monthly_last = pd.DatetimeIndex(monthly_last)

    # 行业月末价格和月度收益率
    industry_prices_monthly = industry_prices_df.loc[monthly_last]
    industry_returns_monthly = industry_prices_monthly.pct_change()

    # Barra因子月度收益率（日度累乘转月度）
    barra_daily = barra_factor_returns_df.copy()
    barra_daily['year_month'] = barra_daily.index.to_period('M')
    
    def compound_returns(group):
        return (1 + group).prod() - 1
    
    barra_numeric = barra_daily.drop(columns=['year_month'])
    barra_numeric['year_month'] = barra_daily['year_month']
    barra_monthly = barra_numeric.groupby('year_month').apply(
        lambda x: compound_returns(x.drop(columns=['year_month']))
    )
    barra_monthly.index = pd.to_datetime([str(p) for p in barra_monthly.index], format='%Y-%m') + pd.offsets.MonthEnd(0)
    
    # 对齐月度索引
    common_months = industry_returns_monthly.index.intersection(barra_monthly.index)
    industry_returns_monthly = industry_returns_monthly.loc[common_months].copy()
    barra_monthly = barra_monthly.loc[common_months].copy()

    # 回看窗口：12个月
    lookback_months = 12

    # ========== Phase 2: 月度因子计算 ==========

    industries = industry_returns_monthly.columns.tolist()
    month_list = common_months.tolist()

    # 初始化月频因子值DataFrame
    factor_monthly = pd.DataFrame(index=common_months,
                                  columns=industries,
                                  dtype=float)

    # 最少需要的观测值：不加常数项，11个因子需要至少12个观测
    n_factors = len(barra_monthly.columns)
    min_obs = n_factors + 1  # 11因子 + 1自由度 = 12

    success_count = 0
    fail_count = 0
    skip_lookback = 0
    skip_min_obs = 0
    
    for i, current_month in enumerate(month_list):
        # 确保有足够的历史数据
        if i < lookback_months:
            skip_lookback += 1
            continue

        # 截取回看窗口内的月度数据（包含当月）
        window_months = month_list[i - lookback_months + 1:i + 1]
        
        X = barra_monthly.loc[window_months]
        
        # 对每个行业进行回归
        for col in industries:
            y = industry_returns_monthly.loc[window_months, col]

            # 数据对齐和清洗
            valid_mask = y.notna() & X.notna().all(axis=1)
            y_valid = y[valid_mask]
            X_valid = X[valid_mask]

            if len(y_valid) < min_obs:
                skip_min_obs += 1
                continue

            try:
                # OLS回归，不加常数项
                model = sm.OLS(y_valid, X_valid).fit()

                # 当月残差 = 特质动量
                current_month_residual = model.resid.iloc[-1]
                
                factor_monthly.loc[current_month, col] = current_month_residual
                success_count += 1

            except Exception:
                fail_count += 1
                continue

    # 输出统计信息
    print(f'  残差动量: 成功{success_count}次, 失败{fail_count}次, 月份数{len(monthly_last)}, 行业数{len(industries)}')
    print(f'  跳过: lookback={skip_lookback}月, min_obs={skip_min_obs}次')
    print(f'  factor_monthly非空值: {factor_monthly.notna().sum().sum()}')

    # ========== Phase 3: 月频因子值扩展到日频 ==========
    factor_daily = pd.DataFrame(index=industry_prices_df.index,
                                columns=industries,
                                dtype=float)

    # 对每个月末日期，填充到下一个月末之前的所有交易日
    valid_months = factor_monthly.dropna(how='all').index
    for i, month_end in enumerate(valid_months):
        if month_end not in factor_daily.index:
            continue

        # 当月末当天填充
        factor_daily.loc[month_end] = factor_monthly.loc[month_end].values

        # 填充到下一个月末之前
        if i + 1 < len(valid_months):
            next_month_end = valid_months[i + 1]
            mask = (factor_daily.index > month_end) & (factor_daily.index <= next_month_end)
        else:
            mask = factor_daily.index > month_end

        factor_daily.loc[mask] = factor_monthly.loc[month_end].values

    return factor_daily


"""
行业间相关性动量因子
"""
def momentum_cross_industry_lasso(prices_df, window, rebalance_freq, benchmark_returns=None, train_periods=36):
    """
    行业间相关性动量因子（基于Lasso回归的领先滞后关系）- 严格按研报逻辑的月频实现。
    
    出处：《行业动量的刻画——量化策略研究之六》的东方证券研究报告，2022年12月01日
    
    研报原文要点：
        - 使用月度数据（每月最后一个交易日）
        - 自变量：所有行业“上月”的超额收益率（相对于行业等权基准）
        - 因变量：目标行业“下月”的超额收益率
        - 对每个行业单独做 Lasso 时间序列回归，使用 AIC 选择正则化强度
        - 每月底用历史样本训练模型，代入当月最新的“上月超额收益”得到下月预测值

    在本实现中：
        - 我们以“月末日期 months[s]”作为回测调仓日，对应持有区间 [months[s], months[s+1]]
        - 对每个样本 s (1 <= s <= T-2)：
              X_s = 所有行业在区间 [months[s-1], months[s]] 的超额收益（即 end-label 为 months[s] 的超额）
              y_s(i) = 行业 i 在下一区间 [months[s], months[s+1]] 的超额收益
        - 在月份 s_pred（调仓日 months[s_pred]）上：
              使用历史 s ∈ [1, s_pred-1] 训练，
              用 X_{s_pred} 生成对 y_{s_pred}(i) 的预测，
              并将该预测值记录在因子日期 months[s_pred]
        - 上层回测中，forward_returns.loc[months[s]] 恰好是区间 [months[s], months[s+1]] 的收益，
          因此本因子与回测目标完全对齐。

    参数:
        prices_df: pd.DataFrame
            行业指数日频价格数据 (index=交易日, columns=行业)，已为后复权价格
        window: int
            回溯窗口（交易日），仅用于预热期和输出标签，不参与模型结构
        rebalance_freq: int
            调仓频率（交易日），为接口兼容参数，本函数内部按月度运算
        benchmark_returns: pd.Series or None
            预留参数（忽略）。函数内部统一按行业等权月度收益构造基准
        train_periods: int or None
            训练样本数量上限（以月为单位）。None 表示使用全部可用历史

    返回:
        pd.DataFrame
            日频因子值 (index=交易日, columns=行业)。
            仅在月末有新值，其余日期向前填充，用于月度调仓回测。
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # === 1. 构造月度价格 & 收益 ===
        # 使用每个月最后一个交易日作为月末（与 factors_analysis 中的月度调仓逻辑保持一致）
        trade_dates = prices_df.index
        monthly_last = trade_dates.to_series().groupby(
            [trade_dates.year, trade_dates.month]
        ).apply(lambda x: x.iloc[-1])
        monthly_last = pd.DatetimeIndex(monthly_last)
        monthly_close = prices_df.loc[monthly_last]

        # 月度简单收益：本月 / 上月 - 1（end-label：index=本月月末，对应区间[上月末, 本月末]）
        monthly_ret = monthly_close.pct_change()

        # 等权行业基准的月度收益
        bench_monthly = monthly_ret.mean(axis=1)
        # 行业月度超额收益
        excess_monthly = monthly_ret.sub(bench_monthly, axis=0)

        industries = prices_df.columns.tolist()
        months = monthly_close.index.tolist()
        T = len(months)

        # 仅在月度索引上暂存预测值（因子日期 = 调仓日 = 区间起点的月末）
        monthly_pred = pd.DataFrame(index=monthly_close.index,
                                    columns=industries,
                                    dtype=float)

        # 至少需要的训练样本数（以月为单位）
        min_train_samples = 10

        # 有效样本 s 范围：1 <= s <= T-2
        #   s 对应调仓日 months[s]，持有区间 [months[s], months[s+1]]
        #   X_s 使用 end-label=months[s] 的超额收益（上一月区间），
        #   y_s 使用 end-label=months[s+1] 的超额收益（下一个区间）。
        for s_pred in range(1 + min_train_samples, T - 1):
            factor_date = months[s_pred]   # 因子 & 调仓日期

            # 训练样本索引 s ∈ [1, s_pred-1]
            train_s = list(range(1, s_pred))
            if train_periods is not None and len(train_s) > train_periods:
                train_s = train_s[-train_periods:]

            X_train_list = []
            y_train_dict = {ind: [] for ind in industries}

            # === 2.1 构造训练样本 ===
            for s in train_s:
                # X_s: 上一月区间[months[s-1], months[s]]的超额收益（end-label=months[s]）
                X_row = excess_monthly.loc[months[s]].values

                # y_s: 下一月区间[months[s], months[s+1]]的超额收益（end-label=months[s+1]）
                y_row = excess_monthly.loc[months[s + 1]]

                # 若整行 X 或 y 全 NaN，则跳过
                if np.all(np.isnan(X_row)) or y_row.isna().all():
                    continue

                X_train_list.append(X_row)
                for ind in industries:
                    y_val = y_row.get(ind, np.nan)
                    y_train_dict[ind].append(y_val)

            if len(X_train_list) < min_train_samples:
                continue

            X_train = np.array(X_train_list)

            # 当前调仓日 s_pred 的特征：X_{s_pred}
            X_current = excess_monthly.loc[months[s_pred]].values.reshape(1, -1)
            if np.any(np.isnan(X_current)):
                continue

            # === 2.2 对每个行业单独做 Lasso 回归 ===
            for ind in industries:
                y_train = np.array(y_train_dict[ind])
                valid_mask = ~np.isnan(y_train)
                if valid_mask.sum() < min_train_samples:
                    continue

                X_valid = X_train[valid_mask]
                y_valid = y_train[valid_mask]

                try:
                    lasso = LassoLarsIC(criterion='aic')
                    lasso.fit(X_valid, y_valid)
                    pred = lasso.predict(X_current)[0]
                    # 在调仓日 factor_date 记录对下一月超额收益的预测
                    monthly_pred.loc[factor_date, ind] = pred
                except Exception:
                    # 回归失败时退化为用历史均值预测
                    monthly_pred.loc[factor_date, ind] = y_valid.mean() if len(y_valid) > 0 else np.nan

        # === 3. 将月度预测扩展回日频索引 ===
        predictions_daily = pd.DataFrame(index=prices_df.index,
                                         columns=industries,
                                         dtype=float)

        # 仅在对应的月末交易日填充值
        for dt in monthly_pred.index:
            if dt in predictions_daily.index:
                predictions_daily.loc[dt] = monthly_pred.loc[dt].values

        # 向前填充，便于统一框架在任意日期取因子值（实际只在月末使用）
        predictions_daily = predictions_daily.ffill()

        return predictions_daily


"""
行业内关系动量因子
"""
def momentum_industry_component(prices_df, window, constituent_df, stock_price_df, industry_code_df, min_stocks=8, std_floor=0.01):
    """
    行业成分股动量因子 (Industry Component Momentum) - 月频版本

    出处：20221201-东方证券-《量化策略研究之六》：行业动量的刻画

    理念：
    - 核心思想："一荣俱荣"才是真景气
    - 行业指数的上涨可能仅由一两只高权重龙头股拉动（虚胖），也可能是行业内所有公司业绩改善带来的普涨（实壮）
    - 后者代表了行业整体的基本面逻辑极其顺畅，多头共识极强，因此未来的趋势持续性更好
    - 类似于在横截面上计算"夏普比率"：不仅看涨幅（均值），还要看涨得乱不乱（标准差）
    - 当行业内个股收益率离散度（Dispersion）低时，意味着市场对该行业的利好解读是一致的，反转风险小

    构造（研报原文）：
    - 公式：过去N个月的 行业内个股收益率均值 / 标准差
    - μ_i：行业内所有成分股在过去N个月的累计收益率的算术平均值（等权）
    - σ_i：行业内所有成分股累计收益率的截面标准差
    - 分子大 → 行业整体涨得好
    - 分母小 → 行业内部分歧小（大家涨幅差不多）
    - 结果 → 涨得又好又稳的行业得分最高

    月频调仓：
    - 每月最后一个交易日计算因子值
    - window参数为月数（1, 3, 6, 12, 24, 36）
    - 研报最优参数：window=12（过去12个月）

    参数:
        prices_df: pd.DataFrame, 行业指数价格数据 (index=日期, columns=行业)
        window: int, 回溯窗口（月数，如12表示过去12个月）
        constituent_df: pd.DataFrame, 行业成分股数据
        stock_price_df: pd.DataFrame, 个股复权收盘价数据
        industry_code_df: pd.DataFrame, 行业代码映射数据
        min_stocks: int, 最少成分股数量门槛，默认8
        std_floor: float, 标准差下限，防止分母过小，默认0.01

    返回:
        pd.DataFrame, 行业成分股动量因子值（月频，仅月末有值）
    """
    from data_loader import get_industry_name_to_code_map

    # ========== 第一阶段：构建行业名称到行业代码的映射 ==========
    industries = prices_df.columns.tolist()
    sw_industry_codes = get_industry_name_to_code_map(industry_code_df)

    industry_name_to_code = {}
    for ind_name in industries:
        clean_name = ind_name.replace('（申万）', '').replace('(申万)', '')
        if clean_name in sw_industry_codes:
            industry_name_to_code[ind_name] = sw_industry_codes[clean_name]

    # ========== 第二阶段：获取月末日期列表 ==========
    # 按月分组，取每月最后一个交易日
    prices_df_copy = prices_df.copy()
    prices_df_copy['year_month'] = prices_df_copy.index.to_period('M')
    month_end_dates = prices_df_copy.groupby('year_month').apply(lambda x: x.index[-1]).tolist()

    # ========== 第三阶段：构建成分股缓存 ==========
    constituent_cache = {}
    for ind_code in industry_name_to_code.values():
        ind_data = constituent_df[constituent_df['index_code'] == ind_code]
        constituent_cache[ind_code] = {
            date: ind_data[ind_data['date'] == date]['wind_code'].tolist()
            for date in ind_data['date'].unique()
        }

    def get_constituent_stocks_pit(industry_code, date):
        """获取指定行业在指定日期的成分股列表（Point-in-Time）"""
        if industry_code not in constituent_cache:
            return []
        ind_cache = constituent_cache[industry_code]
        valid_dates = [d for d in ind_cache.keys() if d <= date]
        if len(valid_dates) == 0:
            return []
        latest_date = max(valid_dates)
        return ind_cache.get(latest_date, [])

    def calculate_stock_returns_monthly(stock_codes, end_date, start_date, price_df):
        """
        计算指定股票从start_date到end_date的累计收益率
        """
        if end_date not in price_df.index or start_date not in price_df.index:
            return pd.Series(dtype=float)

        valid_codes = [code for code in stock_codes if code in price_df.columns]
        if len(valid_codes) == 0:
            return pd.Series(dtype=float)

        start_prices = price_df.loc[start_date, valid_codes]
        end_prices = price_df.loc[end_date, valid_codes]

        # 计算累计收益率
        returns = (end_prices / start_prices) - 1
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.dropna()

        return returns

    # ========== 第四阶段：计算因子值（仅月末） ==========
    factor_df = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)

    print(f"正在计算行业成分股动量因子（月频，回溯{window}个月）...")
    progress = ProgressBar(total=len(month_end_dates), desc='行业成分股动量')

    for i, end_date in enumerate(month_end_dates):
        progress.update(i + 1)

        # 找到window个月前的月末日期
        if i < window:
            continue
        start_date = month_end_dates[i - window]

        for ind_name in industries:
            if ind_name not in industry_name_to_code:
                continue

            ind_code = industry_name_to_code[ind_name]
            stocks = get_constituent_stocks_pit(ind_code, end_date)

            if len(stocks) < min_stocks:
                continue

            # 计算成分股的累计收益率
            stock_returns = calculate_stock_returns_monthly(stocks, end_date, start_date, stock_price_df)

            if len(stock_returns) < min_stocks:
                continue

            # 计算均值和标准差
            mu = stock_returns.mean()
            sigma = stock_returns.std()
            sigma = max(sigma, std_floor)

            # 因子值：μ / σ
            factor_value = mu / sigma
            factor_df.loc[end_date, ind_name] = factor_value

    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    progress.finish()

    return factor_df


def momentum_pca(prices_df, window, pca_window, lag, constituent_df, stock_price_df, stock_mv_df, industry_code_df, weight_threshold=0.80, min_stocks=8, max_stocks=50, suspension_threshold=0.20, n_components=2):
    """
    PcaMom（技术面内生动量）行业轮动因子
    
    出处：兴业证券 - 《分歧和共振——直击行业轮动痛点》2024.04
    分析师：刘海燕、郑兆磊
    
    理念：
    - 解决传统动量策略在"快速轮动"和"抱团瓦解"行情中失效的痛点
    - 不仅看涨幅（量），更看涨得健不健康（质）
    - 共振（Resonance）：行业指数涨 + 成分股合力上涨（集中度高）→ 真主线，敢上仓位
    - 分歧（Divergence）：行业指数涨 + 仅靠权重股拉升/内部混乱（集中度低）→ 鱼尾行情，谨慎参与
    
    构造：
    - 总公式：PcaMom_t = Mom_{t,N} × (PcaScore_t / PcaScore_{t-Lag})
    - Mom_{t,N}：行业指数在过去N天的涨跌幅
    - PcaScore_t：行业内代表性股票过去W天日收益率矩阵的PCA分解，取前2个主成分的解释率之和
    - Ratio：PcaScore_t / PcaScore_{t-Lag}，衡量持有期内结构是凝聚了还是散了
    
    信号解读：
    - 正动量 + Ratio>1（凝聚）：主升浪，Strong Buy
    - 正动量 + Ratio<1（分歧）：强弩之末，Neutral/Sell
    - 负动量 + Ratio>1（凝聚）：主跌浪，Strong Sell
    - 负动量 + Ratio<1（分歧）：左侧震荡，Wait
    
    参数:
        prices_df: pd.DataFrame, 行业指数价格数据 (index=日期, columns=行业)
        window: int, 动量计算窗口（交易日）
        pca_window: int, PCA计算窗口（交易日）
        lag: int, PcaScore变化率的滞后期（交易日）
        constituent_df: pd.DataFrame, 行业成分股数据（通过data_loader.load_constituent_df()加载）
        stock_price_df: pd.DataFrame, 个股复权收盘价数据（通过data_loader.load_stock_price_df()加载）
        stock_mv_df: pd.DataFrame, 个股流通市值数据（通过data_loader.load_stock_mv_df()加载）
        industry_code_df: pd.DataFrame, 行业代码映射数据（通过data_loader.load_industry_code_df()加载）
        weight_threshold: float, 累计权重阈值，默认0.80
        min_stocks: int, 最少选取股票数，默认8
        max_stocks: int, 最多选取股票数，默认50
        suspension_threshold: float, 停牌剔除阈值，默认0.20
        n_components: int, PCA主成分数量，默认2
    
    返回:
        pd.DataFrame, PcaMom因子值，值越大表示动量越强且结构越凝聚
    """
    from sklearn.decomposition import PCA
    from data_loader import get_industry_name_to_code_map
    
    # ========== 第一阶段：数据准备 ==========
    # 计算个股日收益率
    stock_returns_df = stock_price_df.pct_change(fill_method=None)
    
    # ========== 第二阶段：构建行业-成分股映射 ==========
    
    # 获取所有行业（从prices_df的列名中提取）
    industries = prices_df.columns.tolist()
    
    # 获取行业名称到代码的映射
    sw_industry_codes = get_industry_name_to_code_map(industry_code_df)
    
    # ========== 第三阶段：计算动量因子 ==========
    
    # 计算行业指数动量
    # 研报第9页明确说："以双周动量衡量行业短期股价走势"
    # 双周约10个交易日，这里固定使用10天动量，而不是外部传入的window
    momentum_window = 10  # 双周动量
    momentum_df = prices_df.pct_change(momentum_window)
    
    # ========== 第四阶段：计算PcaScore ==========
    
    def get_constituent_stocks(industry_code, date, constituent_data, weight_thresh, min_n, max_n):
        """
        获取指定行业在指定日期的成分股列表
        按权重排序，选取累计权重达到阈值的头部股票
        """
        # 找到该日期之前最近的成分股数据
        valid_dates = constituent_data[constituent_data['date'] <= date]['date'].unique()
        if len(valid_dates) == 0:
            return []
        
        latest_date = max(valid_dates)
        
        # 筛选该行业的成分股
        mask = (constituent_data['date'] == latest_date) & (constituent_data['index_code'] == industry_code)
        industry_constituents = constituent_data[mask].copy()
        
        if len(industry_constituents) == 0:
            return []
        
        # 按权重排序（降序）
        industry_constituents = industry_constituents.sort_values('i_weight', ascending=False)
        
        # 计算累计权重
        industry_constituents['cum_weight'] = industry_constituents['i_weight'].cumsum() / industry_constituents['i_weight'].sum()
        
        # 选取累计权重达到阈值的股票
        selected = industry_constituents[industry_constituents['cum_weight'] <= weight_thresh]
        
        # 如果选取的股票数量不足，补充到最小数量
        if len(selected) < min_n:
            selected = industry_constituents.head(min(min_n, len(industry_constituents)))
        
        # 如果选取的股票数量超过最大值，截断
        if len(selected) > max_n:
            selected = selected.head(max_n)
        
        return selected['wind_code'].tolist()
    
    def calculate_pca_score(stock_codes, date, returns_df, pca_win, susp_thresh, n_comp):
        """
        计算指定股票组合在指定日期的PcaScore
        """
        if len(stock_codes) < n_comp + 1:
            return np.nan
        
        # 获取日期索引
        if date not in returns_df.index:
            return np.nan
        
        date_idx = returns_df.index.get_loc(date)
        if date_idx < pca_win:
            return np.nan
        
        # 获取窗口期内的收益率数据
        start_idx = date_idx - pca_win
        window_returns = returns_df.iloc[start_idx:date_idx]
        
        # 筛选有效的股票代码
        valid_codes = [code for code in stock_codes if code in window_returns.columns]
        if len(valid_codes) < n_comp + 1:
            return np.nan
        
        # 获取这些股票的收益率矩阵
        returns_matrix = window_returns[valid_codes].copy()
        
        # 停牌处理：剔除停牌时间超过阈值的股票
        # 停牌判断：收益率为0或NaN
        suspension_ratio = (returns_matrix.isna() | (returns_matrix == 0)).sum() / len(returns_matrix)
        valid_stocks = suspension_ratio[suspension_ratio < susp_thresh].index.tolist()
        
        if len(valid_stocks) < n_comp + 1:
            return np.nan
        
        returns_matrix = returns_matrix[valid_stocks]
        
        # 填充NaN值（用0填充，表示停牌日无收益）
        returns_matrix = returns_matrix.fillna(0)
        
        # Z-Score标准化（对每列进行标准化）
        returns_std = returns_matrix.std()
        returns_mean = returns_matrix.mean()
        # 避免除以0
        returns_std = returns_std.replace(0, 1)
        returns_matrix_normalized = (returns_matrix - returns_mean) / returns_std
        
        # 再次填充可能产生的NaN
        returns_matrix_normalized = returns_matrix_normalized.fillna(0)
        
        # PCA分解
        try:
            pca = PCA(n_components=min(n_comp, len(valid_stocks)))
            pca.fit(returns_matrix_normalized.values)
            
            # 取前n_comp个主成分的解释率之和
            pca_score = sum(pca.explained_variance_ratio_[:n_comp])
            return pca_score
        except Exception:
            return np.nan
    
    # 初始化PcaScore DataFrame
    pca_score_df = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)
    
    # 构建行业名称到行业代码的映射
    industry_name_to_code = {}
    for ind_name in industries:
        clean_name = ind_name.replace('（申万）', '').replace('(申万)', '')
        if clean_name in sw_industry_codes:
            industry_name_to_code[ind_name] = sw_industry_codes[clean_name]
    
    # 计算每个行业每天的PcaScore
    all_dates = prices_df.index.tolist()

    # 为了效率，只在调仓日计算（这里简化为每天计算）
    # 实际应用中可以只在调仓日计算

    print("正在计算PcaScore...")
    total_dates = len(all_dates)
    progress = ProgressBar(total=total_dates, desc='PcaScore计算')

    for i, date in enumerate(all_dates):
        if i < pca_window + lag:
            continue

        progress.update(i + 1)
        
        for ind_name in industries:
            if ind_name not in industry_name_to_code:
                continue
            
            ind_code = industry_name_to_code[ind_name]
            
            # 获取成分股
            stocks = get_constituent_stocks(
                ind_code, date, constituent_df,
                weight_threshold, min_stocks, max_stocks
            )
            
            if len(stocks) == 0:
                continue
            
            # 计算PcaScore
            pca_score = calculate_pca_score(
                stocks, date, stock_returns_df,
                pca_window, suspension_threshold, n_components
            )
            
            pca_score_df.loc[date, ind_name] = pca_score
    
    # ========== 第五阶段：计算PcaMom因子 ==========
    
    # 计算PcaScore的变化率（Ratio）
    pca_score_lag = pca_score_df.shift(lag)
    pca_ratio = pca_score_df / pca_score_lag
    
    # 处理无穷大和NaN
    pca_ratio = pca_ratio.replace([np.inf, -np.inf], np.nan)
    
    # 计算最终因子值
    pca_mom_factor = momentum_df * pca_ratio
    
    # 处理无穷大和NaN
    pca_mom_factor = pca_mom_factor.replace([np.inf, -np.inf], np.nan)
    
    progress.finish()

    return pca_mom_factor


def momentum_lead_lag_enhanced(prices_df, window, constituent_df, stock_price_df, stock_mv_df, industry_code_file, volume_df, split_ratio=0.5, min_stocks=8):
    """
    龙头领先特征修正后的动量增强因子 (Lead-Lag Enhanced Momentum) - 月频版本

    出处：20180528-光大证券-《行业轮动：从动量谈起——技术指标系列报告之五》

    理念：
    - 核心思想：动量不仅要看"涨了多少"（幅度），还要看"谁带头涨的"（内部结构）
    - 龙头跟随效应（Lead-Lag Effect）：大市值龙头企业对行业基本面转好的反应速度快于小市值跟随企业
    - 补涨逻辑：当龙头股率先拉升、跟随股尚未启动时，预示该行业未来还有"补涨"动力
    - 去伪存真：如果上涨由小盘游资乱炒（龙头没动）或大小齐涨（补涨已完成），后续动量较弱

    构造（原文公式）：
    - Lead_sector = Demean(mean(Leader) - mean(Follower)) / (std(Stock) × MonthlyVolume)
    - AugMomentum = |Lead_sector| × MonthlyReturn_sector

    其中：
    - mean(Leader): 行业内龙头股（前split_ratio市值）等权平均月收益率
    - mean(Follower): 行业内跟随股（后1-split_ratio市值）等权平均月收益率
    - Demean: 截面去均值（减去全市场所有行业的平均差距）
    - std(Stock): 行业内所有股票月收益率的标准差（衡量内部同质化程度）
    - MonthlyVolume: 行业月成交量（惩罚过热行业）
    - MonthlyReturn_sector: 行业指数月收益率

    数据频率：
    - 原文使用月频数据，本实现将日频数据聚合为月频
    - window参数映射：20→1月, 60→3月, 120→6月, 240→12月

    参数:
        prices_df: pd.DataFrame, 行业指数价格数据 (index=日期, columns=行业)
        window: int, 回溯窗口（交易日），会映射为月数：20→1, 60→3, 120→6, 240→12
        constituent_df: pd.DataFrame, 行业成分股数据
        stock_price_df: pd.DataFrame, 个股复权收盘价数据
        stock_mv_df: pd.DataFrame, 个股流通市值数据
        industry_code_file: str, 申万行业指数代码映射文件路径
        volume_df: pd.DataFrame, 行业成交量数据
        split_ratio: float, 龙头/跟随分割参数，默认0.5（前50%市值为龙头）
        min_stocks: int, 最少成分股数量门槛，默认8

    返回:
        pd.DataFrame, 龙头领先增强动量因子值（日频，月末有值），值越大表示动量越强
    """
    import os

    # ========== 第一阶段：窗口映射和月频数据准备 ==========

    # 将日频window映射为月数
    WINDOW_TO_MONTHS = {20: 1, 60: 3, 120: 6, 240: 12}
    lookback_months = WINDOW_TO_MONTHS.get(window, max(1, window // 20))

    print(f"龙头领先增强动量因子: window={window}天 → 回看{lookback_months}个月")

    # 获取每月最后一个交易日
    def get_month_end_dates(date_index):
        """获取每月最后一个交易日"""
        df = pd.DataFrame(index=date_index)
        df['year_month'] = df.index.to_period('M')
        month_ends = df.groupby('year_month').apply(lambda x: x.index[-1], include_groups=False)
        return pd.DatetimeIndex(month_ends.values)

    month_end_dates = get_month_end_dates(prices_df.index)

    # ========== 第二阶段：聚合月频数据 ==========

    # 行业指数月末价格
    prices_monthly = prices_df.loc[prices_df.index.isin(month_end_dates)]

    # 行业月成交量（每月累计）
    volume_df_aligned = volume_df.reindex(index=prices_df.index, columns=prices_df.columns)
    volume_df_aligned['year_month'] = volume_df_aligned.index.to_period('M')
    volume_monthly = volume_df_aligned.groupby('year_month').sum(numeric_only=True)
    volume_monthly.index = month_end_dates[:len(volume_monthly)]  # 对齐到月末日期

    # 个股月末价格和市值
    stock_price_monthly = stock_price_df.loc[stock_price_df.index.isin(month_end_dates)]
    stock_mv_monthly = stock_mv_df.loc[stock_mv_df.index.isin(month_end_dates)]

    # 计算个股月收益率（lookback_months个月）
    stock_returns_monthly = stock_price_monthly.pct_change(lookback_months, fill_method=None)

    # 计算行业月收益率（lookback_months个月）
    sector_returns_monthly = prices_monthly.pct_change(lookback_months, fill_method=None)

    # ========== 第三阶段：构建行业名称到代码的映射 ==========

    industries = prices_df.columns.tolist()

    if not os.path.exists(industry_code_file):
        raise FileNotFoundError(f"行业代码映射文件不存在: {industry_code_file}")

    industry_code_df = pd.read_csv(industry_code_file)

    sw_industry_codes = {}
    for _, row in industry_code_df.iterrows():
        code = row['申万一级行业代码']
        name = row['申万一级行业名称']
        if pd.notna(code) and pd.notna(name):
            clean_name = name.replace('(申万)', '').replace('（申万）', '')
            sw_industry_codes[clean_name] = code

    industry_name_to_code = {}
    for ind_name in industries:
        clean_name = ind_name.replace('（申万）', '').replace('(申万)', '')
        if clean_name in sw_industry_codes:
            industry_name_to_code[ind_name] = sw_industry_codes[clean_name]

    # ========== 第四阶段：定义辅助函数 ==========

    def get_constituent_stocks_pit(industry_code, date, constituent_data):
        """获取指定行业在指定日期的成分股列表（Point-in-Time）"""
        valid_dates = constituent_data[constituent_data['date'] <= date]['date'].unique()
        if len(valid_dates) == 0:
            return []

        latest_date = max(valid_dates)
        mask = (constituent_data['date'] == latest_date) & (constituent_data['index_code'] == industry_code)
        industry_constituents = constituent_data[mask]

        if len(industry_constituents) == 0:
            return []

        return industry_constituents['wind_code'].tolist()

    def calculate_lead_lag_metrics(stock_codes, date, returns_df, mv_df, split_pct):
        """计算龙头领先指标的各项指标"""
        if date not in returns_df.index or date not in mv_df.index:
            return np.nan, np.nan, np.nan, 0

        valid_codes = [code for code in stock_codes
                       if code in returns_df.columns and code in mv_df.columns]

        if len(valid_codes) < 4:
            return np.nan, np.nan, np.nan, 0

        returns = returns_df.loc[date, valid_codes]
        market_values = mv_df.loc[date, valid_codes]

        data = pd.DataFrame({
            'return': returns,
            'mv': market_values
        }).dropna()

        if len(data) < 4:
            return np.nan, np.nan, np.nan, 0

        data = data.sort_values('mv', ascending=False)
        n_stocks = len(data)
        n_leaders = max(1, int(n_stocks * split_pct))

        leaders = data.iloc[:n_leaders]
        followers = data.iloc[n_leaders:]

        if len(followers) == 0:
            return np.nan, np.nan, np.nan, 0

        lead_return = leaders['return'].mean()
        follow_return = followers['return'].mean()
        sector_std = data['return'].std()

        return lead_return, follow_return, sector_std, n_stocks

    # ========== 第五阶段：计算月频龙头领先指标 ==========

    # 初始化月频中间结果
    lead_return_monthly = pd.DataFrame(index=prices_monthly.index, columns=industries, dtype=float)
    follow_return_monthly = pd.DataFrame(index=prices_monthly.index, columns=industries, dtype=float)
    sector_std_monthly = pd.DataFrame(index=prices_monthly.index, columns=industries, dtype=float)

    monthly_dates = prices_monthly.index.tolist()
    total_months = len(monthly_dates)

    print("正在计算龙头领先增强动量因子（月频）...")
    progress = ProgressBar(total=total_months, desc='龙头领先增强动量')

    for i, date in enumerate(monthly_dates):
        if i < lookback_months:
            progress.update(i + 1)
            continue

        progress.update(i + 1)

        for ind_name in industries:
            if ind_name not in industry_name_to_code:
                continue

            ind_code = industry_name_to_code[ind_name]
            stocks = get_constituent_stocks_pit(ind_code, date, constituent_df)

            if len(stocks) < min_stocks:
                continue

            lead_ret, follow_ret, std, n_stocks = calculate_lead_lag_metrics(
                stocks, date, stock_returns_monthly, stock_mv_monthly, split_ratio
            )

            if n_stocks >= min_stocks:
                lead_return_monthly.loc[date, ind_name] = lead_ret
                follow_return_monthly.loc[date, ind_name] = follow_ret
                sector_std_monthly.loc[date, ind_name] = std

    # ========== 第六阶段：计算原始差距并截面去均值 ==========

    raw_gap = lead_return_monthly - follow_return_monthly
    cross_sectional_mean = raw_gap.mean(axis=1)
    demeaned_gap = raw_gap.sub(cross_sectional_mean, axis=0)

    # ========== 第七阶段：计算Lead_sector ==========

    # 对齐月成交量索引
    volume_monthly_aligned = volume_monthly.reindex(index=prices_monthly.index, columns=industries)

    # 分母 = 标准差 × 月成交量
    denominator = sector_std_monthly * volume_monthly_aligned
    denominator = denominator.replace(0, np.nan)

    # 计算Lead_sector
    lead_score = demeaned_gap / denominator
    lead_score = lead_score.replace([np.inf, -np.inf], np.nan)

    # ========== 第八阶段：计算最终因子值 ==========

    # 最终因子 = |Lead_sector| × MonthlyReturn_sector
    factor_monthly = lead_score.abs() * sector_returns_monthly
    factor_monthly = factor_monthly.replace([np.inf, -np.inf], np.nan)

    # ========== 第九阶段：将月频因子扩展回日频 ==========

    # 创建日频因子DataFrame，月末有值，其他日期为NaN
    factor_df = pd.DataFrame(index=prices_df.index, columns=industries, dtype=float)

    # 将月频因子值填入对应的月末日期
    for date in factor_monthly.index:
        if date in factor_df.index:
            factor_df.loc[date] = factor_monthly.loc[date]

    # 向前填充，使得每个月内的所有日期都使用该月末的因子值
    factor_df = factor_df.ffill()

    progress.finish()

    return factor_df

