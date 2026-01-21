# 因子构建模块
# 新增因子注册在factor_analysis.py中
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoLarsIC
from scipy import stats
import warnings


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
                                rebalance_freq=20, lookback_num_for_icir=None):
    """
    量价清洗ICIR加权动量因子
    
    出处：20220406-长江证券-行业轮动系列(五)：动量篇
    
    理念：从量（成交额）和价（波动率）两个维度"清洗"动量，剔除情绪噪音。
          通过ICIR动态加权将短期和长期动量合成为复合因子。
    构造：
        - 量维度：剔除成交额最高10%交易日，累加剩余日收益率
        - 价维度：路径平滑度 = 区间总涨幅 / Σ|日收益率|
        - 合成：M = z(Factor_Amt) + z(Factor_Pric)
        - 加权：IR = Mean(IC)/Std(IC)，按IR归一化加权短期和长期因子
    
    参数:
        prices_df: pd.DataFrame, 价格数据 (index=日期, columns=行业)
        amount_df: pd.DataFrame, 成交额数据 (index=日期, columns=行业)
        window: int, 该参数不使用，保留仅为兼容性。因子固定使用短期20天、长期240天
        rebalance_freq: int, 调仓频率，默认20
        lookback_num_for_icir: int or None, 计算ICIR的回溯IC数量
    
    返回:
        pd.DataFrame, 量价清洗ICIR加权动量因子值，值越大动量越强
    """
    # 固定窗口设置（原文设计：短期20天，长期240天，两者无特定比例关系）
    short_window = 20
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


def momentum_amplitude_cut(high_df, low_df, prices_df, window=60,
                           selection_ratio=0.70, vol_window=20,
                           vol_smooth_window=None):
    """
    波动率调整后的振幅切割动量因子（Vol-Adjusted ID Momentum）
    
    出处：20200721-开源证券-开源量化评论（3）：A股市场中如何构造动量因子？
    
    理念：只统计"风平浪静"日子的涨幅（低振幅日代表真实趋势），
          并用近期波动率作为风险惩罚项，偏好"低调稳健上涨"的行业。
    构造：
        - 分子：按振幅排序，取前selection_ratio比例低振幅日的对数收益率之和
        - 分母：近期vol_window天的波动率
        - 因子 = Mom_raw / Vol_recent
    
    参数:
        high_df: pd.DataFrame, 最高价数据（后复权）
        low_df: pd.DataFrame, 最低价数据（后复权）
        prices_df: pd.DataFrame, 收盘价数据（后复权）
        window: int, 动量窗口，默认60
        selection_ratio: float, 低振幅日保留比例，默认0.70
        vol_window: int, 波动率窗口，默认20
        vol_smooth_window: int or None, 波动率平滑窗口，默认None
    
    返回:
        pd.DataFrame, 波动率调整后的振幅切割动量因子值，值越大动量越稳健
    """
    
    # ========== 步骤1：基础特征计算 ==========
    
    # 对数收益率：解决涨跌幅不对称问题
    # r_t = ln(Close_t) - ln(Close_{t-1})
    log_returns = np.log(prices_df / prices_df.shift(1))
    
    # 日内振幅：Amp_t = High_t / Low_t - 1
    amplitude = high_df / low_df - 1
    
    # 计算需要保留的天数（用于振幅切割）
    keep_n = int(window * selection_ratio)
    
    # ========== 步骤2：预计算滚动波动率 ==========
    # 利用pandas rolling加速计算
    # 这里计算的是每个时点往前看 vol_window 天的波动率
    rolling_vol = log_returns.rolling(window=vol_window).std()
    
    # 可选：对波动率进行平滑处理
    if vol_smooth_window is not None and vol_smooth_window > 1:
        rolling_vol = rolling_vol.rolling(window=vol_smooth_window).mean()
    
    # ========== 步骤3：确定计算起始点 ==========
    # 需要同时满足动量窗口和波动率窗口的数据要求
    start_idx = max(window, vol_window)
    
    # ========== 步骤4：初始化结果DataFrame ==========
    factor_df = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)
    
    # ========== 步骤5：滚动窗口计算因子值 ==========
    
    # 对每个行业分别计算
    for col in prices_df.columns:
        log_ret_series = log_returns[col]
        amp_series = amplitude[col]
        vol_series = rolling_vol[col]
        
        factor_values = [np.nan] * len(prices_df)
        
        # 从 start_idx 开始遍历
        for i in range(start_idx, len(prices_df)):
            
            # ========== A. 分子计算 (ID Momentum) ==========
            # 取 T-window 到 T-1 的切片（不包含当天T，防止未来函数）
            window_log_ret = log_ret_series.iloc[i-window:i]
            window_amp = amp_series.iloc[i-window:i]
            
            # 构建临时DataFrame用于排序
            window_data = pd.DataFrame({
                'log_ret': window_log_ret.values,
                'amp': window_amp.values
            })
            
            # 剔除NaN值
            window_data = window_data.dropna()
            
            if len(window_data) < keep_n:
                # 数据不足，跳过
                continue
            
            # 核心逻辑：
            # 1. 按振幅从小到大排序
            # 2. 取前 selection_ratio 比例（低振幅日）
            # 3. 对数收益率相加，得出这些日子的净趋势
            selected_subset = window_data.nsmallest(keep_n, columns='amp')
            
            # 原始动量：低振幅日的对数收益率之和
            raw_mom = selected_subset['log_ret'].sum()
            
            # ========== B. 分母计算 (Volatility Penalty) ==========
            # 获取 T 时刻对应的 T-1 时的滚动波动率
            # i-1 对应的是昨天的数据行，但rolling已经是往前看的，所以直接用i-1位置的值
            # 注意：rolling_vol.iloc[i-1] 代表的是截止到 i-1 时刻往前看 vol_window 天的波动率
            current_vol = vol_series.iloc[i-1]
            
            # ========== C. 合成 (防除零处理) ==========
            if pd.isna(current_vol) or current_vol == 0:
                # 极低波动或数据缺失时的保护，赋予0
                final_score = 0.0
            else:
                final_score = raw_mom / current_vol
            
            factor_values[i] = final_score
        
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


def momentum_residual(industry_returns_df, barra_factor_returns_df, window):
    """
    行业残差动量因子 (Industry Residual Momentum)
    
    出处：兴业证券《基于盈余惊喜(基本面)、残差动量(技术面)、北向资金(资金流)的行业轮动模型》2022年3月27日
    
    理念：
    - 传统动量存在"动量崩溃"风险，在市场由熊转牛时容易踏空
    - 涨跌幅 = 大盘环境(Beta) + 风格运气(Style) + 自身实力(Alpha/Residual)
    - 残差动量剥离市场和风格因素，只保留行业"特质性"收益
    - 寻找"无论大盘和风格如何变动，凭借自身逻辑依然表现强势"的行业
    
    构造：
    1. 对每个行业，在过去 window 天内进行时间序列OLS回归：
       R_{i,t} = α_i + Σ(β_{i,k} * F_{k,t}) + ε_{i,t}
    2. 残差ε代表剥离Barra因子后的特质收益
    3. 由于OLS回归的残差咍恒为0，改用残差的累计收益率作为特质动量
       因子值 = (1+ε_1)*(1+ε_2)*...*(1+ε_N) - 1 / σ_ε
       简化计算：因子值 = Σε / σ_ε （妙发ε较小时近似等价）
       实际计算：用回归的截距项α作为特质收益率，结合残差标准差构建IR
    
    参数:
        industry_returns_df: pd.DataFrame, 行业日频收益率 (index=日期, columns=行业)
        barra_factor_returns_df: pd.DataFrame, Barra风格因子日频收益率
            - index: 日期
            - columns: 因子名称 (市场, Size, Beta, Momentum, ResidualVolatility, 
                       NonlinearSize, BookToPrice, Liquidity, EarningsYield, Growth, Leverage)
        window: int, 回溯窗口（交易日）
    
    返回:
        pd.DataFrame, 残差动量因子值，值越大表示特质动量越强
    """
    import statsmodels.api as sm
    
    # 确保日期索引对齐
    common_dates = industry_returns_df.index.intersection(barra_factor_returns_df.index)
    industry_returns = industry_returns_df.loc[common_dates].copy()
    barra_returns = barra_factor_returns_df.loc[common_dates].copy()
    
    # 初始化结果DataFrame
    factor_df = pd.DataFrame(index=industry_returns.index, 
                             columns=industry_returns.columns, 
                             dtype=float)
    
    industries = industry_returns.columns.tolist()
    n_dates = len(common_dates)
    
    # 对每个行业进行滚动回归
    for col in industries:
        industry_ret = industry_returns[col]
        
        # 从 window 开始计算，确保有足够的历史数据
        for i in range(window, n_dates):
            # 截取窗口期内的数据
            start_idx = i - window
            end_idx = i  # 不包含当天，避免前视偏差
            
            y = industry_ret.iloc[start_idx:end_idx]
            X = barra_returns.iloc[start_idx:end_idx]
            
            # 数据对齐和清洗
            valid_mask = y.notna() & X.notna().all(axis=1)
            y_valid = y[valid_mask]
            X_valid = X[valid_mask]
            
            # 至少需要一定数量的有效数据点
            min_obs = max(30, len(X.columns) + 5)  # 至少比因子数多5个观测值
            if len(y_valid) < min_obs:
                continue
            
            try:
                # OLS回归，添加常数项
                X_with_const = sm.add_constant(X_valid)
                model = sm.OLS(y_valid, X_with_const).fit()
                
                # 获取回归结果
                # 截距项α代表剥离因子后的平均特质收益率（日均）
                alpha = model.params['const']
                # 残差标准差代表特质收益的波动率
                residual_std = model.resid.std()
                
                # 计算残差动量IR = α * sqrt(N) / σ_ε
                # 年化的特质收益率 / 特质波动率
                # 或者简化为：α * N / σ_ε （累计特质收益 / 特质波动率）
                n_obs = len(y_valid)
                if residual_std > 0:
                    # 累计特质收益 = α * N
                    cumulative_alpha = alpha * n_obs
                    factor_value = cumulative_alpha / residual_std
                else:
                    factor_value = 0.0
                
                # 记录当天的因子值
                current_date = common_dates[i]
                factor_df.loc[current_date, col] = factor_value
                
            except Exception:
                # 回归失败时跳过
                continue
    
    return factor_df


"""
行业间相关性动量因子
"""
def momentum_cross_industry_lasso(prices_df, window, rebalance_freq, benchmark_returns, train_periods=36):
    """
    行业间相关性动量因子（基于Lasso回归的领先滞后关系）
    
    出处：20221201-东方证券-《量化策略研究之六》：行业动量的刻画
    
    理念：利用行业间经济联系（上下游、替代品、宏观周期）预测目标行业未来表现。
          Lasso的L1正则化自动筛选有预测力的行业，避免过拟合。
    构造：用各行业历史超额收益作为特征，Lasso回归预测目标行业未来超额收益。
    
    参数:
        prices_df: pd.DataFrame, 价格数据 (index=日期, columns=行业)
        window: int, 回溯窗口（交易日），常用20/60
        rebalance_freq: int, 调仓频率（交易日），常用20
        benchmark_returns: pd.Series, 基准收益率序列
        train_periods: int, 训练样本数量（调仓周期数），默认36
    
    返回:
        pd.DataFrame, 预测的未来超额收益，值越大预测表现越好
    """
    # 使用上下文管理器局部抑制警告，避免影响全局设置
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        lookback_returns = prices_df.pct_change(window)
        forward_returns = prices_df.pct_change(rebalance_freq).shift(-rebalance_freq)
        
        lookback_benchmark = benchmark_returns.rolling(window).apply(
            lambda x: (1 + x).prod() - 1, raw=True
        )
        forward_benchmark = benchmark_returns.rolling(rebalance_freq).apply(
            lambda x: (1 + x).prod() - 1, raw=True
        ).shift(-rebalance_freq)
        
        excess_lookback = lookback_returns.sub(lookback_benchmark, axis=0)
        excess_forward = forward_returns.sub(forward_benchmark, axis=0)
        
        industries = prices_df.columns.tolist()
        dates = prices_df.index.tolist()
        predictions = pd.DataFrame(index=prices_df.index, columns=industries, dtype=float)
        
        min_train_samples = 10
        min_start = window + min_train_samples * rebalance_freq
        prediction_dates = list(range(min_start, len(dates), rebalance_freq))
        
        for t_idx in prediction_dates:
            if t_idx >= len(dates):
                break
                
            current_date = dates[t_idx]
            X_train_list = []
            y_train_dict = {ind: [] for ind in industries}
            
            sample_indices = list(range(window, t_idx - rebalance_freq, rebalance_freq))
            
            if train_periods is not None and len(sample_indices) > train_periods:
                sample_indices = sample_indices[-train_periods:]
            
            for sample_idx in sample_indices:
                X_row = excess_lookback.iloc[sample_idx].values
                y_idx = sample_idx + rebalance_freq
                if y_idx >= len(dates):
                    continue
                    
                if not np.any(np.isnan(X_row)):
                    X_train_list.append(X_row)
                    for ind in industries:
                        y_val = excess_forward.iloc[sample_idx]
                        if ind in y_val.index:
                            y_train_dict[ind].append(y_val[ind])
                        else:
                            y_train_dict[ind].append(np.nan)
            
            if len(X_train_list) < 10:
                continue
                
            X_train = np.array(X_train_list)
            X_current = excess_lookback.iloc[t_idx].values.reshape(1, -1)
            
            if np.any(np.isnan(X_current)):
                continue
            
            for ind in industries:
                y_train = np.array(y_train_dict[ind])
                valid_mask = ~np.isnan(y_train)
                if valid_mask.sum() < 10:
                    continue
                
                X_valid = X_train[valid_mask]
                y_valid = y_train[valid_mask]
                
                try:
                    lasso = LassoLarsIC(criterion='aic')
                    lasso.fit(X_valid, y_valid)
                    pred = lasso.predict(X_current)[0]
                    predictions.loc[current_date, ind] = pred
                except Exception:
                    predictions.loc[current_date, ind] = y_valid.mean()
        
        predictions = predictions.ffill()
        return predictions


"""
行业内关系动量因子
"""
def momentum_industry_component(prices_df, window, constituent_df, stock_price_df, industry_code_df, min_stocks=8, std_floor=0.01):
    """
    行业成分股动量因子 (Industry Component Momentum)
    
    出处：20221201-东方证券-《量化策略研究之六》：行业动量的刻画
    
    理念：
    - 核心思想："一荣俱荣"才是真景气
    - 行业指数的上涨可能仅由一两只高权重龙头股拉动（虚胖），也可能是行业内所有公司业绩改善带来的普涨（实壮）
    - 后者代表了行业整体的基本面逻辑极其顺畅，多头共识极强，因此未来的趋势持续性更好
    - 类似于在横截面上计算"夏普比率"：不仅看涨幅（均值），还要看涨得乱不乱（标准差）
    - 当行业内个股收益率离散度（Dispersion）低时，意味着市场对该行业的利好解读是一致的，反转风险小
    
    构造：
    - 核心公式：S_i = μ_i / σ_i
    - μ_i：行业内所有成分股在考察期内累计收益率的算术平均值（等权）
    - σ_i：行业内所有成分股累计收益率的标准差
    - 分子大 → 行业整体涨得好
    - 分母小 → 行业内部分歧小（大家涨幅差不多）
    - 结果 → 涨得又好又稳的行业得分最高
    
    注意事项：
    - 使用Point-in-Time数据：必须用"当时那个月该行业包含哪些股票"来计算
    - 等权计算：赋予行业内中小盘股和龙头股相同的权重，反映行业景气度的扩散程度
    - 分母陷阱：对成分股极少的行业，标准差可能异常小，需设置门槛和下限
    
    参数:
        prices_df: pd.DataFrame, 行业指数价格数据 (index=日期, columns=行业)
        window: int, 回溯窗口（交易日）
        constituent_df: pd.DataFrame, 行业成分股数据（通过data_loader.load_constituent_df()加载）
        stock_price_df: pd.DataFrame, 个股复权收盘价数据（通过data_loader.load_stock_price_df()加载）
        industry_code_df: pd.DataFrame, 行业代码映射数据（通过data_loader.load_industry_code_df()加载）
        min_stocks: int, 最少成分股数量门槛，默认8
        std_floor: float, 标准差下限，防止分母过小，默认0.01
    
    返回:
        pd.DataFrame, 行业成分股动量因子值，值越大表示行业"一致性上涨"越强
    """
    from data_loader import get_industry_name_to_code_map
    
    # ========== 第一阶段：构建行业名称到行业代码的映射 ==========
    
    industries = prices_df.columns.tolist()
    
    # 获取行业名称到代码的映射
    sw_industry_codes = get_industry_name_to_code_map(industry_code_df)
    
    # 构建行业名称到代码的映射
    industry_name_to_code = {}
    for ind_name in industries:
        clean_name = ind_name.replace('（申万）', '').replace('(申万)', '')
        if clean_name in sw_industry_codes:
            industry_name_to_code[ind_name] = sw_industry_codes[clean_name]
    
    # ========== 第三阶段：定义辅助函数 ==========
    
    def get_constituent_stocks_pit(industry_code, date, constituent_data):
        """
        获取指定行业在指定日期的成分股列表（Point-in-Time）
        使用该日期之前最近的成分股数据
        """
        # 找到该日期之前最近的成分股数据
        valid_dates = constituent_data[constituent_data['date'] <= date]['date'].unique()
        if len(valid_dates) == 0:
            return []
        
        latest_date = max(valid_dates)
        
        # 筛选该行业的成分股
        mask = (constituent_data['date'] == latest_date) & (constituent_data['index_code'] == industry_code)
        industry_constituents = constituent_data[mask]
        
        if len(industry_constituents) == 0:
            return []
        
        return industry_constituents['wind_code'].tolist()
    
    def calculate_stock_returns(stock_codes, date, price_df, lookback_window):
        """
        计算指定股票在指定日期往前lookback_window天的累计收益率
        剔除数据不完整的股票（新股、长期停牌）
        """
        if date not in price_df.index:
            return pd.Series(dtype=float)
        
        date_idx = price_df.index.get_loc(date)
        if date_idx < lookback_window:
            return pd.Series(dtype=float)
        
        # 获取窗口期的起始和结束价格
        start_idx = date_idx - lookback_window
        start_date = price_df.index[start_idx]
        
        # 筛选有效的股票代码
        valid_codes = [code for code in stock_codes if code in price_df.columns]
        if len(valid_codes) == 0:
            return pd.Series(dtype=float)
      
        # 获取起始和结束价格
        start_prices = price_df.loc[start_date, valid_codes]
        end_prices = price_df.loc[date, valid_codes]
        
        # 计算累计收益率
        returns = (end_prices / start_prices) - 1
        
        # 剔除无效数据（NaN、Inf）
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.dropna()
        
        return returns
    
    # ========== 第四阶段：计算因子值 ==========
    
    # 初始化结果DataFrame
    factor_df = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)
    
    all_dates = prices_df.index.tolist()
    total_dates = len(all_dates)
    
    print("正在计算行业成分股动量因子...")
    
    for i, date in enumerate(all_dates):
        if i < window:
            continue
        
        if i % 100 == 0:
            print(f"  进度: {i}/{total_dates}")
        
        for ind_name in industries:
            if ind_name not in industry_name_to_code:
                continue
            
            ind_code = industry_name_to_code[ind_name]
            
            # 获取当时的成分股（Point-in-Time）
            stocks = get_constituent_stocks_pit(ind_code, date, constituent_df)
            
            # 检查成分股数量门槛
            if len(stocks) < min_stocks:
                continue
            
            # 计算成分股的累计收益率
            stock_returns = calculate_stock_returns(stocks, date, stock_price_df, window)
            
            # 再次检查有效股票数量
            if len(stock_returns) < min_stocks:
                continue
            
            # 计算均值和标准差
            mu = stock_returns.mean()
            sigma = stock_returns.std()
            
            # 应用标准差下限
            sigma = max(sigma, std_floor)
            
            # 计算因子值：μ / σ
            factor_value = mu / sigma
            
            factor_df.loc[date, ind_name] = factor_value
    
    # 处理无穷大和NaN
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    
    print("行业成分股动量因子计算完成")
    
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
    stock_returns_df = stock_price_df.pct_change()
    
    # ========== 第二阶段：构建行业-成分股映射 ==========
    
    # 获取所有行业（从prices_df的列名中提取）
    industries = prices_df.columns.tolist()
    
    # 获取行业名称到代码的映射
    sw_industry_codes = get_industry_name_to_code_map(industry_code_df)
    
    # ========== 第三阶段：计算动量因子 ==========
    
    # 计算行业指数动量
    momentum_df = prices_df.pct_change(window)
    
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
    
    for i, date in enumerate(all_dates):
        if i < pca_window + lag:
            continue
        
        if i % 100 == 0:
            print(f"  进度: {i}/{total_dates}")
        
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
    
    print("PcaMom因子计算完成")
    
    return pca_mom_factor


def momentum_lead_lag_enhanced(prices_df, window, constituent_df, stock_price_df, stock_mv_df, industry_code_file, turnover_df, split_ratio=0.5, min_stocks=8):
    """
    龙头领先特征修正后的动量增强因子 (Lead-Lag Enhanced Momentum)
    
    出处：20180528-光大证券-《行业轮动：从动量谈起——技术指标系列报告之五》
    
    理念：
    - 核心思想：动量不仅要看"涨了多少"（幅度），还要看"谁带头涨的"（内部结构）
    - 龙头跟随效应（Lead-Lag Effect）：大市值龙头企业对行业基本面转好的反应速度快于小市值跟随企业
    - 补涨逻辑：当龙头股率先拉升、跟随股尚未启动时，预示该行业未来还有"补涨"动力
    - 去伪存真：如果上涨由小盘游资乱炒（龙头没动）或大小齐涨（补涨已完成），后续动量较弱
    
    构造：
    - Lead_sector = Demean(R_Leader - R_Follower) / (σ_sector × Volume_sector)
    - Factor = |Lead_sector| × R_sector
    
    其中：
    - R_Leader: 行业内龙头股（前split_ratio市值）等权平均收益率
    - R_Follower: 行业内跟随股（后1-split_ratio市值）等权平均收益率
    - Demean: 截面去均值（减去全市场所有行业的平均差距）
    - σ_sector: 行业内所有股票收益率的标准差（衡量内部同质化程度）
    - Volume_sector: 行业的换手率/成交量（惩罚过热行业）
    - R_sector: 行业指数本身的收益率
    
    信号解读：
    - 绝对值的含义：无论正向差距（龙头领涨）还是负向差距（小票领涨导致的分化），
      只要内部结构撕裂严重且行业总体在涨，就代表一种强动量
    - 分母的作用：
      - 除以标准差：如果行业内部乱作一团（标准差极大），信号不可靠，权重降低
      - 除以成交量：缩量上涨时分母小，因子值暴增，代表"分歧小、反应未结束"的完美动量形态
    
    参数:
        prices_df: pd.DataFrame, 行业指数价格数据 (index=日期, columns=行业)
        window: int, 回溯窗口（交易日）
        constituent_df: pd.DataFrame, 行业成分股数据（通过data_loader.load_constituent_df()加载）
        stock_price_df: pd.DataFrame, 个股复权收盘价数据（通过data_loader.load_stock_price_df()加载）
        stock_mv_df: pd.DataFrame, 个股流通市值数据（通过data_loader.load_stock_mv_df()加载）
        industry_code_file: str, 申万行业指数代码映射文件路径
        turnover_df: pd.DataFrame, 行业换手率数据（用于分母调整，通过data_loader.load_turnover_df()加载）
        split_ratio: float, 龙头/跟随分割参数，默认0.5（前50%市值为龙头）
        min_stocks: int, 最少成分股数量门槛，默认8
    
    返回:
        pd.DataFrame, 龙头领先增强动量因子值，值越大表示动量越强
    """
    import os
    
    # ========== 第一阶段：数据准备 ==========
    # 计算个股收益率
    stock_returns_df = stock_price_df.pct_change(window)
    
    # ========== 第二阶段：构建行业名称到代码的映射 ==========
    
    industries = prices_df.columns.tolist()
    
    # 从申万行业指数CSV文件读取行业代码映射
    if not os.path.exists(industry_code_file):
        raise FileNotFoundError(f"行业代码映射文件不存在: {industry_code_file}")
    
    industry_code_df = pd.read_csv(industry_code_file)
    
    # 构建行业名称到代码的映射字典
    sw_industry_codes = {}
    for _, row in industry_code_df.iterrows():
        code = row['申万一级行业代码']
        name = row['申万一级行业名称']
        if pd.notna(code) and pd.notna(name):
            clean_name = name.replace('(申万)', '').replace('（申万）', '')
            sw_industry_codes[clean_name] = code
    
    # 构建行业名称到代码的映射
    industry_name_to_code = {}
    for ind_name in industries:
        clean_name = ind_name.replace('（申万）', '').replace('(申万)', '')
        if clean_name in sw_industry_codes:
            industry_name_to_code[ind_name] = sw_industry_codes[clean_name]
    
    # ========== 第三阶段：定义辅助函数 ==========
    
    def get_constituent_stocks_pit(industry_code, date, constituent_data):
        """
        获取指定行业在指定日期的成分股列表（Point-in-Time）
        使用该日期之前最近的成分股数据
        """
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
        """
        计算龙头领先指标的各项指标
        
        返回:
            tuple: (lead_return, follow_return, sector_std, n_valid_stocks)
            - lead_return: 龙头组等权平均收益率
            - follow_return: 跟随组等权平均收益率
            - sector_std: 行业内所有股票收益率的标准差
            - n_valid_stocks: 有效股票数量
        """
        if date not in returns_df.index or date not in mv_df.index:
            return np.nan, np.nan, np.nan, 0
        
        # 筛选有效的股票代码
        valid_codes = [code for code in stock_codes 
                       if code in returns_df.columns and code in mv_df.columns]
        
        if len(valid_codes) < 4:  # 至少需要4只股票才能分组
            return np.nan, np.nan, np.nan, 0
        
        # 获取收益率和市值
        returns = returns_df.loc[date, valid_codes]
        market_values = mv_df.loc[date, valid_codes]
        
        # 构建数据框并清洗
        data = pd.DataFrame({
            'return': returns,
            'mv': market_values
        }).dropna()
        
        if len(data) < 4:
            return np.nan, np.nan, np.nan, 0
        
        # 按市值排序（降序）
        data = data.sort_values('mv', ascending=False)
        
        # 计算分割点
        n_stocks = len(data)
        n_leaders = max(1, int(n_stocks * split_pct))
        
        # 划分龙头组和跟随组
        leaders = data.iloc[:n_leaders]
        followers = data.iloc[n_leaders:]
        
        if len(followers) == 0:
            return np.nan, np.nan, np.nan, 0
        
        # 计算等权平均收益率
        lead_return = leaders['return'].mean()
        follow_return = followers['return'].mean()
        
        # 计算行业内所有股票收益率的标准差
        sector_std = data['return'].std()
        
        return lead_return, follow_return, sector_std, n_stocks
    
    # ========== 第四阶段：计算龙头领先指标 ==========
    
    # 初始化中间结果DataFrame
    lead_return_df = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)
    follow_return_df = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)
    sector_std_df = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)
    
    all_dates = prices_df.index.tolist()
    total_dates = len(all_dates)
    
    print("正在计算龙头领先增强动量因子...")
    
    for i, date in enumerate(all_dates):
        if i < window:
            continue
        
        if i % 100 == 0:
            print(f"  进度: {i}/{total_dates}")
        
        for ind_name in industries:
            if ind_name not in industry_name_to_code:
                continue
            
            ind_code = industry_name_to_code[ind_name]
            
            # 获取当时的成分股（Point-in-Time）
            stocks = get_constituent_stocks_pit(ind_code, date, constituent_df)
            
            # 检查成分股数量门槛
            if len(stocks) < min_stocks:
                continue
            
            # 计算龙头领先指标
            lead_ret, follow_ret, std, n_stocks = calculate_lead_lag_metrics(
                stocks, date, stock_returns_df, stock_mv_df, split_ratio
            )
            
            if n_stocks >= min_stocks:
                lead_return_df.loc[date, ind_name] = lead_ret
                follow_return_df.loc[date, ind_name] = follow_ret
                sector_std_df.loc[date, ind_name] = std
    
    # ========== 第五阶段：计算原始差距并截面去均值 ==========
    
    # 计算原始差距：龙头收益率 - 跟随收益率
    raw_gap = lead_return_df - follow_return_df
    
    # 截面去均值：减去当日所有行业的平均差距
    # 目的：剔除市场风格（如全市场都在炒大票）的影响
    cross_sectional_mean = raw_gap.mean(axis=1)
    demeaned_gap = raw_gap.sub(cross_sectional_mean, axis=0)
    
    # ========== 第六阶段：计算Lead_sector ==========
    
    # 分母处理：标准差 × 成交量
    # 使用换手率作为成交量代理
    # 对齐索引
    turnover_aligned = turnover_df.reindex(index=prices_df.index, columns=prices_df.columns)
    # 计算窗口期平均换手率
    avg_turnover = turnover_aligned.rolling(window=window).mean()
    # 分母 = 标准差 × 换手率
    denominator = sector_std_df * avg_turnover
    
    # 避免除以零
    denominator = denominator.replace(0, np.nan)
    
    # 计算Lead_sector
    lead_score = demeaned_gap / denominator
    
    # 处理无穷大值
    lead_score = lead_score.replace([np.inf, -np.inf], np.nan)
    
    # ========== 第七阶段：计算最终因子值 ==========
    
    # 计算行业收益率
    sector_returns = prices_df.pct_change(window)
    
    # 最终因子 = |Lead_sector| × R_sector
    # 取绝对值：无论龙头领涨还是小票领涨，只要分化严重且行业在涨，都是强动量信号
    factor_df = lead_score.abs() * sector_returns
    
    # 处理无穷大和NaN
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    
    print("龙头领先增强动量因子计算完成")
    
    return factor_df

