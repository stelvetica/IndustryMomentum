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


def momentum_cross_industry_lasso(prices_df, window, rebalance_freq, train_periods=36, benchmark_returns=None):
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
        train_periods: int, 训练样本数量（调仓周期数），默认36
        benchmark_returns: pd.Series or None, 基准收益率序列，默认等权平均
    
    返回:
        pd.DataFrame, 预测的未来超额收益，值越大预测表现越好
    """
    # 使用上下文管理器局部抑制警告，避免影响全局设置
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        lookback_returns = prices_df.pct_change(window)
        forward_returns = prices_df.pct_change(rebalance_freq).shift(-rebalance_freq)
        
        if benchmark_returns is None:
            lookback_benchmark = lookback_returns.mean(axis=1)
            forward_benchmark = forward_returns.mean(axis=1)
        else:
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


def momentum_price_volume_icir(prices_df, amount_df, window=20,
                                rebalance_freq=20, lookback_num_for_icir=None):
    """
    量价清洗ICIR加权动量因子（多维度改进的复合动量）
    
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


def momentum_pure_liquidity_stripped(prices_df, turnover_df, window=20,
                                      zscore_window=240, smooth_window=3,
                                      min_industries=15):
    """
    剥离流动性提纯动量因子（Liquidity-Stripped Pure Momentum）
    
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


# ============================================================================
# 兴业证券"正向泡沫"行业轮动因子
# 出处：《如何结合行业轮动的长短信号？》
# 尚未解决 逻辑 + 代码
# ============================================================================
class GaussianBOCD:
    """
    高斯贝叶斯在线变点检测 (Gaussian Bayesian Online Changepoint Detection)
    
    基于 Adams & MacKay (2007) 的算法实现，假设数据服从正态分布。
    使用 Normal-Inverse-Gamma 共轭先验，预测分布为 Student-t 分布。
    
    参数:
        hazard: float, 变点发生的先验概率 (1/expected_run_length)，默认0.01
        mu0: float, 均值的先验均值，默认0
        kappa0: float, 均值先验的强度参数，默认1
        alpha0: float, 方差先验的形状参数，默认1
        beta0: float, 方差先验的尺度参数，默认1
    """
    
    def __init__(self, hazard=0.01, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0):
        self.hazard = hazard
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        
        # 初始化运行长度概率分布
        # run_length_probs[r] = P(run_length = r)
        self.run_length_probs = np.array([1.0])  # 初始时 P(r=0) = 1
        
        # 每个 run length 对应的充分统计量
        # 使用列表存储，索引对应 run length
        self.mu_params = [mu0]      # 后验均值
        self.kappa_params = [kappa0]  # 后验 kappa
        self.alpha_params = [alpha0]  # 后验 alpha
        self.beta_params = [beta0]    # 后验 beta
        
    def _student_t_pdf(self, x, mu, kappa, alpha, beta):
        """
        计算 Student-t 预测分布的概率密度
        
        预测分布: x | data ~ Student-t(2*alpha, mu, beta*(kappa+1)/(alpha*kappa))
        """
        df = 2 * alpha  # 自由度
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        
        # 使用 scipy 的 t 分布
        return stats.t.pdf(x, df=df, loc=mu, scale=scale)
    
    def update(self, x):
        """
        接收新数据点并更新 run length 概率分布
        
        参数:
            x: float, 新的观测值（收益率）
        
        返回:
            np.array, 更新后的 run length 概率分布
        """
        n = len(self.run_length_probs)
        
        # Step 1: 计算每个 run length 下观测 x 的预测概率
        pred_probs = np.zeros(n)
        for r in range(n):
            pred_probs[r] = self._student_t_pdf(
                x,
                self.mu_params[r],
                self.kappa_params[r],
                self.alpha_params[r],
                self.beta_params[r]
            )
        
        # Step 2: 计算增长概率 (growth probabilities)
        # P(r_t = r+1, x_{1:t}) = P(r_{t-1} = r, x_{1:t-1}) * (1 - H) * P(x_t | r)
        growth_probs = self.run_length_probs * (1 - self.hazard) * pred_probs
        
        # Step 3: 计算变点概率 (changepoint probability)
        # P(r_t = 0, x_{1:t}) = sum_r P(r_{t-1} = r, x_{1:t-1}) * H * P(x_t | r)
        cp_prob = np.sum(self.run_length_probs * self.hazard * pred_probs)
        
        # Step 4: 组合新的 run length 分布
        new_run_length_probs = np.zeros(n + 1)
        new_run_length_probs[0] = cp_prob
        new_run_length_probs[1:] = growth_probs
        
        # Step 5: 归一化
        total = np.sum(new_run_length_probs)
        if total > 0:
            new_run_length_probs /= total
        
        # Step 6: 更新充分统计量
        # 对于 r = 0 (新的 run)，使用先验参数
        new_mu = [self.mu0]
        new_kappa = [self.kappa0]
        new_alpha = [self.alpha0]
        new_beta = [self.beta0]
        
        # 对于 r > 0，更新后验参数
        for r in range(n):
            mu_old = self.mu_params[r]
            kappa_old = self.kappa_params[r]
            alpha_old = self.alpha_params[r]
            beta_old = self.beta_params[r]
            
            # Normal-Inverse-Gamma 后验更新公式
            kappa_new = kappa_old + 1
            mu_new = (kappa_old * mu_old + x) / kappa_new
            alpha_new = alpha_old + 0.5
            beta_new = beta_old + 0.5 * kappa_old * (x - mu_old)**2 / kappa_new
            
            new_mu.append(mu_new)
            new_kappa.append(kappa_new)
            new_alpha.append(alpha_new)
            new_beta.append(beta_new)
        
        # 更新状态
        self.run_length_probs = new_run_length_probs
        self.mu_params = new_mu
        self.kappa_params = new_kappa
        self.alpha_params = new_alpha
        self.beta_params = new_beta
        
        return self.run_length_probs
    
    def get_change_prob(self, max_run_length=5):
        """
        获取"最近发生变点"的概率
        
        参数:
            max_run_length: int, 认为是"新趋势"的最大 run length
        
        返回:
            float, P(run_length <= max_run_length)
        """
        if len(self.run_length_probs) <= max_run_length:
            return np.sum(self.run_length_probs)
        return np.sum(self.run_length_probs[:max_run_length + 1])
    
    def reset(self):
        """重置检测器状态"""
        self.run_length_probs = np.array([1.0])
        self.mu_params = [self.mu0]
        self.kappa_params = [self.kappa0]
        self.alpha_params = [self.alpha0]
        self.beta_params = [self.beta0]
def momentum_positive_bubble(prices_df, amount_df,
                              bsadf_min_window=62,
                              bsadf_compare_window=40,
                              bocd_hazard=0.01,
                              bocd_run_length_threshold=5,
                              regime_lookback=4,
                              similarity_lookback=52,
                              similarity_top_n=8,
                              similarity_threshold=4):
    """
    正向泡沫行业轮动因子（兴业证券"理性泡沫"策略）
    
    出处：兴业证券《如何结合行业轮动的长短信号？》
    
    理念：利用"理性泡沫"理论，通过 BSADF（量价爆炸检测）和 BOCD（结构突变检测）
          捕捉行业的起涨点，并结合市场环境进行风控。
    
    构造：
        1. 数据预处理：日频转周频（W-FRI）
        2. BSADF信号：价格和成交额的泡沫检测，双重确认
        3. BOCD信号：收益率变点检测，捕捉趋势起点
        4. 市场环境：根据牛熊市状态调整信号
        5. 信号合成：牛市扩散、熊市清仓、震荡维持
    
    参数:
        prices_df: pd.DataFrame, 日频收盘价数据 (index=日期, columns=行业)
        amount_df: pd.DataFrame, 日频成交额数据 (index=日期, columns=行业)
        bsadf_min_window: int, BSADF最小窗口（周），默认62
        bsadf_compare_window: int, BSADF动态阈值比较窗口（周），默认40
        bocd_hazard: float, BOCD变点先验概率，默认0.01 (1/100)
        bocd_run_length_threshold: int, BOCD新趋势判定阈值，默认5
        regime_lookback: int, 市场状态回看周数，默认4
        similarity_lookback: int, 相似度计算回看周数，默认52
        similarity_top_n: int, 相似行业数量（含自己），默认8
        similarity_threshold: int, 牛市扩散阈值，默认4
    
    返回:
        pd.DataFrame, 周频因子值 (0/1信号)，index为周五日期，columns为行业名称
    """
    from statsmodels.tsa.stattools import adfuller
    from sklearn.metrics.pairwise import cosine_similarity
    
    # ========== Phase 1: 数据预处理 - 日频转周频 ==========
    
    def resample_to_weekly(prices, amount):
        """
        将日频数据重采样为周频
        - 收盘价：取当周最后一个交易日的值
        - 成交额：取当周所有交易日的总和
        """
        # 确保索引是 DatetimeIndex
        prices = prices.copy()
        amount = amount.copy()
        prices.index = pd.to_datetime(prices.index)
        amount.index = pd.to_datetime(amount.index)
        
        # 重采样到周五
        weekly_close = prices.resample('W-FRI').last()
        weekly_amount = amount.resample('W-FRI').sum()
        
        # 剔除空周（长假导致的无数据周）
        valid_weeks = weekly_close.notna().any(axis=1) & weekly_amount.notna().any(axis=1)
        weekly_close = weekly_close[valid_weeks]
        weekly_amount = weekly_amount[valid_weeks]
        
        # 计算周收益率
        weekly_returns = weekly_close.pct_change()
        
        return weekly_close, weekly_amount, weekly_returns
    
    # ========== Phase 2: BSADF 泡沫检测 ==========
    
    def calc_bsadf_stat(series, min_window):
        """
        计算 Backward Sup ADF (BSADF) 统计量序列
        
        对于每个时间点 t，从最小窗口开始递归扩大窗口，
        取所有窗口 ADF 统计量的最大值。
        
        参数:
            series: pd.Series, 对数价格或对数成交额序列
            min_window: int, 最小窗口长度
        
        返回:
            pd.Series, BSADF 统计量序列
        """
        n = len(series)
        bsadf_stats = pd.Series(index=series.index, dtype=float)
        
        for t in range(min_window, n):
            max_adf_stat = -np.inf
            
            # 递归窗口：从 [0, t] 到 [t-min_window, t]
            for s in range(0, t - min_window + 1):
                window_data = series.iloc[s:t+1].dropna()
                
                if len(window_data) < min_window:
                    continue
                
                try:
                    # ADF 检验，返回 (adf_stat, pvalue, usedlag, nobs, critical_values, icbest)
                    adf_result = adfuller(window_data, maxlag=1, regression='c', autolag=None)
                    adf_stat = adf_result[0]
                    
                    if adf_stat > max_adf_stat:
                        max_adf_stat = adf_stat
                except Exception:
                    continue
            
            if max_adf_stat > -np.inf:
                bsadf_stats.iloc[t] = max_adf_stat
        
        return bsadf_stats
    
    def get_bsadf_signal(weekly_close, weekly_amount, min_window, compare_window):
        """
        计算价量双重 BSADF 信号
        
        条件：价格 BSADF > 中位数 且 成交额 BSADF > 中位数
        """
        industries = weekly_close.columns
        signal_df = pd.DataFrame(index=weekly_close.index, columns=industries, dtype=float)
        
        for col in industries:
            # 计算对数序列
            log_close = np.log(weekly_close[col].dropna())
            log_amount = np.log(weekly_amount[col].dropna() + 1)  # +1 避免 log(0)
            
            if len(log_close) < min_window + compare_window:
                continue
            
            # 计算 BSADF 统计量
            bsadf_price = calc_bsadf_stat(log_close, min_window)
            bsadf_amount = calc_bsadf_stat(log_amount, min_window)
            
            # 对齐索引
            common_idx = bsadf_price.dropna().index.intersection(bsadf_amount.dropna().index)
            
            for t_idx, t in enumerate(common_idx):
                if t_idx < compare_window:
                    continue
                
                # 动态阈值：过去 compare_window 周的中位数
                past_price = bsadf_price.loc[common_idx[:t_idx]].tail(compare_window)
                past_amount = bsadf_amount.loc[common_idx[:t_idx]].tail(compare_window)
                
                if len(past_price) < compare_window or len(past_amount) < compare_window:
                    continue
                
                median_price = past_price.median()
                median_amount = past_amount.median()
                
                # 信号生成：双重确认
                current_price = bsadf_price.loc[t]
                current_amount = bsadf_amount.loc[t]
                
                if current_price > median_price and current_amount > median_amount:
                    signal_df.loc[t, col] = 1
                else:
                    signal_df.loc[t, col] = 0
        
        return signal_df.fillna(0).astype(float)
    
    # ========== Phase 3: BOCD 变点检测 ==========
    
    def get_bocd_signal(weekly_returns, hazard, run_length_threshold):
        """
        计算 BOCD 变点检测信号
        
        条件1：最近3周变点概率单调递增
        条件2：当周收益率 > 0
        """
        industries = weekly_returns.columns
        signal_df = pd.DataFrame(index=weekly_returns.index, columns=industries, dtype=float)
        change_prob_df = pd.DataFrame(index=weekly_returns.index, columns=industries, dtype=float)
        
        for col in industries:
            returns = weekly_returns[col].dropna()
            
            if len(returns) < 10:
                continue
            
            # 初始化 BOCD 检测器
            detector = GaussianBOCD(hazard=hazard)
            
            probs = []
            for t, ret in enumerate(returns):
                detector.update(ret)
                prob = detector.get_change_prob(run_length_threshold)
                probs.append(prob)
                change_prob_df.loc[returns.index[t], col] = prob
            
            # 生成信号
            for t in range(2, len(returns)):
                t_date = returns.index[t]
                
                # 条件1：最近3周概率单调递增
                prob_t = probs[t]
                prob_t1 = probs[t-1]
                prob_t2 = probs[t-2]
                
                monotonic_increase = (prob_t > prob_t1) and (prob_t1 > prob_t2)
                
                # 条件2：当周收益率 > 0
                positive_return = returns.iloc[t] > 0
                
                if monotonic_increase and positive_return:
                    signal_df.loc[t_date, col] = 1
                else:
                    signal_df.loc[t_date, col] = 0
        
        return signal_df.fillna(0).astype(float)
    
    # ========== Phase 4: 市场环境判断 ==========
    
    def get_market_regime(weekly_returns, lookback):
        """
        判断市场状态
        
        - 熊市 (-1)：过去 lookback 周中至少 3 周平均收益 < 0
        - 牛市 (1)：过去 lookback 周中至少 3 周平均收益 > 0
        - 震荡 (0)：其他情况
        """
        # 计算全市场平均收益率
        avg_returns = weekly_returns.mean(axis=1)
        
        regime = pd.Series(index=weekly_returns.index, dtype=int)
        
        for t in range(lookback, len(avg_returns)):
            t_date = avg_returns.index[t]
            past_returns = avg_returns.iloc[t-lookback:t]
            
            n_positive = (past_returns > 0).sum()
            n_negative = (past_returns < 0).sum()
            
            if n_negative >= 3:
                regime.loc[t_date] = -1  # 熊市
            elif n_positive >= 3:
                regime.loc[t_date] = 1   # 牛市
            else:
                regime.loc[t_date] = 0   # 震荡
        
        return regime
    
    # ========== Phase 5: 信号合成与风控 ==========
    
    def apply_regime_filter(candidate_df, regime_series, weekly_returns,
                            sim_lookback, top_n, threshold):
        """
        根据市场状态应用风控逻辑
        
        - 熊市：全员清仓
        - 牛市：信号扩散（相似行业补涨）
        - 震荡：维持原判
        """
        final_signal = candidate_df.copy()
        industries = candidate_df.columns
        
        for t_date in candidate_df.index:
            if t_date not in regime_series.index:
                continue
            
            regime = regime_series.loc[t_date]
            
            if regime == -1:
                # 熊市：全员清仓
                final_signal.loc[t_date, :] = 0
                
            elif regime == 1:
                # 牛市：信号扩散
                # 计算行业间相似度
                t_idx = weekly_returns.index.get_loc(t_date)
                
                if t_idx < sim_lookback:
                    continue
                
                # 获取过去 sim_lookback 周的收益率
                past_returns = weekly_returns.iloc[t_idx-sim_lookback:t_idx]
                
                # 转置后计算余弦相似度（行业 x 行业）
                returns_matrix = past_returns.T.fillna(0).values
                
                if returns_matrix.shape[0] < 2:
                    continue
                
                sim_matrix = cosine_similarity(returns_matrix)
                sim_df = pd.DataFrame(sim_matrix, index=industries, columns=industries)
                
                # 对每个行业，找最相似的 top_n 个行业
                current_candidate = candidate_df.loc[t_date]
                
                for ind in industries:
                    if ind not in sim_df.index:
                        continue
                    
                    # 获取最相似的 top_n 个行业（含自己）
                    similar_industries = sim_df.loc[ind].nlargest(top_n).index.tolist()
                    
                    # 统计这些行业中有多少个 candidate = 1
                    n_candidates = sum(current_candidate.get(sim_ind, 0) for sim_ind in similar_industries)
                    
                    # 如果超过阈值，强制置为 1
                    if n_candidates > threshold:
                        final_signal.loc[t_date, ind] = 1
            
            # regime == 0 (震荡)：维持原判，不做修改
        
        return final_signal
    
    # ========== 主流程执行 ==========
    
    # Step 1: 数据预处理
    weekly_close, weekly_amount, weekly_returns = resample_to_weekly(prices_df, amount_df)
    
    # Step 2: 计算 BSADF 信号
    signal_bsadf = get_bsadf_signal(weekly_close, weekly_amount,
                                     bsadf_min_window, bsadf_compare_window)
    
    # Step 3: 计算 BOCD 信号
    signal_bocd = get_bocd_signal(weekly_returns, bocd_hazard, bocd_run_length_threshold)
    
    # Step 4: 生成初始候选
    # Candidate = (BSADF == 1 OR BOCD == 1) AND (Return > 0)
    signal_union = ((signal_bsadf == 1) | (signal_bocd == 1)).astype(float)
    positive_return_mask = (weekly_returns > 0).astype(float)
    
    # 对齐索引
    common_index = signal_union.index.intersection(positive_return_mask.index)
    signal_union = signal_union.loc[common_index]
    positive_return_mask = positive_return_mask.loc[common_index]
    
    candidate = (signal_union * positive_return_mask).fillna(0)
    
    # Step 5: 计算市场状态
    regime = get_market_regime(weekly_returns, regime_lookback)
    
    # Step 6: 应用风控逻辑
    final_factor = apply_regime_filter(candidate, regime, weekly_returns,
                                        similarity_lookback, similarity_top_n,
                                        similarity_threshold)
    
    return final_factor.astype(float)


