"""
正向泡沫行业轮动因子核心模块 (positive_bubble_factor_core.py)

基于兴业证券研报《如何结合行业轮动的长短信号？》复现

核心算法:
- BSADF (Backward Sup ADF): 泡沫检测，捕捉价格/成交额的爆炸性行为
- BOCD (Bayesian Online Changepoint Detection): 变点检测，捕捉趋势起点

信号合成公式 (严格按照研报):
    Buy_Signal = (Signal_BSADF OR Signal_BOCD) AND (Return_t > 0)

模块结构:
1. 数据加载模块
2. BSADF 计算模块 (快速 OLS + numba 加速)
3. BOCD 计算模块 (GaussianBOCD 类)
4. 信号计算接口
5. 回测接口
6. 绩效分析工具函数
7. Excel 导出模块
"""

import pandas as pd
import numpy as np
import warnings
import os
import sys
import time
from scipy import stats
from numba import jit
from datetime import datetime

warnings.filterwarnings('ignore')


# ============================================================================
# 进度显示工具
# ============================================================================

def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}分{secs}秒"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}小时{mins}分"


def print_progress(current: int, total: int, prefix: str = "", 
                   start_time: float = None, bar_length: int = 30):
    """
    打印进度条
    
    参数:
        current: int, 当前进度
        total: int, 总数
        prefix: str, 前缀文字
        start_time: float, 开始时间 (time.time())
        bar_length: int, 进度条长度
    """
    percent = current / total
    filled = int(bar_length * percent)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    # 时间估计
    time_info = ""
    if start_time is not None and current > 0:
        elapsed = time.time() - start_time
        eta = elapsed / current * (total - current)
        time_info = f" | 已用: {format_time(elapsed)} | 剩余: {format_time(eta)}"
    
    sys.stdout.write(f"\r{prefix} |{bar}| {current}/{total} ({percent*100:.1f}%){time_info}")
    sys.stdout.flush()
    
    if current == total:
        print()  # 换行


# ============================================================================
# 第零部分: 数据加载模块
# ============================================================================

# 默认数据文件路径
DEFAULT_CACHE_FILE = "data/sw_industry_data.pkl"


def load_data(
    file_path: str = DEFAULT_CACHE_FILE,
    verbose: bool = True
) -> tuple:
    """
    加载行业数据 (价格和成交额)

    数据格式: pickle 文件，包含
    - 'data': DataFrame (列: 日期, 代码, 名称, CLOSE, HIGH, LOW, VOLUME, DQ_AMTTURNOVER)
    - 'industry_info': DataFrame (列: 代码, 名称)

    参数:
        file_path: str, 数据文件路径
        verbose: bool, 是否打印加载信息

    返回:
        tuple: (prices_df, amount_df)
            - prices_df: pd.DataFrame, 日频收盘价 (index=日期, columns=行业名称)
            - amount_df: pd.DataFrame, 日频成交量 (index=日期, columns=行业名称)
    """
    import pickle

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    with open(file_path, 'rb') as f:
        data_store = pickle.load(f)

    df = data_store['data']

    # 加载价格数据
    prices_df = df.pivot(index='日期', columns='名称', values='CLOSE')
    if not isinstance(prices_df.index, pd.DatetimeIndex):
        prices_df.index = pd.to_datetime(prices_df.index)
    prices_df = prices_df.sort_index()

    # 加载成交量数据 (作为成交额代理)
    amount_df = df.pivot(index='日期', columns='名称', values='VOLUME')
    if not isinstance(amount_df.index, pd.DatetimeIndex):
        amount_df.index = pd.to_datetime(amount_df.index)
    amount_df = amount_df.sort_index()

    if verbose:
        n_industries = df['代码'].nunique()
        print(f"已加载 {n_industries} 个行业的数据")
        print(f"日期范围: {prices_df.index[0].date()} 至 {prices_df.index[-1].date()}")
        print(f"价格数据 shape: {prices_df.shape}")
        print(f"成交量数据 shape: {amount_df.shape}")

    return prices_df, amount_df


# ============================================================================
# 第一部分: 快速 OLS 回归模块 - 使用 numpy 矩阵运算 + numba 加速
# ============================================================================

@jit(nopython=True)
def _fast_ols_t_stat_numba(y: np.ndarray) -> float:
    """
    使用 numba 加速的 OLS 回归 t 统计量计算 (内部函数)

    回归方程: Δy_t = α + β * y_{t-1} + ε_t  (Lag=0, 无时间趋势项)

    参数:
        y: np.ndarray, 对数价格或对数成交额序列 (float64)

    返回:
        float, β 系数的 t 统计量
    """
    n = len(y)

    # 至少需要 3 个数据点才能进行回归
    if n < 3:
        return np.nan

    # 构造因变量 Y = Δy[1:] = y[1:] - y[:-1]
    delta_y = np.diff(y)
    Y = delta_y

    # 构造自变量
    y_lag = y[:-1]
    n_obs = len(Y)

    # 手动计算 (X'X) 和 (X'Y)，避免创建完整的 X 矩阵
    # X = [1, y_lag]，所以 X'X 是 2x2 矩阵
    sum_1 = float(n_obs)
    sum_y = np.sum(y_lag)
    sum_y2 = np.sum(y_lag * y_lag)

    # X'X = [[n, sum_y], [sum_y, sum_y2]]
    det = sum_1 * sum_y2 - sum_y * sum_y

    if abs(det) < 1e-10:
        return np.nan

    # (X'X)^(-1) = 1/det * [[sum_y2, -sum_y], [-sum_y, sum_1]]
    inv_00 = sum_y2 / det
    inv_01 = -sum_y / det
    inv_10 = -sum_y / det
    inv_11 = sum_1 / det

    # X'Y
    sum_Y = np.sum(Y)
    sum_yY = np.sum(y_lag * Y)

    # β̂ = (X'X)^(-1) X'Y
    beta_0 = inv_00 * sum_Y + inv_01 * sum_yY  # 截距
    beta_1 = inv_10 * sum_Y + inv_11 * sum_yY  # y_{t-1} 的系数

    # 计算残差
    residuals = Y - beta_0 - beta_1 * y_lag

    # 计算残差方差 σ² = ε'ε / (n-k)
    k = 2
    sse = np.sum(residuals * residuals)
    sigma_sq = sse / (n_obs - k)

    # 计算 β_1 的标准误
    se_beta = np.sqrt(sigma_sq * inv_11)

    if se_beta > 0:
        t_stat = beta_1 / se_beta
    else:
        t_stat = np.nan

    return t_stat


def fast_ols_t_stat(y) -> float:
    """
    使用 numpy 矩阵运算计算 ADF 回归的 t 统计量 (Lag=0 简化版)

    回归方程: Δy_t = α + β * y_{t-1} + ε_t
    
    参数:
        y: np.ndarray 或 pd.Series, 对数价格或对数成交额序列

    返回:
        float, β 系数的 t 统计量
    """
    if isinstance(y, pd.Series):
        y = y.values
    y = np.asarray(y, dtype=np.float64)
    return _fast_ols_t_stat_numba(y)


# ============================================================================
# 第二部分: BSADF (Backward Sup ADF) 泡沫检测模块
# ============================================================================

def calc_bsadf_stat(series: pd.Series, min_window: int) -> pd.Series:
    """
    计算 Backward Sup ADF (BSADF) 统计量序列

    对于每个时间点 t，从最小窗口开始递归扩大窗口，
    取所有窗口 ADF 统计量的最大值。

    时间复杂度: O(N²)

    参数:
        series: pd.Series, 对数价格或对数成交额序列
        min_window: int, 最小窗口长度 (研报建议 52 周 ≈ 1 年)

    返回:
        pd.Series, BSADF 统计量序列
    """
    n = len(series)
    bsadf_stats = pd.Series(index=series.index, dtype=float)
    values = series.values

    for t in range(min_window, n):
        max_adf_stat = -np.inf

        # 递归窗口: 从 [0, t] 到 [t-min_window, t]
        for s in range(0, t - min_window + 1):
            window_data = values[s:t + 1]

            if np.any(np.isnan(window_data)):
                continue
            if len(window_data) < min_window:
                continue

            adf_stat = fast_ols_t_stat(window_data)

            if not np.isnan(adf_stat) and adf_stat > max_adf_stat:
                max_adf_stat = adf_stat

        if max_adf_stat > -np.inf:
            bsadf_stats.iloc[t] = max_adf_stat

    return bsadf_stats


def compute_bsadf_signal(
    weekly_close: pd.DataFrame,
    weekly_amount: pd.DataFrame,
    min_window: int = 52,
    compare_window: int = 40,
    verbose: bool = True
) -> pd.DataFrame:
    """
    计算价量双重 BSADF 信号

    信号逻辑:
        Signal = 1 当且仅当:
            (BSADF_Price > 40周 Price 中位数) AND
            (BSADF_Volume > 40周 Volume 中位数)

    参数:
        weekly_close: pd.DataFrame, 周频收盘价
        weekly_amount: pd.DataFrame, 周频成交额
        min_window: int, BSADF 最小窗口 (默认 52 周)
        compare_window: int, 动态阈值比较窗口 (默认 40 周)
        verbose: bool, 是否打印进度

    返回:
        pd.DataFrame, 0/1 信号矩阵
    """
    industries = weekly_close.columns
    signal_df = pd.DataFrame(index=weekly_close.index, columns=industries, dtype=float)

    total_industries = len(industries)
    start_time = time.time()
    
    if verbose:
        print(f"    待处理行业数: {total_industries}")
        print(f"    每个行业需计算 2 个 BSADF 序列 (价格 + 成交量)，耗时较长...")
    
    for idx, col in enumerate(industries):
        if verbose:
            print_progress(idx + 1, total_industries, 
                          prefix="    BSADF", start_time=start_time)

        price_series = weekly_close[col].dropna()
        amount_series = weekly_amount[col].dropna()

        if len(price_series) < min_window + compare_window:
            continue

        log_close = np.log(price_series)
        log_amount = np.log(amount_series + 1)  # +1 避免 log(0)

        bsadf_price = calc_bsadf_stat(log_close, min_window)
        bsadf_amount = calc_bsadf_stat(log_amount, min_window)

        common_idx = bsadf_price.dropna().index.intersection(bsadf_amount.dropna().index)

        for t_idx, t in enumerate(common_idx):
            if t_idx < compare_window:
                continue

            past_price = bsadf_price.loc[common_idx[:t_idx]].tail(compare_window)
            past_amount = bsadf_amount.loc[common_idx[:t_idx]].tail(compare_window)

            if len(past_price) < compare_window or len(past_amount) < compare_window:
                continue

            median_price = past_price.median()
            median_amount = past_amount.median()

            current_price = bsadf_price.loc[t]
            current_amount = bsadf_amount.loc[t]

            if current_price > median_price and current_amount > median_amount:
                signal_df.loc[t, col] = 1
            else:
                signal_df.loc[t, col] = 0

    if verbose:
        elapsed = time.time() - start_time
        print(f"    BSADF 计算完成，总耗时: {format_time(elapsed)}")

    return signal_df.fillna(0).astype(float)


# ============================================================================
# 第三部分: BOCD (贝叶斯在线变点检测) 模块
# ============================================================================

class GaussianBOCD:
    """
    高斯贝叶斯在线变点检测 (Gaussian Bayesian Online Changepoint Detection)

    基于 Adams & MacKay (2007) 的算法实现，假设数据服从正态分布。
    使用 Normal-Inverse-Gamma 共轭先验，预测分布为 Student-t 分布。

    参数:
        hazard: float, 变点发生的先验概率 (1/expected_run_length)，默认 0.01 (1/100)
        mu0: float, 均值的先验均值，默认 0
        kappa0: float, 均值先验的强度参数，默认 1
        alpha0: float, 方差先验的形状参数，默认 50 (稳定方差估计)
        beta0: float, 方差先验的尺度参数，默认 0.01 (周收益率典型方差量级)
    """

    def __init__(
        self,
        hazard: float = 0.01,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 50.0,
        beta0: float = 0.01
    ):
        self.hazard = hazard
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

        # 初始化运行长度概率分布
        self.run_length_probs = np.array([1.0])

        # 每个 run length 对应的充分统计量
        self.mu_params = [mu0]
        self.kappa_params = [kappa0]
        self.alpha_params = [alpha0]
        self.beta_params = [beta0]

    def _student_t_pdf(
        self, x: float, mu: float, kappa: float, alpha: float, beta: float
    ) -> float:
        """
        计算 Student-t 预测分布的概率密度

        预测分布: x | data ~ Student-t(2*alpha, mu, beta*(kappa+1)/(alpha*kappa))
        """
        df = 2 * alpha
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        return stats.t.pdf(x, df=df, loc=mu, scale=scale)

    def update(self, x: float) -> np.ndarray:
        """
        接收新数据点并更新 run length 概率分布

        参数:
            x: float, 新的观测值 (收益率)

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
        growth_probs = self.run_length_probs * (1 - self.hazard) * pred_probs

        # Step 3: 计算变点概率 (changepoint probability)
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
        new_mu = [self.mu0]
        new_kappa = [self.kappa0]
        new_alpha = [self.alpha0]
        new_beta = [self.beta0]

        for r in range(n):
            mu_old = self.mu_params[r]
            kappa_old = self.kappa_params[r]
            alpha_old = self.alpha_params[r]
            beta_old = self.beta_params[r]

            # Normal-Inverse-Gamma 后验更新公式
            kappa_new = kappa_old + 1
            mu_new = (kappa_old * mu_old + x) / kappa_new
            alpha_new = alpha_old + 0.5
            beta_new = beta_old + 0.5 * kappa_old * (x - mu_old) ** 2 / kappa_new

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

    def get_changepoint_prob(self) -> float:
        """
        获取当前时刻的变点概率 P(r_t=0)

        返回:
            float, P(r_t=0)
        """
        return self.run_length_probs[0]

    def reset(self):
        """重置检测器状态"""
        self.run_length_probs = np.array([1.0])
        self.mu_params = [self.mu0]
        self.kappa_params = [self.kappa0]
        self.alpha_params = [self.alpha0]
        self.beta_params = [self.beta0]


def compute_bocd_signal(
    weekly_returns: pd.DataFrame,
    hazard: float = 0.01,
    verbose: bool = True
) -> pd.DataFrame:
    """
    计算 BOCD 变点检测信号

    信号逻辑 (严格按照研报):
        Signal = 1 当且仅当:
            最近 3 周变点概率单调递增 (P_t > P_{t-1} > P_{t-2})

    参数:
        weekly_returns: pd.DataFrame, 周频收益率
        hazard: float, 变点先验概率 (默认 0.01 即 1/100)
        verbose: bool, 是否打印进度

    返回:
        pd.DataFrame, 0/1 信号矩阵
    """
    industries = weekly_returns.columns
    signal_df = pd.DataFrame(index=weekly_returns.index, columns=industries, dtype=float)

    total_industries = len(industries)
    start_time = time.time()
    
    if verbose:
        print(f"    待处理行业数: {total_industries}")
    
    for idx, col in enumerate(industries):
        if verbose:
            print_progress(idx + 1, total_industries,
                          prefix="    BOCD ", start_time=start_time)

        returns = weekly_returns[col].dropna()

        if len(returns) < 10:
            continue

        detector = GaussianBOCD(hazard=hazard)
        probs = []

        for ret in returns:
            detector.update(ret)
            prob = detector.get_changepoint_prob()
            probs.append(prob)

        # 生成信号: 最近 3 周概率单调递增
        for t in range(2, len(returns)):
            t_date = returns.index[t]

            prob_t = probs[t]
            prob_t1 = probs[t - 1]
            prob_t2 = probs[t - 2]

            monotonic_increase = (prob_t > prob_t1) and (prob_t1 > prob_t2)

            if monotonic_increase:
                signal_df.loc[t_date, col] = 1
            else:
                signal_df.loc[t_date, col] = 0

    if verbose:
        elapsed = time.time() - start_time
        print(f"    BOCD 计算完成，总耗时: {format_time(elapsed)}")

    return signal_df.fillna(0).astype(float)


# ============================================================================
# 第四部分: 信号计算主接口
# ============================================================================

def resample_to_weekly(
    prices: pd.DataFrame,
    amount: pd.DataFrame
) -> tuple:
    """
    将日频数据重采样为周频

    - 收盘价: 取当周最后一个交易日的值
    - 成交额: 取当周所有交易日的总和

    参数:
        prices: pd.DataFrame, 日频收盘价
        amount: pd.DataFrame, 日频成交额

    返回:
        tuple: (weekly_close, weekly_amount, weekly_returns)
    """
    prices = prices.copy()
    amount = amount.copy()
    prices.index = pd.to_datetime(prices.index)
    amount.index = pd.to_datetime(amount.index)

    # 重采样到周五
    weekly_close = prices.resample('W-FRI').last()
    weekly_amount = amount.resample('W-FRI').sum()

    # 剔除空周
    valid_weeks = weekly_close.notna().any(axis=1) & weekly_amount.notna().any(axis=1)
    weekly_close = weekly_close[valid_weeks]
    weekly_amount = weekly_amount[valid_weeks]

    # 计算周收益率
    weekly_returns = weekly_close.pct_change()

    return weekly_close, weekly_amount, weekly_returns


def compute_positive_bubble_signal(
    prices_df: pd.DataFrame,
    amount_df: pd.DataFrame,
    bsadf_min_window: int = 52,
    bsadf_compare_window: int = 40,
    bocd_hazard: float = 0.01,
    verbose: bool = True
) -> pd.DataFrame:
    """
    计算正向泡沫行业轮动因子信号

    信号合成公式 (严格按照研报):
        Buy_Signal = (Signal_BSADF OR Signal_BOCD) AND (Return_t > 0)

    参数:
        prices_df: pd.DataFrame, 日频收盘价数据 (index=日期, columns=行业)
        amount_df: pd.DataFrame, 日频成交额数据 (index=日期, columns=行业)
        bsadf_min_window: int, BSADF 最小窗口 (周)，默认 52
        bsadf_compare_window: int, BSADF 动态阈值比较窗口 (周)，默认 40
        bocd_hazard: float, BOCD 变点先验概率，默认 0.01
        verbose: bool, 是否打印进度

    返回:
        pd.DataFrame, 周频因子值 (0/1 信号)，index 为周五日期，columns 为行业名称
    """
    if verbose:
        print("Phase 1: 数据预处理 - 日频转周频...")
    weekly_close, weekly_amount, weekly_returns = resample_to_weekly(prices_df, amount_df)
    if verbose:
        print(f"    周频数据点数: {len(weekly_close)}")

    if verbose:
        print("Phase 2: 计算 BSADF 信号...")
    signal_bsadf = compute_bsadf_signal(
        weekly_close, weekly_amount,
        min_window=bsadf_min_window,
        compare_window=bsadf_compare_window,
        verbose=verbose
    )

    if verbose:
        print("Phase 3: 计算 BOCD 信号...")
    signal_bocd = compute_bocd_signal(weekly_returns, hazard=bocd_hazard, verbose=verbose)

    if verbose:
        print("Phase 4: 信号合成...")

    # 信号合成: Buy_Signal = (BSADF OR BOCD) AND (Return > 0)
    signal_union = ((signal_bsadf == 1) | (signal_bocd == 1)).astype(float)
    positive_return_mask = (weekly_returns > 0).astype(float)

    # 对齐索引
    common_index = signal_union.index.intersection(positive_return_mask.index)
    signal_union = signal_union.loc[common_index]
    positive_return_mask = positive_return_mask.loc[common_index]

    final_signal = (signal_union * positive_return_mask).fillna(0)

    if verbose:
        print("信号计算完成！")

    return final_signal.astype(float)


# ============================================================================
# 第五部分: 回测接口
# ============================================================================

def backtest_positive_bubble(
    signal_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    verbose: bool = True
) -> dict:
    """
    基于因子信号进行回测

    回测逻辑:
    - 每周五根据上周信号选择持仓行业 (信号=1 的行业)
    - 等权持有所有信号=1 的行业
    - 基准: 全行业等权均值

    参数:
        signal_df: pd.DataFrame, 周频信号 (0/1)
        prices_df: pd.DataFrame, 日频收盘价数据
        verbose: bool, 是否打印进度

    返回:
        dict: 包含回测结果的字典
            - 'signal_df': 周频信号 DataFrame
            - 'holdings_history': 历史持仓记录 {日期: [行业列表]}
            - 'strategy_nav': 策略净值序列
            - 'benchmark_nav': 基准净值序列
            - 'excess_nav': 超额净值序列
            - 'performance_metrics': 绩效指标
            - 'yearly_returns': 每年收益统计
            - 'rebalance_details': 每次调仓详情
    """
    start_time = time.time()
    
    # 获取周频价格数据
    prices_df_copy = prices_df.copy()
    prices_df_copy.index = pd.to_datetime(prices_df_copy.index)
    weekly_prices = prices_df_copy.resample('W-FRI').last()

    # 对齐信号和价格的索引
    common_dates = signal_df.index.intersection(weekly_prices.index)
    signal_df = signal_df.loc[common_dates]
    weekly_prices = weekly_prices.loc[common_dates]

    if verbose:
        print(f"    有效周数: {len(common_dates)}")

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
        if verbose:
            print_progress(i, total_weeks - 1, prefix="    回测  ", start_time=start_time)

        prev_date = common_dates[i - 1]
        curr_date = common_dates[i]

        # 获取上周的信号 (用于本周持仓)
        prev_signal = signal_df.loc[prev_date]

        # 选择信号=1 的行业
        selected_industries = prev_signal[prev_signal == 1].index.tolist()

        # 记录持仓
        holdings_history[prev_date] = selected_industries

        # 计算本周收益
        curr_returns = weekly_returns.loc[curr_date]

        # 策略收益: 等权持有信号=1 的行业
        if len(selected_industries) > 0:
            strategy_return = curr_returns[selected_industries].mean()
        else:
            strategy_return = 0

        # 基准收益: 全市场等权 (行业均值)
        benchmark_return = curr_returns.mean()

        # 更新净值
        strategy_nav.iloc[i] = strategy_nav.iloc[i - 1] * (1 + strategy_return)
        benchmark_nav.iloc[i] = benchmark_nav.iloc[i - 1] * (1 + benchmark_return)

    # 记录最后一期持仓
    last_date = common_dates[-1]
    last_signal = signal_df.loc[last_date]
    holdings_history[last_date] = last_signal[last_signal == 1].index.tolist()

    # 计算超额净值
    excess_nav = strategy_nav / benchmark_nav

    if verbose:
        elapsed = time.time() - start_time
        print(f"    回测完成，总耗时: {format_time(elapsed)}")

    # 计算绩效指标
    performance_metrics = calculate_performance_metrics(strategy_nav, benchmark_nav)

    # 计算每年收益统计
    yearly_returns = calculate_yearly_returns(strategy_nav, benchmark_nav)

    # 计算每次调仓详情
    rebalance_details = calculate_rebalance_details(strategy_nav, benchmark_nav, holdings_history)

    return {
        'signal_df': signal_df,
        'holdings_history': holdings_history,
        'strategy_nav': strategy_nav,
        'benchmark_nav': benchmark_nav,
        'excess_nav': excess_nav,
        'performance_metrics': performance_metrics,
        'yearly_returns': yearly_returns,
        'rebalance_details': rebalance_details,
    }


# ============================================================================
# 第六部分: 绩效分析工具函数
# ============================================================================

def calculate_performance_metrics(
    strategy_nav: pd.Series,
    benchmark_nav: pd.Series
) -> dict:
    """
    计算绩效指标

    输出格式: 年化收益率(%) 年化波动率(%) 最大回撤(%) 收益风险比 收益回撤比 月度胜率(%)

    参数:
        strategy_nav: pd.Series, 策略净值
        benchmark_nav: pd.Series, 基准净值

    返回:
        dict: 绩效指标字典
    """
    start_date = strategy_nav.index[0]
    end_date = strategy_nav.index[-1]
    years = (end_date - start_date).days / 365.25

    periods_per_year = 52  # 周频

    strategy_returns = strategy_nav.pct_change().dropna()

    # 年化收益率
    annual_return = ((strategy_nav.iloc[-1] / strategy_nav.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0

    # 年化波动率
    volatility = strategy_returns.std() * np.sqrt(periods_per_year) * 100

    # 最大回撤
    cummax = strategy_nav.cummax()
    drawdown = (strategy_nav - cummax) / cummax
    max_drawdown = abs(drawdown.min() * 100)

    # 收益风险比 (简化夏普比率，无风险利率=0)
    risk_return_ratio = annual_return / volatility if volatility > 0 else 0

    # 收益回撤比 (Calmar 比率)
    return_drawdown_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

    # 月度胜率
    monthly_strategy = strategy_nav.resample('M').last()
    monthly_benchmark = benchmark_nav.resample('M').last()

    monthly_strategy_ret = monthly_strategy.pct_change().dropna()
    monthly_benchmark_ret = monthly_benchmark.pct_change().dropna()

    monthly_win_count = (monthly_strategy_ret > monthly_benchmark_ret).sum()
    monthly_total = len(monthly_strategy_ret)
    monthly_win_rate = (monthly_win_count / monthly_total * 100) if monthly_total > 0 else 0

    return {
        '年化收益率(%)': round(annual_return, 2),
        '年化波动率(%)': round(volatility, 2),
        '最大回撤(%)': round(max_drawdown, 2),
        '收益风险比': round(risk_return_ratio, 2),
        '收益回撤比': round(return_drawdown_ratio, 2),
        '月度胜率(%)': round(monthly_win_rate, 2),
    }


def calculate_yearly_returns(
    strategy_nav: pd.Series,
    benchmark_nav: pd.Series,
    start_year: int = 2017
) -> pd.DataFrame:
    """
    计算每年的收益统计

    输出格式: 每年的多头收益% 超额收益% 基准%

    参数:
        strategy_nav: pd.Series, 策略净值
        benchmark_nav: pd.Series, 基准净值
        start_year: int, 起始年份

    返回:
        pd.DataFrame: 每年的收益统计
    """
    years = sorted(set(strategy_nav.index.year))
    years = [y for y in years if y >= start_year]

    yearly_data = []

    for year in years:
        year_mask = strategy_nav.index.year == year
        year_dates = strategy_nav.index[year_mask]

        if len(year_dates) < 2:
            continue

        start_date = year_dates[0]
        end_date = year_dates[-1]

        strategy_return = (strategy_nav.loc[end_date] / strategy_nav.loc[start_date] - 1) * 100
        bench_return = (benchmark_nav.loc[end_date] / benchmark_nav.loc[start_date] - 1) * 100
        excess_return = strategy_return - bench_return

        yearly_data.append({
            '年份': year,
            '多头收益(%)': round(strategy_return, 2),
            '超额收益(%)': round(excess_return, 2),
            '基准(%)': round(bench_return, 2),
        })

    # 添加全样本统计
    if len(strategy_nav) >= 2:
        total_strategy_return = (strategy_nav.iloc[-1] / strategy_nav.iloc[0] - 1) * 100
        total_bench_return = (benchmark_nav.iloc[-1] / benchmark_nav.iloc[0] - 1) * 100
        total_excess_return = total_strategy_return - total_bench_return

        yearly_data.append({
            '年份': '全样本',
            '多头收益(%)': round(total_strategy_return, 2),
            '超额收益(%)': round(total_excess_return, 2),
            '基准(%)': round(total_bench_return, 2),
        })

    return pd.DataFrame(yearly_data)


def calculate_rebalance_details(
    strategy_nav: pd.Series,
    benchmark_nav: pd.Series,
    holdings_history: dict
) -> pd.DataFrame:
    """
    计算每次调仓的详细信息

    输出格式: 每次调仓的单位净值、多头收益、超额收益、基准、选出来的行业

    参数:
        strategy_nav: pd.Series, 策略净值
        benchmark_nav: pd.Series, 基准净值
        holdings_history: dict, {日期: [行业列表]}

    返回:
        pd.DataFrame: 每次调仓的详细信息
    """
    rebalance_data = []
    dates = sorted(holdings_history.keys())

    for i, date in enumerate(dates):
        industries = holdings_history[date]
        cleaned_industries = [clean_industry_name(ind) for ind in industries]

        nav = strategy_nav.loc[date] if date in strategy_nav.index else np.nan
        bench_nav = benchmark_nav.loc[date] if date in benchmark_nav.index else np.nan

        if i > 0:
            prev_date = dates[i - 1]
            prev_nav = strategy_nav.loc[prev_date] if prev_date in strategy_nav.index else np.nan
            prev_bench = benchmark_nav.loc[prev_date] if prev_date in benchmark_nav.index else np.nan

            if not np.isnan(prev_nav) and not np.isnan(nav) and prev_nav > 0:
                period_return = (nav / prev_nav - 1) * 100
            else:
                period_return = np.nan

            if not np.isnan(prev_bench) and not np.isnan(bench_nav) and prev_bench > 0:
                bench_return = (bench_nav / prev_bench - 1) * 100
            else:
                bench_return = np.nan

            if not np.isnan(period_return) and not np.isnan(bench_return):
                excess_return = period_return - bench_return
            else:
                excess_return = np.nan
        else:
            period_return = 0
            bench_return = 0
            excess_return = 0

        rebalance_data.append({
            '调仓日期': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
            '单位净值': round(nav, 4) if not np.isnan(nav) else np.nan,
            '多头收益(%)': round(period_return, 2) if not np.isnan(period_return) else np.nan,
            '超额收益(%)': round(excess_return, 2) if not np.isnan(excess_return) else np.nan,
            '基准(%)': round(bench_return, 2) if not np.isnan(bench_return) else np.nan,
            '持仓数量': len(cleaned_industries),
            '选出的行业': ', '.join(cleaned_industries) if cleaned_industries else '空仓',
        })

    return pd.DataFrame(rebalance_data)


def clean_industry_name(name: str) -> str:
    """
    清理行业名称，删除"（申万）"后缀

    参数:
        name: str, 行业名称

    返回:
        str, 清理后的行业名称
    """
    if isinstance(name, str):
        return name.replace('（申万）', '').replace('(申万)', '')
    return name


# ============================================================================
# 第七部分: 便捷主函数
# ============================================================================

def run_full_backtest(
    prices_df: pd.DataFrame,
    amount_df: pd.DataFrame,
    bsadf_min_window: int = 52,
    bsadf_compare_window: int = 40,
    bocd_hazard: float = 0.01,
    verbose: bool = True
) -> dict:
    """
    一键运行完整的因子计算 + 回测 + 绩效分析

    参数:
        prices_df: pd.DataFrame, 日频收盘价数据 (index=日期, columns=行业)
        amount_df: pd.DataFrame, 日频成交额数据 (index=日期, columns=行业)
        bsadf_min_window: int, BSADF 最小窗口 (周)，默认 52
        bsadf_compare_window: int, BSADF 动态阈值比较窗口 (周)，默认 40
        bocd_hazard: float, BOCD 变点先验概率，默认 0.01
        verbose: bool, 是否打印进度

    返回:
        dict: 包含完整回测结果的字典
    """
    if verbose:
        print("=" * 60)
        print("正向泡沫行业轮动因子回测")
        print("基于兴业证券研报《如何结合行业轮动的长短信号？》")
        print("=" * 60)

    # Step 1: 计算因子信号
    signal_df = compute_positive_bubble_signal(
        prices_df, amount_df,
        bsadf_min_window=bsadf_min_window,
        bsadf_compare_window=bsadf_compare_window,
        bocd_hazard=bocd_hazard,
        verbose=verbose
    )

    if verbose:
        print(f"\n信号计算完成，周频数据点数: {len(signal_df)}")
        print("\n开始回测计算...")

    # Step 2: 运行回测
    result = backtest_positive_bubble(signal_df, prices_df, verbose=verbose)

    if verbose:
        print("\n" + "=" * 60)
        print("绩效指标")
        print("=" * 60)
        for key, value in result['performance_metrics'].items():
            print(f"  {key}: {value}")

        print("\n" + "=" * 60)
        print("每年收益统计")
        print("=" * 60)
        print(result['yearly_returns'].to_string(index=False))

        print("\n回测完成！")

    return result


def print_selected_industries(signal_df: pd.DataFrame, n_recent: int = 10):
    """
    打印最近几个调仓日选出的行业

    参数:
        signal_df: pd.DataFrame, 信号矩阵
        n_recent: int, 显示最近几期
    """
    print("\n" + "=" * 60)
    print(f"最近 {n_recent} 个调仓日选出的行业")
    print("=" * 60)

    recent_dates = signal_df.index[-n_recent:]

    for date in recent_dates:
        signal_row = signal_df.loc[date]
        selected = signal_row[signal_row == 1].index.tolist()
        cleaned = [clean_industry_name(ind) for ind in selected]

        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)

        if cleaned:
            print(f"\n{date_str} ({len(cleaned)}个行业):")
            print(f"  {', '.join(cleaned)}")
        else:
            print(f"\n{date_str}: 空仓")


# ============================================================================
# 第八部分: Excel 导出模块
# ============================================================================

def create_nav_df(
    strategy_nav: pd.Series,
    benchmark_nav: pd.Series,
    excess_nav: pd.Series
) -> pd.DataFrame:
    """
    创建净值 DataFrame

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

    nav_df.index = nav_df.index.strftime('%Y-%m-%d')
    nav_df.index.name = '日期'
    nav_df = nav_df.round(4)

    return nav_df


def create_holdings_df(holdings_history: dict) -> pd.DataFrame:
    """
    创建历史持仓 DataFrame

    参数:
        holdings_history: dict, {日期: [行业列表]}

    返回:
        pd.DataFrame: 历史持仓记录
    """
    sorted_dates = sorted(holdings_history.keys(), reverse=True)

    data = []
    for date in sorted_dates:
        industries = holdings_history[date]
        cleaned_industries = [clean_industry_name(ind) for ind in industries]

        data.append({
            '日期': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
            '持仓数量': len(cleaned_industries),
            '持仓行业': ', '.join(cleaned_industries) if cleaned_industries else '空仓',
        })

    return pd.DataFrame(data)


def export_to_excel(
    backtest_result: dict,
    output_file: str = None,
    verbose: bool = True
) -> str:
    """
    将回测结果导出到 Excel

    参数:
        backtest_result: dict, 回测结果
        output_file: str, 输出文件路径，默认为 "positive_bubble_backtest_YYYYMMDD_HHMMSS.xlsx"
        verbose: bool, 是否打印导出信息

    返回:
        str: 导出文件路径
    """
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"positive_bubble_backtest_{timestamp}.xlsx"

    if verbose:
        print(f"\n正在导出回测结果到: {output_file}")

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # ========== Sheet 1: 策略概览 ==========
        start_row = 0

        # 因子说明
        factor_doc = """
正向泡沫行业轮动因子 (momentum_positive_bubble)

出处：兴业证券《如何结合行业轮动的长短信号？》

理念：利用"理性泡沫"理论，通过 BSADF（量价爆炸检测）和 BOCD（结构突变检测）
      捕捉行业的起涨点。

构造：
    1. 数据预处理：日频转周频（W-FRI）
    2. BSADF信号：价格和成交额的泡沫检测，双重确认
    3. BOCD信号：收益率变点检测，捕捉趋势起点
    4. 信号合成：Buy_Signal = (BSADF OR BOCD) AND (Return > 0)

特点：
    - 返回0/1信号，而非连续因子值
    - 周频调仓（每周五）
    - 信号=1表示持有，信号=0表示不持有
"""
        doc_df = pd.DataFrame({'【因子说明】': [factor_doc]})
        doc_df.to_excel(writer, sheet_name='策略概览', startrow=start_row, index=False)
        start_row += 20

        # 绩效指标
        metrics = backtest_result['performance_metrics']
        metrics_df = pd.DataFrame([metrics]).T
        metrics_df.columns = ['数值']
        metrics_df.index.name = '指标'

        header_df = pd.DataFrame({'【绩效指标】': ['']})
        header_df.to_excel(writer, sheet_name='策略概览', startrow=start_row, index=False)
        start_row += 1
        metrics_df.to_excel(writer, sheet_name='策略概览', startrow=start_row, index=True)
        start_row += len(metrics_df) + 3

        # 每年收益统计
        yearly_df = backtest_result['yearly_returns']
        if not yearly_df.empty:
            header_df = pd.DataFrame({'【每年收益统计】': ['']})
            header_df.to_excel(writer, sheet_name='策略概览', startrow=start_row, index=False)
            start_row += 1
            yearly_df.to_excel(writer, sheet_name='策略概览', startrow=start_row, index=False)
            start_row += len(yearly_df) + 3

        # ========== Sheet 2: 净值序列 ==========
        nav_df = create_nav_df(
            backtest_result['strategy_nav'],
            backtest_result['benchmark_nav'],
            backtest_result['excess_nav']
        )
        nav_df.to_excel(writer, sheet_name='净值序列', index=True)

        # ========== Sheet 3: 调仓详情 ==========
        if 'rebalance_details' in backtest_result:
            rebalance_df = backtest_result['rebalance_details']
            rebalance_df.to_excel(writer, sheet_name='调仓详情', index=False)

        # ========== Sheet 4: 历史持仓 ==========
        holdings_df = create_holdings_df(backtest_result['holdings_history'])
        holdings_df.to_excel(writer, sheet_name='历史持仓', index=False)

        # ========== Sheet 5: 原始信号 ==========
        signal_df = backtest_result['signal_df'].copy()
        signal_df.columns = [clean_industry_name(col) for col in signal_df.columns]
        signal_df.index = signal_df.index.strftime('%Y-%m-%d')
        signal_df.index.name = '日期'
        signal_df.to_excel(writer, sheet_name='原始信号', index=True)

    if verbose:
        print(f"导出完成！")
        print(f"\nExcel 包含以下 Sheet:")
        print(f"  1. 策略概览 - 因子说明、绩效指标、每年收益")
        print(f"  2. 净值序列 - 策略/基准/超额净值")
        print(f"  3. 调仓详情 - 每次调仓的收益和持仓")
        print(f"  4. 历史持仓 - 每期持仓行业列表")
        print(f"  5. 原始信号 - 周频 0/1 信号矩阵")

    return output_file


# ============================================================================
# 第九部分: 一键运行主函数 (含数据加载和 Excel 导出)
# ============================================================================

def main(
    data_file: str = DEFAULT_CACHE_FILE,
    output_file: str = None,
    bsadf_min_window: int = 52,
    bsadf_compare_window: int = 40,
    bocd_hazard: float = 0.01
) -> dict:
    """
    一键运行: 加载数据 -> 计算因子 -> 回测 -> 导出 Excel

    参数:
        data_file: str, 数据文件路径
        output_file: str, 输出 Excel 文件路径
        bsadf_min_window: int, BSADF 最小窗口 (周)，默认 52
        bsadf_compare_window: int, BSADF 动态阈值比较窗口 (周)，默认 40
        bocd_hazard: float, BOCD 变点先验概率，默认 0.01

    返回:
        dict: 回测结果
    """
    print("=" * 60)
    print("正向泡沫行业轮动因子回测")
    print("基于兴业证券研报《如何结合行业轮动的长短信号？》")
    print("=" * 60)

    # Step 1: 加载数据
    print("\n[步骤 1/4] 加载数据...")
    prices_df, amount_df = load_data(data_file, verbose=True)

    # Step 2: 计算因子信号
    print("\n[步骤 2/4] 计算因子信号...")
    signal_df = compute_positive_bubble_signal(
        prices_df, amount_df,
        bsadf_min_window=bsadf_min_window,
        bsadf_compare_window=bsadf_compare_window,
        bocd_hazard=bocd_hazard,
        verbose=True
    )

    # Step 3: 运行回测
    print("\n[步骤 3/4] 运行回测...")
    result = backtest_positive_bubble(signal_df, prices_df, verbose=True)

    # 打印绩效指标
    print("\n" + "=" * 60)
    print("绩效指标")
    print("=" * 60)
    for key, value in result['performance_metrics'].items():
        print(f"  {key}: {value}")

    # 打印每年收益
    print("\n" + "=" * 60)
    print("每年收益统计")
    print("=" * 60)
    print(result['yearly_returns'].to_string(index=False))

    # 打印最近选出的行业
    print_selected_industries(result['signal_df'], n_recent=10)

    # Step 4: 导出 Excel
    print("\n[步骤 4/4] 导出结果...")
    output_path = export_to_excel(result, output_file, verbose=True)

    print("\n" + "=" * 60)
    print(f"回测完成！结果已保存至: {output_path}")
    print("=" * 60)

    return result


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    # 直接运行时，执行完整流程
    result = main()
