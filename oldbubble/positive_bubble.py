"""
正向泡沫因子（momentum_positive_bubble）专用回测模块

基于兴业证券研报《如何结合行业轮动的长短信号？》复现

该因子与其他因子不同：
1. 返回0/1信号，而非连续因子值
2. 周频数据（每周五）
3. 信号=1表示持有，信号=0表示不持有

核心算法：
- BSADF (Backward Sup ADF)：泡沫检测，捕捉价格/成交额的爆炸性行为
- BOCD (Bayesian Online Changepoint Detection)：变点检测，捕捉趋势起点

回测逻辑：
- 每周五根据信号选择持仓行业（信号=1的行业）
- 等权持有所有信号=1的行业
- 计算策略收益、基准收益、超额收益

信号合成公式（严格按照研报）：
    Buy_Signal = (Signal_BSADF OR Signal_BOCD) AND (Return_t > 0)
"""

import pandas as pd
import numpy as np
import warnings
from scipy import stats
from numba import jit
warnings.filterwarnings('ignore')

# 导入数据加载模块
from data_loader import (
    load_price_df, load_volume_df, DEFAULT_CACHE_FILE
)


def clean_industry_name(name):
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
# 快速OLS回归模块 - 使用numpy矩阵运算实现
# ============================================================================

@jit(nopython=True)
def _fast_ols_t_stat_numba(y):
    """
    使用numba加速的OLS回归t统计量计算（内部函数）
    
    参数:
        y: np.ndarray, 对数价格或对数成交额序列（float64）
    
    返回:
        float, β系数的t统计量
    """
    n = len(y)
    
    # 至少需要3个数据点才能进行回归
    if n < 3:
        return np.nan
    
    # 构造因变量 Y = Δy[1:] = y[1:] - y[:-1]
    delta_y = np.diff(y)  # 长度为 n-1
    Y = delta_y
    
    # 构造自变量
    y_lag = y[:-1]  # 长度为 n-1
    n_obs = len(Y)
    
    # 手动计算 (X'X) 和 (X'Y)，避免创建完整的X矩阵
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
    beta_1 = inv_10 * sum_Y + inv_11 * sum_yY  # y_{t-1}的系数
    
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


def fast_ols_t_stat(y):
    """
    使用numpy矩阵运算计算ADF回归的t统计量（Lag=0简化版）
    
    回归方程: Δy_t = α + β*y_{t-1} + ε_t
    
    参数:
        y: np.ndarray 或 pd.Series, 对数价格或对数成交额序列
    
    返回:
        float, β系数的t统计量
    
    实现细节:
        1. 构造 Y = Δy[1:] (因变量)
        2. 构造 X = [1, y[:-1]] (自变量矩阵，含截距项)
        3. 计算 β̂ = (X'X)^(-1)X'Y
        4. 计算残差 ε = Y - Xβ̂
        5. 计算 σ² = ε'ε / (n-2)
        6. 计算 SE_β = sqrt(σ² * (X'X)^(-1)_11)
        7. 返回 t = β̂_1 / SE_β
    
    注意：
        此函数使用numba加速版本进行计算
    """
    # 转换为numpy数组
    if isinstance(y, pd.Series):
        y = y.values
    
    y = np.asarray(y, dtype=np.float64)
    
    # 使用numba加速版本
    return _fast_ols_t_stat_numba(y)


# ============================================================================
# BOCD (贝叶斯在线变点检测) 模块
# ============================================================================

class GaussianBOCD:
    """
    高斯贝叶斯在线变点检测 (Gaussian Bayesian Online Changepoint Detection)

    基于 Adams & MacKay (2007) 的算法实现，假设数据服从正态分布。
    使用 Normal-Inverse-Gamma 共轭先验，预测分布为 Student-t 分布。

    参数:
        hazard: float, 变点发生的先验概率 (1/expected_run_length)，默认0.01 (1/100)
        mu0: float, 均值的先验均值，默认0
        kappa0: float, 均值先验的强度参数，默认1
        alpha0: float, 方差先验的形状参数，默认50（研报建议设为较大值以稳定方差估计）
        beta0: float, 方差先验的尺度参数，默认0.01（对应周收益率的典型方差量级）
    """

    def __init__(self, hazard=0.01, mu0=0.0, kappa0=1.0, alpha0=50.0, beta0=0.01):
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
        
        参数:
            x: float, 观测值
            mu: float, 位置参数
            kappa: float, 精度参数
            alpha: float, 形状参数
            beta: float, 尺度参数
        
        返回:
            float, 概率密度值
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
    
    def get_changepoint_prob(self):
        """
        获取当前时刻的变点概率 P(r_t=0)
        
        根据研报要求，变点概率定义为 run_length = 0 的概率，
        即当前时刻刚刚发生结构性变化的概率。
        
        返回:
            float, P(r_t=0)，即 run_length_probs[0]
        """
        return self.run_length_probs[0]
    
    def reset(self):
        """重置检测器状态"""
        self.run_length_probs = np.array([1.0])
        self.mu_params = [self.mu0]
        self.kappa_params = [self.kappa0]
        self.alpha_params = [self.alpha0]
        self.beta_params = [self.beta0]


# ============================================================================
# 正向泡沫因子计算模块
# ============================================================================

def momentum_positive_bubble(prices_df, amount_df,
                              bsadf_min_window=52,
                              bsadf_compare_window=40,
                              bocd_hazard=0.01):
    """
    正向泡沫行业轮动因子（兴业证券"理性泡沫"策略）
    
    出处：兴业证券《如何结合行业轮动的长短信号？》
    
    理念：利用"理性泡沫"理论，通过 BSADF（量价爆炸检测）和 BOCD（结构突变检测）
          捕捉行业的起涨点。
    
    构造：
        1. 数据预处理：日频转周频（W-FRI）
        2. BSADF信号：价格和成交额的泡沫检测，双重确认
        3. BOCD信号：收益率变点检测，捕捉趋势起点
        4. 信号合成：Buy_Signal = (BSADF OR BOCD) AND (Return > 0)
    
    参数:
        prices_df: pd.DataFrame, 日频收盘价数据 (index=日期, columns=行业)
        amount_df: pd.DataFrame, 日频成交额数据 (index=日期, columns=行业)
        bsadf_min_window: int, BSADF最小窗口（周），默认52（约一年）
        bsadf_compare_window: int, BSADF动态阈值比较窗口（周），默认40
        bocd_hazard: float, BOCD变点先验概率，默认0.01 (1/100)
    
    返回:
        pd.DataFrame, 周频因子值 (0/1信号)，index为周五日期，columns为行业名称
    """
    
    # ========== Phase 1: 数据预处理 - 日频转周频 ==========
    
    def resample_to_weekly(prices, amount):
        """
        将日频数据重采样为周频
        - 收盘价：取当周最后一个交易日的值
        - 成交额：取当周所有交易日的总和
        
        参数:
            prices: pd.DataFrame, 日频收盘价
            amount: pd.DataFrame, 日频成交额
        
        返回:
            weekly_close: pd.DataFrame, 周频收盘价
            weekly_amount: pd.DataFrame, 周频成交额
            weekly_returns: pd.DataFrame, 周频收益率
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
        
        使用 fast_ols_t_stat 函数进行高性能计算。
        
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
        
        # 转换为numpy数组以提高性能
        values = series.values
        
        for t in range(min_window, n):
            max_adf_stat = -np.inf
            
            # 递归窗口：从 [0, t] 到 [t-min_window, t]
            for s in range(0, t - min_window + 1):
                window_data = values[s:t+1]
                
                # 检查是否有NaN
                if np.any(np.isnan(window_data)):
                    continue
                
                if len(window_data) < min_window:
                    continue
                
                # 使用快速OLS计算t统计量
                adf_stat = fast_ols_t_stat(window_data)
                
                if not np.isnan(adf_stat) and adf_stat > max_adf_stat:
                    max_adf_stat = adf_stat
            
            if max_adf_stat > -np.inf:
                bsadf_stats.iloc[t] = max_adf_stat
        
        return bsadf_stats
    
    def get_bsadf_signal(weekly_close, weekly_amount, min_window, compare_window):
        """
        计算价量双重 BSADF 信号
        
        信号逻辑：
            Signal = 1 当且仅当:
                (BSADF_Price > 40周Price中位数) AND 
                (BSADF_Volume > 40周Volume中位数)
        
        参数:
            weekly_close: pd.DataFrame, 周频收盘价
            weekly_amount: pd.DataFrame, 周频成交额
            min_window: int, BSADF最小窗口
            compare_window: int, 动态阈值比较窗口
        
        返回:
            pd.DataFrame, 0/1信号矩阵
        """
        industries = weekly_close.columns
        signal_df = pd.DataFrame(index=weekly_close.index, columns=industries, dtype=float)
        
        total_industries = len(industries)
        for idx, col in enumerate(industries):
            if (idx + 1) % 5 == 0:
                print(f"    BSADF计算进度: {idx+1}/{total_industries} 行业")
            
            # 计算对数序列
            price_series = weekly_close[col].dropna()
            amount_series = weekly_amount[col].dropna()
            
            if len(price_series) < min_window + compare_window:
                continue
            
            log_close = np.log(price_series)
            log_amount = np.log(amount_series + 1)  # +1 避免 log(0)
            
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
    
    def get_bocd_signal(weekly_returns, hazard):
        """
        计算 BOCD 变点检测信号
        
        信号逻辑（严格按照研报）：
            Signal = 1 当且仅当:
                最近3周变点概率单调递增 (P_t > P_{t-1} > P_{t-2})
        
        注意：
            - 变点概率使用 P(r_t=0)，而非 P(r_t <= threshold)
            - 当周收益率>0的条件移至信号合成阶段
        
        参数:
            weekly_returns: pd.DataFrame, 周频收益率
            hazard: float, 变点先验概率
        
        返回:
            pd.DataFrame, 0/1信号矩阵
        """
        industries = weekly_returns.columns
        signal_df = pd.DataFrame(index=weekly_returns.index, columns=industries, dtype=float)
        
        for col in industries:
            returns = weekly_returns[col].dropna()
            
            if len(returns) < 10:
                continue
            
            # 初始化 BOCD 检测器
            detector = GaussianBOCD(hazard=hazard)
            
            probs = []
            for t, ret in enumerate(returns):
                detector.update(ret)
                # 使用 P(r_t=0) 作为变点概率
                prob = detector.get_changepoint_prob()
                probs.append(prob)
            
            # 生成信号：最近3周概率单调递增
            for t in range(2, len(returns)):
                t_date = returns.index[t]
                
                # 条件：最近3周概率单调递增
                prob_t = probs[t]
                prob_t1 = probs[t-1]
                prob_t2 = probs[t-2]
                
                monotonic_increase = (prob_t > prob_t1) and (prob_t1 > prob_t2)
                
                if monotonic_increase:
                    signal_df.loc[t_date, col] = 1
                else:
                    signal_df.loc[t_date, col] = 0
        
        return signal_df.fillna(0).astype(float)
    
    # ========== 主流程执行 ==========
    
    print("Phase 1: 数据预处理 - 日频转周频...")
    weekly_close, weekly_amount, weekly_returns = resample_to_weekly(prices_df, amount_df)
    print(f"    周频数据点数: {len(weekly_close)}")
    
    print("Phase 2: 计算 BSADF 信号...")
    signal_bsadf = get_bsadf_signal(weekly_close, weekly_amount,
                                     bsadf_min_window, bsadf_compare_window)
    
    print("Phase 3: 计算 BOCD 信号...")
    signal_bocd = get_bocd_signal(weekly_returns, bocd_hazard)
    
    print("Phase 4: 信号合成...")
    # 严格按照研报公式：Buy_Signal = (BSADF OR BOCD) AND (Return > 0)
    signal_union = ((signal_bsadf == 1) | (signal_bocd == 1)).astype(float)
    positive_return_mask = (weekly_returns > 0).astype(float)
    
    # 对齐索引
    common_index = signal_union.index.intersection(positive_return_mask.index)
    signal_union = signal_union.loc[common_index]
    positive_return_mask = positive_return_mask.loc[common_index]
    
    # 最终信号
    final_factor = (signal_union * positive_return_mask).fillna(0)
    
    print("信号计算完成！")
    
    return final_factor.astype(float)



# ============================================================================
# 回测模块
# ============================================================================

def run_positive_bubble_backtest(prices_df, amount_df, 
                                  bsadf_min_window=52,
                                  bsadf_compare_window=40,
                                  bocd_hazard=0.01):
    """
    运行正向泡沫因子回测
    
    参数:
        prices_df: pd.DataFrame, 日频收盘价数据 (index=日期, columns=行业)
        amount_df: pd.DataFrame, 日频成交额数据 (index=日期, columns=行业)
        bsadf_min_window: int, BSADF最小窗口（周），默认52
        bsadf_compare_window: int, BSADF动态阈值比较窗口（周），默认40
        bocd_hazard: float, BOCD变点先验概率，默认0.01 (1/100)
    
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
    print("=" * 60)
    print("正在计算正向泡沫因子信号...")
    print("=" * 60)
    
    # 计算因子信号（周频0/1信号）
    signal_df = momentum_positive_bubble(
        prices_df, amount_df,
        bsadf_min_window=bsadf_min_window,
        bsadf_compare_window=bsadf_compare_window,
        bocd_hazard=bocd_hazard
    )
    
    print(f"\n信号计算完成，周频数据点数: {len(signal_df)}")
    
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
    
    print("\n开始回测计算...")
    
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

    # ========== 计算每次调仓详情 ==========

    rebalance_details = calculate_rebalance_details(
        strategy_nav, benchmark_nav, holdings_history
    )

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


def calculate_performance_metrics(strategy_nav, benchmark_nav, excess_nav):
    """
    计算绩效指标（按研报要求格式）

    输出格式：年化收益率(%) 年化波动率(%) 最大回撤(%) 收益风险比 收益回撤比 月度胜率(%)

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

    # ========== 策略指标 ==========

    # 年化收益率
    annual_return = ((strategy_nav.iloc[-1] / strategy_nav.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0

    # 年化波动率
    volatility = strategy_returns.std() * np.sqrt(periods_per_year) * 100

    # 最大回撤
    cummax = strategy_nav.cummax()
    drawdown = (strategy_nav - cummax) / cummax
    max_drawdown = abs(drawdown.min() * 100)

    # 收益风险比 (夏普比率简化版，无风险利率=0)
    risk_return_ratio = annual_return / volatility if volatility > 0 else 0

    # 收益回撤比 (Calmar比率)
    return_drawdown_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

    # ========== 月度胜率计算 ==========
    # 将周频净值转换为月频
    monthly_strategy = strategy_nav.resample('M').last()
    monthly_benchmark = benchmark_nav.resample('M').last()

    monthly_strategy_ret = monthly_strategy.pct_change().dropna()
    monthly_benchmark_ret = monthly_benchmark.pct_change().dropna()

    # 月度胜率：策略月收益 > 基准月收益的月数占比
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


def calculate_yearly_returns(strategy_nav, benchmark_nav, start_year=2017):
    """
    计算每年的收益统计

    输出格式：每年的多头收益% 超额收益% 基准%

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

        # 多头收益（策略收益）
        strategy_return = (strategy_nav.loc[end_date] / strategy_nav.loc[start_date] - 1) * 100

        # 基准收益
        bench_return = (benchmark_nav.loc[end_date] / benchmark_nav.loc[start_date] - 1) * 100

        # 超额收益
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


def calculate_rebalance_details(strategy_nav, benchmark_nav, holdings_history):
    """
    计算每次调仓的详细信息

    输出格式：每次调仓的单位净值、多头收益、超额收益、基准、选出来的行业

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

        # 获取净值
        nav = strategy_nav.loc[date] if date in strategy_nav.index else np.nan
        bench_nav = benchmark_nav.loc[date] if date in benchmark_nav.index else np.nan

        # 计算收益（相对于上一期）
        if i > 0:
            prev_date = dates[i-1]
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



# ============================================================================
# 输出格式模块
# ============================================================================

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


def export_to_excel(backtest_result, output_file, sheet_name='正向泡沫因子回测'):
    """
    将回测结果导出到Excel

    参数:
        backtest_result: dict, 回测结果
        output_file: str, 输出文件路径
        sheet_name: str, 要写入的sheet名称
    """
    print(f"\n正在导出回测结果到: {output_file}")

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # ========== Sheet 1: 策略概览 ==========
        start_row = 0

        # 因子说明
        factor_doc = """
正向泡沫行业轮动因子（momentum_positive_bubble）

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

        # ========== Sheet 3: 调仓详情 ==========
        if 'rebalance_details' in backtest_result:
            rebalance_df = backtest_result['rebalance_details']
            rebalance_df.to_excel(writer, sheet_name='调仓详情', index=False)

        # ========== Sheet 4: 历史持仓 ==========
        holdings_df = create_holdings_df(backtest_result['holdings_history'])
        holdings_df.to_excel(writer, sheet_name='历史持仓', index=False)

        # ========== Sheet 5: 原始信号 ==========
        signal_df = backtest_result['signal_df'].copy()
        # 清理列名
        signal_df.columns = [clean_industry_name(col) for col in signal_df.columns]
        # 格式化日期索引
        signal_df.index = signal_df.index.strftime('%Y-%m-%d')
        signal_df.index.name = '日期'
        signal_df.to_excel(writer, sheet_name='原始信号', index=True)

    print(f"导出完成！")


def print_selected_industries(signal_df, n_recent=10):
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
# 主函数
# ============================================================================

def main():
    """
    主函数
    """
    print("=" * 60)
    print("正向泡沫因子（momentum_positive_bubble）回测")
    print("基于兴业证券研报《如何结合行业轮动的长短信号？》")
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
        bsadf_min_window=52,      # 研报要求：约一年（52周）
        bsadf_compare_window=40,  # 研报要求：40周滚动中位数
        bocd_hazard=0.01          # 研报建议：1/100
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
    
    # 打印最近选出的行业
    print_selected_industries(backtest_result['signal_df'], n_recent=10)

    # 导出到Excel
    output_file = 'positive_bubble_backtest_report.xlsx'
    export_to_excel(backtest_result, output_file)

    print("\n" + "=" * 60)
    print(f"回测完成！结果已保存至: {output_file}")
    print("=" * 60)

    return backtest_result


if __name__ == "__main__":
    main()

