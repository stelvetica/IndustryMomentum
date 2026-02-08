"""
因子交集合成模块 (factor_intersection.py)

实现合成因子与泡沫因子的交集选股策略：
- 每月月末，合成因子选出Top N行业
- 同一天，泡沫因子给出信号=1的行业
- 取两者交集作为最终持仓

不涉及频率转换，保持各因子原有频率。
"""

import pandas as pd
import numpy as np
import warnings
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings('ignore')


def get_top_industries_by_factor(factor_df: pd.DataFrame, 
                                  date: pd.Timestamp, 
                                  top_n: int = 6) -> List[str]:
    """
    根据因子值选出Top N行业
    
    参数:
        factor_df: pd.DataFrame, 因子值 (index=日期, columns=行业)
        date: pd.Timestamp, 选股日期
        top_n: int, 选取行业数量
    
    返回:
        List[str]: Top N行业名称列表
    """
    if date not in factor_df.index:
        return []
    
    factor_values = factor_df.loc[date].dropna()
    if len(factor_values) == 0:
        return []
    
    # 按因子值降序排列，取前N个
    top_industries = factor_values.nlargest(top_n).index.tolist()
    return top_industries


def get_bubble_signal_industries(bubble_signal_df: pd.DataFrame, 
                                  date: pd.Timestamp) -> List[str]:
    """
    获取泡沫因子信号=1的行业
    
    参数:
        bubble_signal_df: pd.DataFrame, 泡沫信号 (index=日期, columns=行业, 值为0/1)
        date: pd.Timestamp, 查询日期
    
    返回:
        List[str]: 信号=1的行业名称列表
    """
    if date not in bubble_signal_df.index:
        return []
    
    signals = bubble_signal_df.loc[date]
    return signals[signals == 1].index.tolist()


def find_nearest_date(target_date: pd.Timestamp, 
                      available_dates: pd.DatetimeIndex, 
                      direction: str = 'backward') -> Optional[pd.Timestamp]:
    """
    找到最近的可用日期
    
    参数:
        target_date: pd.Timestamp, 目标日期
        available_dates: pd.DatetimeIndex, 可用日期列表
        direction: str, 'backward'向前找, 'forward'向后找
    
    返回:
        pd.Timestamp: 最近的可用日期，如果没有则返回None
    """
    if direction == 'backward':
        valid_dates = available_dates[available_dates <= target_date]
        return valid_dates[-1] if len(valid_dates) > 0 else None
    else:
        valid_dates = available_dates[available_dates >= target_date]
        return valid_dates[0] if len(valid_dates) > 0 else None


def compute_intersection_holdings(composite_factor_df: pd.DataFrame,
                                   bubble_signal_df: pd.DataFrame,
                                   rebalance_dates: pd.DatetimeIndex,
                                   top_n: int = 6,
                                   verbose: bool = True) -> Dict[pd.Timestamp, List[str]]:
    """
    计算合成因子与泡沫因子的交集持仓

    策略逻辑：
    1. 每个调仓日，合成因子选出Top N行业
    2. 同一天（或最近的周五），泡沫因子给出信号=1的行业
    3. 取两者交集作为最终持仓
    4. 交集为空则空仓（不做回退）

    参数:
        composite_factor_df: pd.DataFrame, 合成因子值 (月频，月末有值)
        bubble_signal_df: pd.DataFrame, 泡沫信号 (周频，周五有值，0/1)
        rebalance_dates: pd.DatetimeIndex, 调仓日期（通常是月末）
        top_n: int, 合成因子选取的行业数量，默认6
        verbose: bool, 是否打印详细信息

    返回:
        Dict[pd.Timestamp, List[str]]: {调仓日期: [持仓行业列表]}
    """
    holdings_history = {}
    bubble_dates = bubble_signal_df.index

    for date in rebalance_dates:
        # 1. 合成因子选出Top N
        composite_top = get_top_industries_by_factor(composite_factor_df, date, top_n)

        # 2. 找到最近的泡沫信号日期（向前找最近的周五）
        nearest_bubble_date = find_nearest_date(date, bubble_dates, 'backward')

        if nearest_bubble_date is not None:
            bubble_industries = get_bubble_signal_industries(bubble_signal_df, nearest_bubble_date)
        else:
            bubble_industries = []

        # 3. 取交集（不做回退，交集为空则空仓）
        intersection = list(set(composite_top) & set(bubble_industries))
        holdings_history[date] = intersection

        # 打印调仓信息
        if verbose:
            print(f"{date.strftime('%Y-%m-%d')}: "
                  f"合成Top={len(composite_top)}个, "
                  f"泡沫信号={len(bubble_industries)}个, "
                  f"交集={len(intersection)}个")

    return holdings_history


def get_monthly_rebalance_dates(prices_df: pd.DataFrame) -> pd.DatetimeIndex:
    """
    获取每月最后一个交易日作为调仓日期

    参数:
        prices_df: pd.DataFrame, 价格数据（用于获取交易日历）

    返回:
        pd.DatetimeIndex: 月末调仓日期
    """
    trade_dates = prices_df.index
    monthly_last = trade_dates.to_series().groupby(
        [trade_dates.year, trade_dates.month]
    ).apply(lambda x: x.iloc[-1])
    return pd.DatetimeIndex(monthly_last.values)


def run_intersection_backtest(composite_factor_df: pd.DataFrame,
                               bubble_signal_df: pd.DataFrame,
                               prices_df: pd.DataFrame,
                               top_n: int = 6,
                               verbose: bool = True) -> Dict:
    """
    运行交集策略回测

    参数:
        composite_factor_df: pd.DataFrame, 合成因子值
        bubble_signal_df: pd.DataFrame, 泡沫信号
        prices_df: pd.DataFrame, 价格数据
        top_n: int, 合成因子选取的行业数量
        verbose: bool, 是否打印详细信息

    返回:
        Dict: 回测结果
    """
    # 获取调仓日期
    rebalance_dates = get_monthly_rebalance_dates(prices_df)

    # 过滤有效日期（因子和泡沫信号都有数据的日期）
    valid_dates = rebalance_dates[
        (rebalance_dates >= composite_factor_df.index.min()) &
        (rebalance_dates <= composite_factor_df.index.max()) &
        (rebalance_dates >= bubble_signal_df.index.min())
    ]

    if verbose:
        print(f"回测期间: {valid_dates[0].strftime('%Y-%m-%d')} 至 {valid_dates[-1].strftime('%Y-%m-%d')}")
        print(f"调仓次数: {len(valid_dates)}")
        print("=" * 60)

    # 计算持仓
    holdings_history = compute_intersection_holdings(
        composite_factor_df, bubble_signal_df, valid_dates,
        top_n, verbose
    )

    # 计算净值
    nav_series = calculate_strategy_nav(holdings_history, prices_df)
    benchmark_nav = calculate_benchmark_nav(prices_df, valid_dates)

    # 计算统计指标
    stats = calculate_performance_stats(nav_series, benchmark_nav)

    return {
        'holdings_history': holdings_history,
        'nav_series': nav_series,
        'benchmark_nav': benchmark_nav,
        'stats': stats,
        'rebalance_dates': valid_dates
    }


def calculate_strategy_nav(holdings_history: Dict[pd.Timestamp, List[str]],
                            prices_df: pd.DataFrame) -> pd.Series:
    """
    计算策略净值

    参数:
        holdings_history: Dict, 持仓历史
        prices_df: pd.DataFrame, 价格数据

    返回:
        pd.Series: 策略净值序列
    """
    rebalance_dates = sorted(holdings_history.keys())
    nav = pd.Series(index=prices_df.index, dtype=float)
    nav.iloc[0] = 1.0

    current_nav = 1.0

    for i, date in enumerate(rebalance_dates):
        holdings = holdings_history[date]

        if len(holdings) == 0:
            # 空仓，净值不变
            if i + 1 < len(rebalance_dates):
                next_date = rebalance_dates[i + 1]
                period_mask = (prices_df.index > date) & (prices_df.index <= next_date)
            else:
                period_mask = prices_df.index > date
            nav.loc[period_mask] = current_nav
            continue

        # 计算持仓期收益
        if i + 1 < len(rebalance_dates):
            next_date = rebalance_dates[i + 1]
        else:
            next_date = prices_df.index[-1]

        # 等权持有
        period_returns = prices_df.loc[date:next_date, holdings].pct_change()
        period_returns = period_returns.iloc[1:]  # 去掉第一行NaN

        if len(period_returns) > 0:
            daily_returns = period_returns.mean(axis=1)  # 等权平均
            for d in daily_returns.index:
                current_nav *= (1 + daily_returns.loc[d])
                nav.loc[d] = current_nav

    # 前向填充
    nav = nav.ffill()
    nav = nav.fillna(1.0)

    return nav


def calculate_benchmark_nav(prices_df: pd.DataFrame,
                             rebalance_dates: pd.DatetimeIndex) -> pd.Series:
    """
    计算基准净值（行业等权）

    参数:
        prices_df: pd.DataFrame, 价格数据
        rebalance_dates: pd.DatetimeIndex, 调仓日期

    返回:
        pd.Series: 基准净值序列
    """
    start_date = rebalance_dates[0]
    prices_aligned = prices_df.loc[start_date:]

    # 等权日收益率
    daily_returns = prices_aligned.pct_change().mean(axis=1)
    daily_returns = daily_returns.fillna(0)

    # 累计净值
    nav = (1 + daily_returns).cumprod()
    nav.iloc[0] = 1.0

    return nav


def calculate_performance_stats(strategy_nav: pd.Series,
                                 benchmark_nav: pd.Series) -> Dict:
    """
    计算绩效统计指标

    参数:
        strategy_nav: pd.Series, 策略净值
        benchmark_nav: pd.Series, 基准净值

    返回:
        Dict: 绩效指标
    """
    # 对齐索引
    common_index = strategy_nav.index.intersection(benchmark_nav.index)
    strategy_nav = strategy_nav.loc[common_index]
    benchmark_nav = benchmark_nav.loc[common_index]

    # 计算收益率
    strategy_returns = strategy_nav.pct_change().dropna()
    benchmark_returns = benchmark_nav.pct_change().dropna()
    excess_returns = strategy_returns - benchmark_returns

    # 年化天数
    years = (common_index[-1] - common_index[0]).days / 365.25
    trading_days = len(strategy_returns)

    # 总收益
    total_return = strategy_nav.iloc[-1] / strategy_nav.iloc[0] - 1
    benchmark_total_return = benchmark_nav.iloc[-1] / benchmark_nav.iloc[0] - 1
    excess_total_return = total_return - benchmark_total_return

    # 年化收益
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    benchmark_annual_return = (1 + benchmark_total_return) ** (1 / years) - 1 if years > 0 else 0
    excess_annual_return = annual_return - benchmark_annual_return

    # 年化波动率
    annual_vol = strategy_returns.std() * np.sqrt(252)
    excess_vol = excess_returns.std() * np.sqrt(252)

    # 夏普比率（假设无风险利率为0）
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    # 信息比率
    ir = excess_annual_return / excess_vol if excess_vol > 0 else 0

    # 最大回撤
    cummax = strategy_nav.cummax()
    drawdown = (strategy_nav - cummax) / cummax
    max_drawdown = drawdown.min()

    # 超额最大回撤
    excess_nav = strategy_nav / benchmark_nav
    excess_cummax = excess_nav.cummax()
    excess_drawdown = (excess_nav - excess_cummax) / excess_cummax
    excess_max_drawdown = excess_drawdown.min()

    # 胜率
    win_rate = (excess_returns > 0).sum() / len(excess_returns) if len(excess_returns) > 0 else 0

    return {
        '总收益': f"{total_return:.2%}",
        '基准总收益': f"{benchmark_total_return:.2%}",
        '超额总收益': f"{excess_total_return:.2%}",
        '年化收益': f"{annual_return:.2%}",
        '基准年化收益': f"{benchmark_annual_return:.2%}",
        '超额年化收益': f"{excess_annual_return:.2%}",
        '年化波动率': f"{annual_vol:.2%}",
        '夏普比率': f"{sharpe:.2f}",
        '信息比率': f"{ir:.2f}",
        '最大回撤': f"{max_drawdown:.2%}",
        '超额最大回撤': f"{excess_max_drawdown:.2%}",
        '日胜率': f"{win_rate:.2%}",
        '回测年数': f"{years:.1f}年",
        '交易日数': trading_days
    }


# ============================================================================
# Excel导出功能
# ============================================================================

def export_to_excel(result: Dict,
                    composite_factor_df: pd.DataFrame,
                    bubble_signal_df: pd.DataFrame,
                    output_file: str = None,
                    verbose: bool = True) -> str:
    """
    将回测结果导出到Excel

    参数:
        result: Dict, 回测结果
        composite_factor_df: pd.DataFrame, 合成因子值
        bubble_signal_df: pd.DataFrame, 泡沫信号
        output_file: str, 输出文件路径
        verbose: bool, 是否打印信息

    返回:
        str: 输出文件路径
    """
    from datetime import datetime

    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"factor_intersection_backtest_{timestamp}.xlsx"

    if verbose:
        print(f"\n正在导出回测结果到: {output_file}")

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # ========== Sheet 1: 策略概览 ==========
        overview_data = []
        overview_data.append({'项目': '【策略说明】', '内容': ''})
        overview_data.append({'项目': '策略名称', '内容': '因子交集策略'})
        overview_data.append({'项目': '策略逻辑', '内容': '合成因子Top N ∩ 泡沫信号=1 → 最终持仓'})
        overview_data.append({'项目': '调仓频率', '内容': '月频（每月最后一个交易日）'})
        overview_data.append({'项目': '', '内容': ''})
        overview_data.append({'项目': '【绩效指标】', '内容': ''})
        for key, value in result['stats'].items():
            overview_data.append({'项目': key, '内容': value})

        # 添加额外统计
        holdings_history = result['holdings_history']
        empty_count = sum(1 for h in holdings_history.values() if len(h) == 0)
        total_count = len(holdings_history)
        avg_holdings = np.mean([len(h) for h in holdings_history.values()])

        overview_data.append({'项目': '', '内容': ''})
        overview_data.append({'项目': '【持仓统计】', '内容': ''})
        overview_data.append({'项目': '空仓月份', '内容': f"{empty_count}/{total_count} ({empty_count/total_count*100:.1f}%)"})
        overview_data.append({'项目': '平均持仓', '内容': f"{avg_holdings:.1f}个行业"})

        overview_df = pd.DataFrame(overview_data)
        overview_df.to_excel(writer, sheet_name='策略概览', index=False)

        # ========== Sheet 2: 净值序列 ==========
        nav_df = pd.DataFrame({
            '日期': result['nav_series'].index,
            '策略净值': result['nav_series'].values,
            '基准净值': result['benchmark_nav'].reindex(result['nav_series'].index).values
        })
        nav_df['超额净值'] = nav_df['策略净值'] / nav_df['基准净值']
        nav_df['日期'] = nav_df['日期'].dt.strftime('%Y-%m-%d')
        nav_df.to_excel(writer, sheet_name='净值序列', index=False)

        # ========== Sheet 3: 历史持仓 ==========
        holdings_data = []
        for date, holdings in sorted(holdings_history.items()):
            holdings_data.append({
                '调仓日期': date.strftime('%Y-%m-%d'),
                '持仓数量': len(holdings),
                '持仓行业': ', '.join(holdings) if holdings else '空仓'
            })
        holdings_df = pd.DataFrame(holdings_data)
        holdings_df.to_excel(writer, sheet_name='历史持仓', index=False)

        # ========== Sheet 4: 调仓详情 ==========
        detail_data = []
        rebalance_dates = sorted(holdings_history.keys())

        for date in rebalance_dates:
            # 获取合成因子Top N
            composite_top = get_top_industries_by_factor(composite_factor_df, date, 6)

            # 获取泡沫信号行业
            bubble_dates = bubble_signal_df.index
            nearest_bubble_date = find_nearest_date(date, bubble_dates, 'backward')
            if nearest_bubble_date is not None:
                bubble_industries = get_bubble_signal_industries(bubble_signal_df, nearest_bubble_date)
            else:
                bubble_industries = []

            intersection = holdings_history[date]

            detail_data.append({
                '调仓日期': date.strftime('%Y-%m-%d'),
                '合成因子Top6': ', '.join(composite_top) if composite_top else '-',
                '泡沫信号行业': ', '.join(bubble_industries) if bubble_industries else '-',
                '交集持仓': ', '.join(intersection) if intersection else '空仓',
                '合成Top数': len(composite_top),
                '泡沫信号数': len(bubble_industries),
                '交集数': len(intersection)
            })

        detail_df = pd.DataFrame(detail_data)
        detail_df.to_excel(writer, sheet_name='调仓详情', index=False)

        # ========== Sheet 5: 泡沫信号矩阵 ==========
        bubble_signal_export = bubble_signal_df.copy()
        bubble_signal_export.index = bubble_signal_export.index.strftime('%Y-%m-%d')
        bubble_signal_export.index.name = '日期'
        bubble_signal_export.to_excel(writer, sheet_name='泡沫信号矩阵', index=True)

    if verbose:
        print(f"导出完成！")
        print(f"\nExcel 包含以下 Sheet:")
        print(f"  1. 策略概览 - 策略说明、绩效指标、持仓统计")
        print(f"  2. 净值序列 - 策略/基准/超额净值")
        print(f"  3. 历史持仓 - 每期持仓行业列表")
        print(f"  4. 调仓详情 - 每次调仓的因子选股和交集")
        print(f"  5. 泡沫信号矩阵 - 周频0/1信号")

    return output_file


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    import time
    from factor_value_backtest import DataContainer
    import factor_value as f
    from bubble.positive_bubble_factor_core import compute_positive_bubble_signal

    print("=" * 70)
    print("因子交集策略回测")
    print("合成因子 Top N ∩ 泡沫信号=1 → 最终持仓")
    print("=" * 70)

    start_time = time.time()

    # ========== 1. 加载数据 ==========
    print("\n[Step 1] 加载数据...")
    data = DataContainer(load_constituent=True)
    print(f"  价格数据: {data.prices_df.shape[0]}天 × {data.prices_df.shape[1]}个行业")
    print(f"  日期范围: {data.prices_df.index[0].strftime('%Y-%m-%d')} 至 {data.prices_df.index[-1].strftime('%Y-%m-%d')}")

    # ========== 2. 计算合成因子 ==========
    print("\n[Step 2] 计算合成因子（等权合成）...")
    composite_factor = f.momentum_synthesis_equal(
        data.prices_df,
        barra_factor_returns_df=data.barra_factor_returns_df,
        constituent_df=data.constituent_df,
        stock_price_df=data.stock_price_df,
        stock_mv_df=data.stock_mv_df,
        volume_df=data.volume_df,
        industry_code_df=data.industry_code_df,
        high_df=data.high_df,
        low_df=data.low_df
    )
    print(f"  合成因子计算完成: {composite_factor.shape[0]}天")

    # ========== 3. 计算泡沫信号 ==========
    print("\n[Step 3] 计算泡沫信号（BSADF + BOCD）...")
    bubble_signal = compute_positive_bubble_signal(
        data.prices_df, data.amount_df, verbose=True
    )
    print(f"  泡沫信号计算完成: {bubble_signal.shape[0]}周")

    # ========== 4. 运行交集回测 ==========
    print("\n[Step 4] 运行交集策略回测...")
    print("=" * 70)

    result = run_intersection_backtest(
        composite_factor_df=composite_factor,
        bubble_signal_df=bubble_signal,
        prices_df=data.prices_df,
        top_n=6,  # 合成因子选Top 6
        verbose=True
    )

    # ========== 5. 打印结果 ==========
    print("\n" + "=" * 70)
    print("绩效统计")
    print("=" * 70)
    for key, value in result['stats'].items():
        print(f"  {key}: {value}")

    # 统计空仓情况
    holdings_history = result['holdings_history']
    empty_count = sum(1 for h in holdings_history.values() if len(h) == 0)
    total_count = len(holdings_history)
    print(f"\n  空仓月份: {empty_count}/{total_count} ({empty_count/total_count*100:.1f}%)")

    # 平均持仓数
    avg_holdings = np.mean([len(h) for h in holdings_history.values()])
    print(f"  平均持仓: {avg_holdings:.1f}个行业")

    # ========== 6. 导出Excel ==========
    output_file = export_to_excel(
        result, composite_factor, bubble_signal,
        output_file=None,  # 自动生成文件名
        verbose=True
    )

    # ========== 7. 计算耗时 ==========
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\n总耗时: {minutes}分{seconds}秒")
    print(f"输出文件: {output_file}")
    print("=" * 70)

