"""
单因子测试脚本

职责：
1. 配置：指定要测试的因子名称和截止日期
2. 调用组装：调用 factors_analysis 模块完成分析流程

所有计算逻辑在 factors_analysis 模块中实现。
"""

import data_loader
import factors_analysis as fa

# ============================================================
# 配置区域 - 只需修改这里
# ============================================================

# 要测试的因子名称（必须在 factors_analysis.FACTOR_CONFIG 中定义）
FACTOR_NAME = 'momentum_lead_lag_enhanced'

# 截止日期（None 表示使用最新数据）
END_DATE = '2024-12-31'     # 格式: '2022-11-30' 或 None

# ============================================================
# 调用组装 - 无需修改
# ============================================================

def run_analysis(factor_name, end_date=None):
    """运行单因子分析"""
    import time
    start_time = time.time()

    print("=" * 60)
    print(f"单因子测试: {factor_name}")
    print("=" * 60)

    # 检查因子是否存在
    if factor_name not in fa.list_factors():
        print(f"\n错误: 因子 '{factor_name}' 不存在!")
        print(f"可用因子列表: {fa.list_factors()}")
        return

    # 获取因子配置
    factor_config = fa.FACTOR_CONFIG[factor_name]
    requires_constituent = factor_config.get('requires_constituent', False)
    windows = fa.get_factor_windows(factor_name)

    # 显示配置信息
    print(f"\n配置信息:")
    print(f"  因子名称: {factor_name}")
    print(f"  回测年限: {fa.DEFAULT_BACKTEST_YEARS}年")
    print(f"  截止日期: {end_date or '最新数据'}")
    print(f"  回看窗口: {windows}")
    print(f"  需要成分股数据: {requires_constituent}")

    # 加载数据
    print(f"\n正在加载数据...")
    data = fa.DataContainer(
        data_loader.DEFAULT_CACHE_FILE,
        end_date=end_date,
        backtest_years=fa.DEFAULT_BACKTEST_YEARS,
        load_constituent=requires_constituent
    )

    print("\n调仓方式: 每月最后一个交易日调仓")

    # 分析因子
    print("\n开始分析因子...")
    all_results = fa.analyze_single_factor(factor_name, data, windows=windows)

    # 计算耗时
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    duration_str = f"{hours}h{minutes:02d}m" if hours > 0 else f"{minutes}m"

    # 导出到Excel（时间戳和耗时在导出时生成）
    print("\n正在导出到Excel...")
    output_file = fa.get_output_path(factor_name, duration_str)
    fa.export_to_excel(all_results, output_file, windows)

    # 格式化Excel报告
    print("\n正在格式化Excel报告...")
    fa.format_excel_report(output_file)

    print(f"\n分析完成！报告已保存至: {output_file}")
    print(f"总耗时: {hours}小时{minutes}分钟" if hours > 0 else f"总耗时: {minutes}分钟")

    # 打印关键指标
    print("\n" + "=" * 60)
    print("关键指标汇总")
    print("=" * 60)

    if factor_name in all_results:
        for window in windows:
            result = all_results[factor_name].get(window)
            if result:
                print(f"\n{window}日窗口:")
                print(f"  IC均值: {result['ic_mean']:.4f}")
                print(f"  ICIR: {result['icir']:.4f}")
                print(f"  IC胜率: {result['ic_win_rate']:.4f}")
                if 'G5' in result['excess_metrics']:
                    g5 = result['excess_metrics']['G5']
                    print(f"  G5年化收益率: {g5['年化收益率(%)']:.2f}%")
                    print(f"  G5超额年化收益率: {g5['超额年化收益率(%)']:.2f}%")
                    print(f"  G5夏普比率: {g5['夏普比率']:.2f}")


if __name__ == "__main__":
    run_analysis(FACTOR_NAME, END_DATE)

