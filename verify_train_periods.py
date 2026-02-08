"""
验证 factor_value.py 中统一的日到月映射逻辑
"""

# 导入全局常量和函数
from factor_value import WINDOW_TO_MONTHS, window_to_months

print("=" * 60)
print("验证统一的日到月映射逻辑")
print("=" * 60)

print(f"\n全局常量 WINDOW_TO_MONTHS:")
print(f"  {WINDOW_TO_MONTHS}")

test_windows = [20, 60, 120, 240, 480, 720]
expected_months = [1, 3, 6, 12, 24, 36]

print(f"\n{'传入window(日)':<15} {'期望月数':<10} {'实际映射':<10} {'是否正确':<10}")
print("-" * 50)

all_pass = True
for window, expected in zip(test_windows, expected_months):
    result = window_to_months(window)
    match = "✓" if result == expected else "✗"
    if result != expected:
        all_pass = False
    print(f"{window:<15} {expected:<10} {result:<10} {match:<10}")

# 测试非标准窗口（fallback逻辑）
print("\n" + "-" * 50)
print("测试非标准窗口（fallback逻辑: window // 20）:")
non_standard = [100, 200, 300]
for window in non_standard:
    result = window_to_months(window)
    expected = window // 20
    match = "✓" if result == expected else "✗"
    print(f"{window:<15} {expected:<10} {result:<10} {match:<10}")

print("\n" + "=" * 60)
print("使用该映射的因子:")
print("=" * 60)
print("""
1. momentum_cross_industry_lasso (Lasso因子)
   - train_periods = window_to_months(window)

2. momentum_industry_component (行业成分股动量因子)
   - window_months = window_to_months(window)

3. momentum_lead_lag_enhanced (龙头领先增强动量因子)
   - lookback_months = window_to_months(window)
""")

if all_pass:
    print("✓ 所有标准窗口映射正确！")
else:
    print("✗ 存在映射错误，请检查！")

