"""
Lasso系数深度诊断
检查Lasso学到了什么样的系数结构，以及为什么IC为负
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LassoLarsIC, Ridge
from scipy import stats
import warnings
import data_loader

def diagnose_lasso_coefficients():
    """深度诊断Lasso系数"""
    
    print("=" * 80)
    print("Lasso系数深度诊断")
    print("=" * 80)
    
    # 加载数据
    prices_df = data_loader.load_price_df()
    
    # 构造月度数据
    trade_dates = prices_df.index
    monthly_last = trade_dates.to_series().groupby(
        [trade_dates.year, trade_dates.month]
    ).apply(lambda x: x.iloc[-1])
    monthly_last = pd.DatetimeIndex(monthly_last)
    monthly_close = prices_df.loc[monthly_last]
    
    # 计算月度收益和超额收益
    monthly_ret = monthly_close.pct_change()
    bench_monthly = monthly_ret.mean(axis=1)
    excess_monthly = monthly_ret.sub(bench_monthly, axis=0)
    
    industries = prices_df.columns.tolist()
    months = monthly_close.index.tolist()
    T = len(months)
    n_ind = len(industries)
    
    print(f"\n行业数量: {n_ind}")
    print(f"月份数量: {T}")
    
    # 取最近一个时点，详细看Lasso系数
    print("\n" + "=" * 80)
    print("1. 最近时点的Lasso系数分析")
    print("=" * 80)
    
    s_pred = T - 2  # 倒数第二个月，用于预测最后一个月
    factor_date = months[s_pred]
    print(f"\n预测时点: {factor_date.date()}")
    
    # 构建训练数据
    train_periods = 60
    train_s = list(range(1, s_pred))
    if len(train_s) > train_periods:
        train_s = train_s[-train_periods:]
    
    X_train_list = []
    y_train_dict = {ind: [] for ind in industries}
    
    for s in train_s:
        X_row = excess_monthly.loc[months[s]].values
        y_row = excess_monthly.loc[months[s + 1]]
        
        if np.all(np.isnan(X_row)) or y_row.isna().all():
            continue
        
        X_train_list.append(X_row)
        for ind in industries:
            y_val = y_row.get(ind, np.nan)
            y_train_dict[ind].append(y_val)
    
    X_train = np.array(X_train_list)
    X_current = excess_monthly.loc[months[s_pred]].values.reshape(1, -1)
    
    print(f"训练样本数: {len(X_train)}")
    print(f"特征数(行业数): {X_train.shape[1]}")
    
    # 对几个代表性行业进行详细分析
    sample_industries = industries[:5]  # 取前5个行业
    
    for ind in sample_industries:
        ind_idx = industries.index(ind)
        y_train = np.array(y_train_dict[ind])
        valid_mask = ~np.isnan(y_train)
        
        X_valid = X_train[valid_mask]
        y_valid = y_train[valid_mask]
        
        print(f"\n--- 行业: {ind} ---")
        print(f"有效训练样本: {valid_mask.sum()}")
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lasso = LassoLarsIC(criterion='aic')
            lasso.fit(X_valid, y_valid)
        
        # 系数分析
        coef = lasso.coef_
        nonzero_idx = np.where(coef != 0)[0]
        
        print(f"非零系数数量: {len(nonzero_idx)}")
        print(f"截距: {lasso.intercept_:.6f}")
        
        if len(nonzero_idx) > 0:
            print("非零系数:")
            for idx in nonzero_idx:
                print(f"  {industries[idx]}: {coef[idx]:.4f}")
        
        # 检查自己对自己的系数（对角线）
        self_coef = coef[ind_idx]
        print(f"自相关系数 (自己对自己): {self_coef:.4f}")
        
        # 预测值
        pred = lasso.predict(X_current)[0]
        print(f"预测值: {pred:.4f}")
        
        # 实际值（下个月）
        if s_pred + 1 < T:
            actual = excess_monthly.loc[months[s_pred + 1], ind]
            print(f"实际值: {actual:.4f}")
    
    # 2. 检查是否是因为Lasso系数符号问题
    print("\n" + "=" * 80)
    print("2. 所有行业Lasso系数符号统计")
    print("=" * 80)
    
    all_self_coefs = []  # 自相关系数
    all_nonzero_counts = []
    all_positive_coefs = []
    all_negative_coefs = []
    
    for ind in industries:
        y_train = np.array(y_train_dict[ind])
        valid_mask = ~np.isnan(y_train)
        
        if valid_mask.sum() < 10:
            continue
        
        X_valid = X_train[valid_mask]
        y_valid = y_train[valid_mask]
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lasso = LassoLarsIC(criterion='aic')
            lasso.fit(X_valid, y_valid)
        
        coef = lasso.coef_
        ind_idx = industries.index(ind)
        
        all_self_coefs.append(coef[ind_idx])
        all_nonzero_counts.append(np.sum(coef != 0))
        all_positive_coefs.append(np.sum(coef > 0))
        all_negative_coefs.append(np.sum(coef < 0))
    
    print(f"\n自相关系数统计:")
    print(f"  均值: {np.mean(all_self_coefs):.4f}")
    print(f"  >0的比例: {np.mean([c > 0 for c in all_self_coefs]):.2%}")
    print(f"  =0的比例: {np.mean([c == 0 for c in all_self_coefs]):.2%}")
    
    print(f"\n非零系数统计:")
    print(f"  平均非零数: {np.mean(all_nonzero_counts):.1f}")
    print(f"  平均正系数数: {np.mean(all_positive_coefs):.1f}")
    print(f"  平均负系数数: {np.mean(all_negative_coefs):.1f}")
    
    # 3. 对比：Ridge回归（无稀疏性约束）
    print("\n" + "=" * 80)
    print("3. 对比: Ridge回归 vs Lasso")
    print("=" * 80)
    
    ridge_preds = {}
    lasso_preds = {}
    actuals = {}
    
    for ind in industries:
        y_train = np.array(y_train_dict[ind])
        valid_mask = ~np.isnan(y_train)
        
        if valid_mask.sum() < 10:
            continue
        
        X_valid = X_train[valid_mask]
        y_valid = y_train[valid_mask]
        
        # Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_valid, y_valid)
        ridge_preds[ind] = ridge.predict(X_current)[0]
        
        # Lasso
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lasso = LassoLarsIC(criterion='aic')
            lasso.fit(X_valid, y_valid)
        lasso_preds[ind] = lasso.predict(X_current)[0]
        
        # Actual
        if s_pred + 1 < T:
            actuals[ind] = excess_monthly.loc[months[s_pred + 1], ind]
    
    common = set(ridge_preds.keys()) & set(lasso_preds.keys()) & set(actuals.keys())
    common = [ind for ind in common if not np.isnan(actuals[ind])]
    
    ridge_vals = [ridge_preds[ind] for ind in common]
    lasso_vals = [lasso_preds[ind] for ind in common]
    actual_vals = [actuals[ind] for ind in common]
    
    ridge_ic, _ = stats.spearmanr(ridge_vals, actual_vals)
    lasso_ic, _ = stats.spearmanr(lasso_vals, actual_vals)
    
    print(f"\n单期（{factor_date.date()}）IC对比:")
    print(f"  Ridge IC: {ridge_ic:.4f}")
    print(f"  Lasso IC: {lasso_ic:.4f}")
    
    # 4. 最关键：检查预测值和当月超额收益的相关性
    print("\n" + "=" * 80)
    print("4. 预测值 vs 当月超额收益 的相关性")
    print("=" * 80)
    
    current_excess = [excess_monthly.loc[months[s_pred], ind] for ind in common]
    
    pred_vs_current, _ = stats.spearmanr(lasso_vals, current_excess)
    current_vs_actual, _ = stats.spearmanr(current_excess, actual_vals)
    
    print(f"  Lasso预测 vs 当月超额: {pred_vs_current:.4f}")
    print(f"  当月超额 vs 下月实际: {current_vs_actual:.4f}")
    print(f"  Lasso预测 vs 下月实际: {lasso_ic:.4f}")
    
    if pred_vs_current < 0:
        print("\n  ⚠️ Lasso预测与当月超额收益负相关！")
        print("  这说明Lasso可能学到了\"反转\"效应而非\"动量\"效应")
    
    # 5. 检查Lasso的alpha值
    print("\n" + "=" * 80)
    print("5. LassoLarsIC选择的alpha值")
    print("=" * 80)
    
    alphas = []
    for ind in industries[:10]:
        y_train = np.array(y_train_dict[ind])
        valid_mask = ~np.isnan(y_train)
        
        if valid_mask.sum() < 10:
            continue
        
        X_valid = X_train[valid_mask]
        y_valid = y_train[valid_mask]
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lasso = LassoLarsIC(criterion='aic')
            lasso.fit(X_valid, y_valid)
            alphas.append(lasso.alpha_)
            print(f"  {ind}: alpha={lasso.alpha_:.6f}")
    
    print(f"\n平均alpha: {np.mean(alphas):.6f}")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)


if __name__ == "__main__":
    diagnose_lasso_coefficients()
