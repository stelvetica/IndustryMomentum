"""
测试不同train_periods对Lasso因子IC的影响
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LassoLarsIC
from scipy import stats
import warnings
import data_loader

def test_lasso_with_train_periods(train_periods):
    """用指定的train_periods测试Lasso IC"""
    
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
    
    min_train_samples = 10
    
    predictions = []
    actuals = []
    
    for s_pred in range(1 + min_train_samples, T - 1):
        train_s = list(range(1, s_pred))
        if train_periods is not None and len(train_s) > train_periods:
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
        
        if len(X_train_list) < min_train_samples:
            continue
        
        X_train = np.array(X_train_list)
        X_current = excess_monthly.loc[months[s_pred]].values.reshape(1, -1)
        
        if np.any(np.isnan(X_current)):
            continue
        
        pred_row = {}
        actual_row = {}
        
        for ind in industries:
            y_train = np.array(y_train_dict[ind])
            valid_mask = ~np.isnan(y_train)
            
            if valid_mask.sum() < min_train_samples:
                continue
            
            X_valid = X_train[valid_mask]
            y_valid = y_train[valid_mask]
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    lasso = LassoLarsIC(criterion='aic')
                    lasso.fit(X_valid, y_valid)
                    pred = lasso.predict(X_current)[0]
                    pred_row[ind] = pred
            except:
                continue
        
        if s_pred + 1 < T:
            actual_ret = excess_monthly.loc[months[s_pred + 1]]
            for ind in pred_row.keys():
                if ind in actual_ret.index and not np.isnan(actual_ret[ind]):
                    actual_row[ind] = actual_ret[ind]
        
        if len(pred_row) > 5 and len(actual_row) > 5:
            predictions.append(pred_row)
            actuals.append(actual_row)
    
    # 计算IC
    ic_list = []
    for pred_row, actual_row in zip(predictions, actuals):
        common_inds = set(pred_row.keys()) & set(actual_row.keys())
        if len(common_inds) < 5:
            continue
        
        pred_vals = [pred_row[ind] for ind in common_inds]
        actual_vals = [actual_row[ind] for ind in common_inds]
        
        ic, _ = stats.spearmanr(pred_vals, actual_vals)
        if not np.isnan(ic):
            ic_list.append(ic)
    
    if len(ic_list) > 0:
        ic_mean = np.mean(ic_list)
        icir = ic_mean / np.std(ic_list) if np.std(ic_list) > 0 else 0
        return ic_mean, icir, len(ic_list)
    else:
        return np.nan, np.nan, 0


def main():
    print("=" * 80)
    print("测试不同train_periods对Lasso因子IC的影响")
    print("=" * 80)
    
    # 测试不同的train_periods设置
    test_periods = [
        (30, "30个月"),
        (60, "60个月（当前设置）"),
        (90, "90个月"),
        (120, "120个月"),
        (None, "全部历史（研报可能用法）"),
    ]
    
    results = []
    
    for periods, desc in test_periods:
        print(f"\n测试 train_periods={periods} ({desc})...")
        ic_mean, icir, n_months = test_lasso_with_train_periods(periods)
        results.append({
            'train_periods': periods if periods else '全部',
            'description': desc,
            'IC均值': ic_mean,
            'ICIR': icir,
            '有效月数': n_months
        })
        print(f"  IC均值: {ic_mean:.4f}, ICIR: {icir:.4f}, 有效月数: {n_months}")
    
    print("\n" + "=" * 80)
    print("汇总结果:")
    print("=" * 80)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # 结论
    print("\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    best = max(results, key=lambda x: x['IC均值'] if not np.isnan(x['IC均值']) else -999)
    print(f"最优train_periods: {best['train_periods']} ({best['description']})")
    print(f"最优IC均值: {best['IC均值']:.4f}")


if __name__ == "__main__":
    main()
