"""
对比不同回归方法在行业间动量因子上的表现
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LassoLarsIC, Ridge, Lasso, ElasticNet
from scipy import stats
import warnings
import data_loader

def test_regression_method(method_name, model_factory, train_periods=60):
    """用指定的回归方法测试IC"""
    
    prices_df = data_loader.load_price_df()
    
    trade_dates = prices_df.index
    monthly_last = trade_dates.to_series().groupby(
        [trade_dates.year, trade_dates.month]
    ).apply(lambda x: x.iloc[-1])
    monthly_last = pd.DatetimeIndex(monthly_last)
    monthly_close = prices_df.loc[monthly_last]
    
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
                    model = model_factory()
                    model.fit(X_valid, y_valid)
                    pred = model.predict(X_current)[0]
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
        ic_win_rate = np.mean([1 if ic > 0 else 0 for ic in ic_list])
        return ic_mean, icir, ic_win_rate, len(ic_list)
    else:
        return np.nan, np.nan, np.nan, 0


def test_simple_momentum():
    """简单动量基准（当月超额收益直接预测下月）"""
    prices_df = data_loader.load_price_df()
    
    trade_dates = prices_df.index
    monthly_last = trade_dates.to_series().groupby(
        [trade_dates.year, trade_dates.month]
    ).apply(lambda x: x.iloc[-1])
    monthly_last = pd.DatetimeIndex(monthly_last)
    monthly_close = prices_df.loc[monthly_last]
    
    monthly_ret = monthly_close.pct_change()
    bench_monthly = monthly_ret.mean(axis=1)
    excess_monthly = monthly_ret.sub(bench_monthly, axis=0)
    
    ic_list = []
    for i in range(1, len(monthly_last) - 1):
        curr_ret = excess_monthly.iloc[i]
        next_ret = excess_monthly.iloc[i + 1]
        
        valid_mask = curr_ret.notna() & next_ret.notna()
        if valid_mask.sum() < 5:
            continue
        
        ic, _ = stats.spearmanr(curr_ret[valid_mask], next_ret[valid_mask])
        if not np.isnan(ic):
            ic_list.append(ic)
    
    if len(ic_list) > 0:
        ic_mean = np.mean(ic_list)
        icir = ic_mean / np.std(ic_list) if np.std(ic_list) > 0 else 0
        ic_win_rate = np.mean([1 if ic > 0 else 0 for ic in ic_list])
        return ic_mean, icir, ic_win_rate, len(ic_list)
    return np.nan, np.nan, np.nan, 0


def main():
    print("=" * 80)
    print("对比不同回归方法在行业间动量因子上的表现")
    print("=" * 80)
    
    # 定义要测试的方法
    methods = [
        ("简单动量(当月超额→下月)", None, test_simple_momentum),
        ("LassoLarsIC(aic)", lambda: LassoLarsIC(criterion='aic'), None),
        ("LassoLarsIC(bic)", lambda: LassoLarsIC(criterion='bic'), None),
        ("Lasso(alpha=0.01)", lambda: Lasso(alpha=0.01, max_iter=10000), None),
        ("Lasso(alpha=0.001)", lambda: Lasso(alpha=0.001, max_iter=10000), None),
        ("Ridge(alpha=1.0)", lambda: Ridge(alpha=1.0), None),
        ("Ridge(alpha=10.0)", lambda: Ridge(alpha=10.0), None),
        ("ElasticNet(0.5)", lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000), None),
    ]
    
    results = []
    
    for method_name, model_factory, special_func in methods:
        print(f"\n测试 {method_name}...")
        
        if special_func:
            ic_mean, icir, win_rate, n_months = special_func()
        else:
            ic_mean, icir, win_rate, n_months = test_regression_method(method_name, model_factory)
        
        results.append({
            '方法': method_name,
            'IC均值': ic_mean,
            'ICIR': icir,
            'IC胜率': win_rate,
            '有效月数': n_months
        })
        print(f"  IC均值: {ic_mean:.4f}, ICIR: {icir:.4f}, IC胜率: {win_rate:.2%}")
    
    print("\n" + "=" * 80)
    print("汇总结果:")
    print("=" * 80)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # 结论
    print("\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    valid_results = [r for r in results if not np.isnan(r['IC均值'])]
    best = max(valid_results, key=lambda x: x['IC均值'])
    worst = min(valid_results, key=lambda x: x['IC均值'])
    
    print(f"最优方法: {best['方法']} (IC={best['IC均值']:.4f})")
    print(f"最差方法: {worst['方法']} (IC={worst['IC均值']:.4f})")
    
    # 检查Lasso是否比简单动量差
    simple_mom = next((r for r in results if '简单动量' in r['方法']), None)
    lasso_aic = next((r for r in results if 'LassoLarsIC(aic)' in r['方法']), None)
    
    if simple_mom and lasso_aic:
        print(f"\n简单动量IC: {simple_mom['IC均值']:.4f}")
        print(f"Lasso(aic)IC: {lasso_aic['IC均值']:.4f}")
        if lasso_aic['IC均值'] < simple_mom['IC均值']:
            print("⚠️ Lasso比简单动量还差！说明Lasso选择的跨行业关系是噪音")
            print("   建议：直接使用简单动量，或改用Ridge回归")


if __name__ == "__main__":
    main()
