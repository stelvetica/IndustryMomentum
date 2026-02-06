"""
Lasso因子调试脚本
逐步检查因子计算的每个环节，找出IC接近0的原因
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

def debug_lasso_factor():
    """逐步调试Lasso因子"""
    
    print("=" * 80)
    print("Lasso因子调试")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n[1] 加载数据...")
    prices_df = data_loader.load_price_df()
    print(f"    价格数据: {prices_df.shape[0]}个交易日, {prices_df.shape[1]}个行业")
    print(f"    日期范围: {prices_df.index[0].date()} 至 {prices_df.index[-1].date()}")
    print(f"    行业列表: {list(prices_df.columns)[:5]}...")
    
    # 2. 构造月度数据
    print("\n[2] 构造月度数据...")
    trade_dates = prices_df.index
    monthly_last = trade_dates.to_series().groupby(
        [trade_dates.year, trade_dates.month]
    ).apply(lambda x: x.iloc[-1])
    monthly_last = pd.DatetimeIndex(monthly_last)
    monthly_close = prices_df.loc[monthly_last]
    print(f"    月度数据: {len(monthly_close)}个月")
    
    # 3. 计算月度收益和超额收益
    print("\n[3] 计算月度收益...")
    monthly_ret = monthly_close.pct_change()
    bench_monthly = monthly_ret.mean(axis=1)
    excess_monthly = monthly_ret.sub(bench_monthly, axis=0)
    
    # 检查数据质量
    nan_ratio = excess_monthly.isna().sum().sum() / (excess_monthly.shape[0] * excess_monthly.shape[1])
    print(f"    超额收益NaN比例: {nan_ratio:.2%}")
    print(f"    超额收益均值: {excess_monthly.mean().mean():.6f}")
    print(f"    超额收益标准差: {excess_monthly.std().mean():.6f}")
    
    # 4. 检查单期IC（直接用当月超额收益预测下月超额收益）
    print("\n[4] 检查单期IC（简单动量基准）...")
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
    
    simple_ic_mean = np.mean(ic_list)
    simple_icir = np.mean(ic_list) / np.std(ic_list) if len(ic_list) > 0 else 0
    print(f"    简单动量IC均值: {simple_ic_mean:.4f}")
    print(f"    简单动量ICIR: {simple_icir:.4f}")
    print(f"    说明: 这是用当月超额收益直接预测下月超额收益的IC")
    
    # 5. 检查Lasso模型效果
    print("\n[5] 检查Lasso模型效果...")
    industries = prices_df.columns.tolist()
    months = monthly_close.index.tolist()
    T = len(months)
    
    min_train_samples = 10
    train_periods = 60  # 使用60个月训练数据
    
    # 存储预测结果
    predictions = []
    actuals = []
    pred_dates = []
    
    # 记录模型诊断信息
    n_nonzero_coefs = []
    model_scores = []
    
    for s_pred in range(1 + min_train_samples, T - 1):
        factor_date = months[s_pred]
        
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
        
        # 对每个行业做Lasso回归
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
                    
                    # 记录非零系数数量
                    n_nonzero = np.sum(lasso.coef_ != 0)
                    n_nonzero_coefs.append(n_nonzero)
                    
            except Exception as e:
                continue
        
        # 获取实际的下月收益
        if s_pred + 1 < T:
            actual_ret = excess_monthly.loc[months[s_pred + 1]]
            for ind in pred_row.keys():
                if ind in actual_ret.index and not np.isnan(actual_ret[ind]):
                    actual_row[ind] = actual_ret[ind]
        
        if len(pred_row) > 5 and len(actual_row) > 5:
            predictions.append(pred_row)
            actuals.append(actual_row)
            pred_dates.append(factor_date)
    
    # 6. 计算Lasso预测的IC
    print("\n[6] Lasso预测效果...")
    lasso_ic_list = []
    for pred_row, actual_row in zip(predictions, actuals):
        common_inds = set(pred_row.keys()) & set(actual_row.keys())
        if len(common_inds) < 5:
            continue
        
        pred_vals = [pred_row[ind] for ind in common_inds]
        actual_vals = [actual_row[ind] for ind in common_inds]
        
        ic, _ = stats.spearmanr(pred_vals, actual_vals)
        if not np.isnan(ic):
            lasso_ic_list.append(ic)
    
    if len(lasso_ic_list) > 0:
        lasso_ic_mean = np.mean(lasso_ic_list)
        lasso_icir = lasso_ic_mean / np.std(lasso_ic_list)
        lasso_win_rate = np.mean([1 if ic > 0 else 0 for ic in lasso_ic_list])
        
        print(f"    Lasso预测IC均值: {lasso_ic_mean:.4f}")
        print(f"    Lasso预测ICIR: {lasso_icir:.4f}")
        print(f"    IC胜率: {lasso_win_rate:.2%}")
        print(f"    有效月份数: {len(lasso_ic_list)}")
    else:
        print("    警告: 没有有效的Lasso预测结果！")
    
    # 7. 诊断信息
    print("\n[7] 模型诊断...")
    if len(n_nonzero_coefs) > 0:
        print(f"    平均非零系数数量: {np.mean(n_nonzero_coefs):.1f}")
        print(f"    非零系数数量分布: min={np.min(n_nonzero_coefs)}, max={np.max(n_nonzero_coefs)}")
    
    # 8. 对比分析
    print("\n[8] 问题诊断...")
    print(f"    简单动量IC: {simple_ic_mean:.4f}")
    if len(lasso_ic_list) > 0:
        print(f"    Lasso预测IC: {lasso_ic_mean:.4f}")
        
        if lasso_ic_mean < simple_ic_mean * 0.5:
            print("\n    ⚠️ 问题: Lasso预测IC远低于简单动量IC！")
            print("    可能原因:")
            print("    1. Lasso正则化过强，大多数系数被压缩为0")
            print("    2. 训练样本不足，模型难以学习有效的领先滞后关系")
            print("    3. 行业分类不同（研报用中信，你用申万）")
            print("    4. 特征和目标之间的关系本身就很弱")
    
    # 9. 检查预测值分布
    print("\n[9] 预测值分布...")
    if len(predictions) > 0:
        all_preds = []
        for pred_row in predictions:
            all_preds.extend(pred_row.values())
        
        print(f"    预测值均值: {np.mean(all_preds):.6f}")
        print(f"    预测值标准差: {np.std(all_preds):.6f}")
        print(f"    预测值范围: [{np.min(all_preds):.4f}, {np.max(all_preds):.4f}]")
        
        # 检查预测值是否过于集中
        if np.std(all_preds) < 0.001:
            print("\n    ⚠️ 问题: 预测值标准差过小，预测值几乎没有区分度！")
            print("    这说明Lasso模型倾向于预测接近均值的结果。")
    
    print("\n" + "=" * 80)
    print("调试完成")
    print("=" * 80)


if __name__ == "__main__":
    debug_lasso_factor()
