# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Industry momentum factor analysis system for Chinese A-stock markets (Shenwan Level-1 industries). Implements 15 momentum factors across 5 categories with backtesting and IC/IR analysis.

## Commands

```bash
# Run single factor analysis (configure FACTOR_NAME in file first)
python 单因子测试.py

# Debug scripts in 单独测试/ directory
python 单独测试/debug_lasso_factor.py
python 单独测试/test_regression_methods.py
python 单独测试/generate_debug_report.py
```

## Architecture

### Core Modules

- **factors_analysis.py** - Main analysis framework (~1800 lines)

  - `FACTOR_CONFIG` dict: Central registry for all 15 factors with config (func, warmup, windows)
  - `DataContainer` class: Unified data management with date filtering
  - `analyze_single_factor()`: Analyze one factor across multiple lookback windows
  - `export_to_excel()` / `format_excel_report()`: Excel output with formatting
- **factor_.py** - Factor calculation functions (~1000 lines)

  - Each factor function takes `(price_df, lookback_window, ...)` and returns `pd.DataFrame`
  - Functions are registered in `FACTOR_CONFIG` in factors_analysis.py
- **data_loader.py** - Data loading with caching

  - `load_raw_data()`: Load from pickle cache (`data/sw_industry_data.pkl`)
  - `load_price_df()`, `load_turnover_df()`, etc.: Pivot to wide format DataFrames
  - Caches loaded data to avoid redundant disk I/O
- **单因子测试.py** - Entry point for single factor testing

### Data Flow

```
data/sw_industry_data.pkl → DataContainer → factor functions → backtest engine → Excel report
```

### Directory Structure

- `data/` - Cached data files (pickle, CSV, Excel)
- `bubble/` - Bubble detection module (PSY/BSADF methodology)
- `factor分析/` - Output directory for analysis Excel files
- `单独测试/` - Debug and test scripts

## Factor Categories

1. **Basic momentum (5)**: momentum, zscore, rank_zscore, sharpe, calmar
2. **Stable momentum (4)**: turnover_adj, price_volume_icir, rebound_with_crowding_filter, amplitude_cut
3. **Idiosyncratic returns (2)**: pure_liquidity_stripped, residual
4. **Cross-industry (1)**: cross_industry_lasso
5. **Intra-industry (3)**: industry_component, pca, lead_lag_enhanced

Full factor methodology documented in `介绍.markdown`.

## Key Configuration

In `factors_analysis.py`:

- `LOOKBACK_WINDOWS = [20, 60, 120, 240]` - Default windows (trading days)
- `DEFAULT_BACKTEST_YEARS = 10` - Backtest period
- `N_LAYERS = 5` - Stratified backtest layers
- `DEFAULT_REBALANCE_TYPE = 'monthly'` - Rebalancing frequency

## Dependencies

pandas, numpy, scipy, sklearn (LassoLarsIC), openpyxl


回答始终用中文
