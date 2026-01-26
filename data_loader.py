"""
数据加载模块
统一管理从 sw_industry_data.pkl 加载数据的逻辑

数据格式：
- 'data': DataFrame (列: 日期, 代码, 名称, CLOSE, HIGH, LOW, VOLUME, DQ_AMTTURNOVER)
- 'industry_info': DataFrame (列: 代码, 名称)
- 'download_time': str
- 'columns': list
"""
import pandas as pd
import pickle
import os

# 默认数据文件路径
DEFAULT_CACHE_FILE = "data/sw_industry_data.pkl"
DEFAULT_BARRA_FILE = "data/全市场纯因子收益-仅风格因子-日频.xlsx"
DEFAULT_CONSTITUENT_FILE = "data/申万一级行业成分股_2012起.csv"
DEFAULT_STOCK_PRICE_FILE = "data/个股_复权收盘价.csv"
DEFAULT_STOCK_MV_FILE = "data/个股_流通市值.csv"
DEFAULT_INDUSTRY_CODE_FILE = "data/申万行业指数.csv"


# 缓存已加载的原始数据，避免重复加载
_raw_data_cache = {}

def load_raw_data(file_path=DEFAULT_CACHE_FILE, verbose=True):
    """
    加载原始数据（pickle格式）

    参数:
    file_path: str, 数据文件路径
    verbose: bool, 是否打印加载信息（默认True，但缓存命中时不打印）

    返回:
    dict: 包含 'data', 'industry_info', 'download_time', 'columns' 的字典
    """
    # 检查缓存
    if file_path in _raw_data_cache:
        return _raw_data_cache[file_path]

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    with open(file_path, 'rb') as f:
        data_store = pickle.load(f)

    # 缓存数据
    _raw_data_cache[file_path] = data_store

    if verbose:
        df = data_store['data']
        n_industries = df['代码'].nunique()
        print(f"  行业指数: 行[{df['日期'].min()} - {df['日期'].max()}] 列[{n_industries}个行业]")

    return data_store


def load_price_df(file_path=DEFAULT_CACHE_FILE):
    """
    加载价格数据（收盘价）
    
    参数:
    file_path: str, 数据文件路径
    
    返回:
    pd.DataFrame: 价格数据框 (index=日期, columns=行业名称)
    """
    data_store = load_raw_data(file_path)
    df = data_store['data']
    
    # 将长格式转换为宽格式（透视表）
    # 行索引为日期，列为行业名称，值为收盘价
    price_df = df.pivot(index='日期', columns='名称', values='CLOSE')
    
    # 确保日期索引为DatetimeIndex
    if not isinstance(price_df.index, pd.DatetimeIndex):
        price_df.index = pd.to_datetime(price_df.index)
    
    price_df = price_df.sort_index()
    #price_df = price_df.ffill()
    
    return price_df


def load_high_df(file_path=DEFAULT_CACHE_FILE):
    """
    加载最高价数据
    
    参数:
    file_path: str, 数据文件路径
    
    返回:
    pd.DataFrame: 最高价数据框 (index=日期, columns=行业名称)
    """
    data_store = load_raw_data(file_path)
    df = data_store['data']
    
    # 将长格式转换为宽格式（透视表）
    high_df = df.pivot(index='日期', columns='名称', values='HIGH')
    
    # 确保日期索引为DatetimeIndex
    if not isinstance(high_df.index, pd.DatetimeIndex):
        high_df.index = pd.to_datetime(high_df.index)
    
    high_df = high_df.sort_index()
    
    return high_df


def load_low_df(file_path=DEFAULT_CACHE_FILE):
    """
    加载最低价数据
    
    参数:
    file_path: str, 数据文件路径
    
    返回:
    pd.DataFrame: 最低价数据框 (index=日期, columns=行业名称)
    """
    data_store = load_raw_data(file_path)
    df = data_store['data']
    
    # 将长格式转换为宽格式（透视表）
    low_df = df.pivot(index='日期', columns='名称', values='LOW')
    
    # 确保日期索引为DatetimeIndex
    if not isinstance(low_df.index, pd.DatetimeIndex):
        low_df.index = pd.to_datetime(low_df.index)
    
    low_df = low_df.sort_index()
    
    return low_df


def load_turnover_df(file_path=DEFAULT_CACHE_FILE):
    """
    加载换手率数据（成交额换手率）
    
    参数:
    file_path: str, 数据文件路径
    
    返回:
    pd.DataFrame: 换手率数据框 (index=日期, columns=行业名称)
    """
    data_store = load_raw_data(file_path)
    df = data_store['data']
    
    # 将长格式转换为宽格式（透视表）
    # 行索引为日期，列为行业名称，值为成交额换手率
    turnover_df = df.pivot(index='日期', columns='名称', values='DQ_AMTTURNOVER')
    
    # 确保日期索引为DatetimeIndex
    if not isinstance(turnover_df.index, pd.DatetimeIndex):
        turnover_df.index = pd.to_datetime(turnover_df.index)
    
    turnover_df = turnover_df.sort_index()
    
    return turnover_df


def load_volume_df(file_path=DEFAULT_CACHE_FILE):
    """
    加载成交量数据
    
    参数:
    file_path: str, 数据文件路径
    
    返回:
    pd.DataFrame: 成交量数据框 (index=日期, columns=行业名称)
    """
    data_store = load_raw_data(file_path)
    df = data_store['data']
    
    # 将长格式转换为宽格式（透视表）
    volume_df = df.pivot(index='日期', columns='名称', values='VOLUME')
    
    # 确保日期索引为DatetimeIndex
    if not isinstance(volume_df.index, pd.DatetimeIndex):
        volume_df.index = pd.to_datetime(volume_df.index)
    
    volume_df = volume_df.sort_index()
    #volume_df = volume_df.ffill()
    
    return volume_df


def get_industry_list(file_path=DEFAULT_CACHE_FILE):
    """
    获取行业列表
    
    参数:
    file_path: str, 数据文件路径
    
    返回:
    pd.DataFrame: 行业信息 (列: 代码, 名称)
    """
    data_store = load_raw_data(file_path)
    return data_store['industry_info']


def load_constituent_df(file_path=DEFAULT_CONSTITUENT_FILE):
    """
    加载申万一级行业成分股数据
    
    数据格式：
    - date: 日期
    - 权重更新日期: 权重更新时间
    - wind_code: 股票代码（如601118.SH）
    - i_weight: 权重（百分比）
    - index_code: 行业指数代码（如801010.SI）
    - sec_name: 股票名称
    - industry: 行业分类
    - 数据来源: 数据来源
    
    参数:
    file_path: str, 成分股数据文件路径
    
    返回:
    pd.DataFrame: 成分股数据框
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"成分股数据文件不存在: {file_path}")

    df = pd.read_csv(file_path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])

    print(f"  成分股: 行[{df['date'].min().date()} - {df['date'].max().date()}] 列[{df['index_code'].nunique()}个行业]")

    return df


def load_stock_price_df(file_path=DEFAULT_STOCK_PRICE_FILE):
    """
    加载个股复权收盘价数据

    参数:
    file_path: str, 个股价格数据文件路径

    返回:
    pd.DataFrame: 个股价格数据框 (index=日期, columns=股票代码)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"个股价格数据文件不存在: {file_path}")

    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    print(f"  个股价格: 行[{df.index[0].date()} - {df.index[-1].date()}] 列[{len(df.columns)}只股票]")

    return df


def load_stock_mv_df(file_path=DEFAULT_STOCK_MV_FILE):
    """
    加载个股流通市值数据

    参数:
    file_path: str, 个股流通市值数据文件路径

    返回:
    pd.DataFrame: 个股流通市值数据框 (index=日期, columns=股票代码)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"个股流通市值数据文件不存在: {file_path}")

    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    print(f"  个股市值: 行[{df.index[0].date()} - {df.index[-1].date()}] 列[{len(df.columns)}只股票]")

    return df


def load_barra_factor_returns(file_path=DEFAULT_BARRA_FILE):
    """
    加载 Barra 风格因子日频收益率数据

    数据来源：Barra CNE5/CNE6 模型的风格因子收益率
    用途：用于行业残差动量因子的计算，剥离市场和风格因素

    参数:
    file_path: str, Barra因子数据文件路径（Excel格式）

    返回:
    pd.DataFrame: Barra因子收益率数据框
        - index: 日期 (DatetimeIndex)
        - columns: 因子名称 (市场, Size, Beta, Momentum, ResidualVolatility,
                   NonlinearSize, BookToPrice, Liquidity, EarningsYield, Growth, Leverage)

    注意:
    - 返回的是因子收益率 (Factor Return)，而非因子载荷 (Factor Exposure)
    - 市场因子代表大盘整体涨跌
    - 其他10个风格因子代表各风格特征的超额收益
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Barra因子数据文件不存在: {file_path}")

    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 第一列是日期，重命名并设置为索引
    df = df.rename(columns={'Unnamed: 0': '日期'})
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.set_index('日期')
    df = df.sort_index()

    print(f"  Barra因子: 行[{df.index[0].date()} - {df.index[-1].date()}] 列[{len(df.columns)}个因子]")

    return df


def load_industry_code_df(file_path=DEFAULT_INDUSTRY_CODE_FILE):
    """
    加载申万行业指数代码映射数据
    
    数据格式：
    - 申万一级行业代码: 行业指数代码（如801010.SI）
    - 申万一级行业名称: 行业名称（如农林牧渔(申万)）
    
    参数:
    file_path: str, 行业代码映射文件路径
    
    返回:
    pd.DataFrame: 行业代码映射数据框
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"行业代码映射文件不存在: {file_path}")

    df = pd.read_csv(file_path)

    return df


def get_industry_name_to_code_map(industry_code_df=None):
    """
    构建行业名称到行业代码的映射字典
    
    参数:
    industry_code_df: pd.DataFrame or None, 行业代码映射数据框
                      如果为None，则自动加载
    
    返回:
    dict: 行业名称（去掉"(申万)"后缀）到行业代码的映射
    """
    if industry_code_df is None:
        industry_code_df = load_industry_code_df()
    
    sw_industry_codes = {}
    for _, row in industry_code_df.iterrows():
        code = row['申万一级行业代码']
        name = row['申万一级行业名称']
        if pd.notna(code) and pd.notna(name):
            # 去掉"(申万)"后缀作为key
            clean_name = name.replace('(申万)', '').replace('（申万）', '')
            sw_industry_codes[clean_name] = code
    
    return sw_industry_codes


if __name__ == "__main__":
    # 测试数据加载
    print("=" * 50)
    print("测试数据加载模块")
    print("=" * 50)
    
    # 获取行业信息
    industry_info = get_industry_list()
    # 获取行业列表
    print(f"\n行业列表预览:\n{industry_info.head(10)}")
    print(f"行业数量: {len(industry_info.index)}")
    
    # 加载价格数据（收盘价）
    price_df = load_price_df()
    print(f"日期实际范围: {price_df.index[0].date()} 至 {price_df.index[-1].date()}")
    print(f"\n收盘价数据预览（最新5行）:\n{price_df.tail()}")
    
    # 加载最高价数据
    high_df = load_high_df()
    print(f"\n最高价数据预览（最新5行）:\n{high_df.tail()}")
    
    # 加载最低价数据
    low_df = load_low_df()
    print(f"\n最低价数据预览（最新5行）:\n{low_df.tail()}")
    
    # 加载换手率数据
    turnover_df = load_turnover_df()
    print(f"\n换手率数据预览（最新5行）:\n{turnover_df.tail()}")
    
    # 加载成交量数据
    volume_df = load_volume_df()
    print(f"\n成交量数据预览（最新5行）:\n{volume_df.tail()}")