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
DEFAULT_CACHE_FILE = "sw_industry_data.pkl"


def load_raw_data(file_path=DEFAULT_CACHE_FILE):
    """
    加载原始数据（pickle格式）
    
    参数:
    file_path: str, 数据文件路径
    
    返回:
    dict: 包含 'data', 'industry_info', 'download_time', 'columns' 的字典
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    with open(file_path, 'rb') as f:
        data_store = pickle.load(f)
    
    df = data_store['data']
    n_industries = df['代码'].nunique()
    print(f"已加载 {n_industries} 个行业的数据")
    print(f"日期范围: {df['日期'].min()} 至 {df['日期'].max()}")
    
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