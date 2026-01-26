"""
使用Wind API下载申万一级行业指数数据
获取指标：收盘价(close)、最高价(high)、最低价(low)、成交量(volume)、成交额换手率(dq_amtturnover)

数据结构：每个行业单独获取，最终合并为一个大DataFrame
列顺序：日期, 代码, 名称, CLOSE, HIGH, LOW, VOLUME, DQ_AMTTURNOVER
"""

import pandas as pd
import pickle
from datetime import datetime
import os


def download_industry_data(start_date, end_date):
    """
    从Wind API下载申万一级行业指数数据
    每个行业单独调用一次wsd获取多个指标
    """
    # 导入WindPy
    try:
        from WindPy import w
    except ImportError:
        print("错误：无法导入WindPy，请确保已安装Wind API插件")
        print("请在Wind终端中点击'开始/插件修复'选项，修复Python接口")
        return None
    
    # 启动Wind API
    print("正在启动Wind API...")
    w.start()
    
    # 检查连接状态
    if not w.isconnected():
        print("错误：Wind API连接失败，请确保Wind终端已登录")
        return None
    
    print("Wind API连接成功！")
    
    # 读取申万行业指数CSV文件
    csv_path = "data/申万行业指数.csv"
    if not os.path.exists(csv_path):
        print(f"错误：找不到文件 {csv_path}")
        w.stop()
        return None
    
    # 读取CSV文件
    raw_df = pd.read_csv(csv_path, encoding='utf-8')
    
    # 提取一级行业代码和名称（去重）
    industry_df = raw_df[['申万一级行业代码', '申万一级行业名称']].drop_duplicates()
    industry_df = industry_df.dropna()  # 去除空值行
    industry_df.columns = ['代码', '名称']
    industry_df = industry_df.reset_index(drop=True)
    
    # 剔除行业名称开头是"综合"的行业
    industry_df = industry_df[~industry_df['名称'].str.startswith('综合')]
    industry_df = industry_df.reset_index(drop=True)
    
    print(f"读取到 {len(industry_df)} 个申万一级行业（已剔除综合类行业）")
    
    # 创建代码到名称的映射
    code_to_name = dict(zip(industry_df['代码'], industry_df['名称']))
    # 获取行业代码列表
    codes = industry_df['代码'].tolist()
    
    # 要获取的指标（单品种多指标）
    # close: 收盘价, high: 最高价, low: 最低价, volume: 成交量, dq_amtturnover: 成交额换手率
    fields = "close,high,low,volume,dq_amtturnover"
    
    print(f"\n开始下载数据...")
    print(f"日期范围: {start_date} 至 {end_date}")
    print(f"指标: {fields}")
    print(f"行业数量: {len(codes)}")
    
    # 存储所有行业数据的列表
    all_industry_data = []
    
    # 对每个行业单独获取数据（单品种多指标）
    for i, code in enumerate(codes):
        name = code_to_name[code]
        print(f"\r正在下载: [{i+1}/{len(codes)}] {code} {name}", end="", flush=True)
        
        # 使用wsd获取单品种多指标的时间序列数据
        # PriceAdj=B 表示后复权
        result = w.wsd(
            code,           # 单个行业代码
            fields,         # 多个指标
            start_date,
            end_date,
            "Days=Trading;Fill=Blank;PriceAdj=B",
            usedf=True        # 直接返回DataFrame格式
        )
        
        # result返回的是(ErrorCode, DataFrame)元组
        error_code, df = result
        
        # 检查返回结果
        if error_code != 0:
            print(f"\n警告：获取 {code} 数据时出错，错误码: {error_code}")
            continue
        
        # df的结构：行索引是日期，列是指标名称
        # 添加代码和名称列
        df.insert(0, '代码', code)
        df.insert(1, '名称', name)
        
        # 重置索引，将日期变为普通列
        df = df.reset_index()
        df = df.rename(columns={'index': '日期'})
        
        all_industry_data.append(df)
    
    print(f"\n\n数据下载完成！成功获取 {len(all_industry_data)} 个行业的数据")
    
    # 合并所有行业数据
    if all_industry_data:
        combined_df = pd.concat(all_industry_data, ignore_index=True)
        # 调整列顺序（列名会是Wind返回的大写形式）
        # Wind返回的列名可能是 CLOSE, HIGH, LOW, VOLUME, DQ_AMTTURNOVER
        combined_df = combined_df[['日期', '代码', '名称', 'CLOSE', 'HIGH', 'LOW', 'VOLUME', 'DQ_AMTTURNOVER']]
    else:
        combined_df = None
    
    # 关闭Wind API连接
    w.stop()
    print("Wind API连接已关闭")
    
    return combined_df, industry_df

def save_data(combined_df, industry_df, output_path="data/sw_industry_data.pkl"):
    """
    保存数据为pkl格式
    
    参数:
    combined_df: pd.DataFrame, 合并后的数据
    industry_df: pd.DataFrame, 行业信息
    output_path: str, 输出文件路径
    """
    if combined_df is None:
        print("没有数据可保存")
        return
    
    # 创建保存的数据结构
    save_dict = {
        'data': combined_df,                   # 合并后的完整数据
        'industry_info': industry_df,          # 行业信息
        'download_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'columns': ['日期', '代码', '名称', 'CLOSE', 'HIGH', 'LOW', 'VOLUME', 'DQ_AMTTURNOVER']
    }
    
    # 保存为pkl文件
    with open(output_path, 'wb') as f:
        pickle.dump(save_dict, f)
    
    print(f"\n数据已保存至: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    # 打印数据摘要
    print("\n数据摘要:")
    print(f"  总行数: {len(combined_df)}")
    print(f"  列名: {list(combined_df.columns)}")
    print(f"  行业数量: {combined_df['代码']}")
    print(f"  日期范围: {combined_df['日期'].min()} 至 {combined_df['日期'].max()}")
    print(combined_df.head())


if __name__ == "__main__":
    # ========== 日期参数配置 ==========
    START_DATE = "2010-01-01"  # 数据起始日期
    END_DATE = "2026-01-13"    # 数据截止日期
    # =================================

    # ========== 下载数据 ==========
    # 取消下面的注释来下载数据
    '''
    result = download_industry_data(start_date=START_DATE, end_date=END_DATE)
    
    if result is not None:
        combined_df, industry_df = result
        # 保存数据
        save_data(combined_df, industry_df)
    '''
    # ========== 测试：加载和查询数据 ==========
    file_path = "data/sw_industry_data.pkl"
    
    # 加载数据
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        print("数据加载成功！")
        print(f"下载时间: {loaded_data.get('download_time', '未知')}")
        
        # 获取完整数据
        df = loaded_data['data']
        print(f"\n数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        
        # 获取特定行业数据示例
        example_code = '801750.SI'
        industry_data = df[df['代码'] == example_code].copy()
        print(f"\n示例行业数据 ({example_code}) 预览:")
        print(industry_data.head())
    else:
        print(f"错误：找不到文件 {file_path}，请先运行下载数据")