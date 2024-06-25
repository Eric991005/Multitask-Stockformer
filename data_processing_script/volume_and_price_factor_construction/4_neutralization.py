#导入需要用到的包
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import warnings 
from numpy import *

warnings.filterwarnings('ignore') #忽略匹配的警告
matplotlib.rc("font", family='Kaiti')#设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False#正确显示正负号



def fun(factor, df_ltsz, df_indus):    

    # 存储所有因子
    all_factor = []

    # 获取行业
    indus = df_indus
    # 获取所有股票代码
    codes = indus.index.to_list()

    factor_1 = factor.drop(columns=['LABEL0'])
    grouped_dfs = [group_df for _, group_df in factor_1.groupby('datetime')]

    # 每个日期的所有因子
    for dfs in grouped_dfs:

        dfs = dfs.rename(columns={'instrument':'code'})
        dfs = dfs.set_index(['code'])
        # 获取时间
        date = dfs['datetime'][0]

        # 获取市值
        ltsz = np.log(df_ltsz.loc[date].to_frame())

        # 获取因子名称
        lst_f = dfs.columns[2:].tolist()

        # 记录每个日期的所有因子
        all_factor_temp = []
        for x in lst_f:

            # x表示每个因子名称，y表示codes对应的因子值
            y = dfs[x].to_frame(name = date)

            #检验是否全部为Na
            if (~y.isna().values).sum() == 0:
                
                break

            # 拼接
            df = pd.concat([y,ltsz,indus], axis=1)
            df.columns = range(df.shape[1])
            # 回归
            model = sm.OLS(df.iloc[:,0].to_frame(), df.drop(0,axis=1),missing='drop')
            results = model.fit()
            # 取残差为中性化后的因子
            y = (y.loc[results.fittedvalues.index] - results.fittedvalues.to_frame().rename(columns={0:date}))
            y = y.reindex(y.index)
            y = y.rename(columns = {date:x})
            # 因子标准化
            y = y.replace(np.inf,np.nan).replace(-np.inf,np.nan)
            y = (y - y.mean()) / y.std()

            all_factor_temp.append(y)

        df_pro = pd.concat(all_factor_temp,axis=1).reindex(codes)
        df_pro['datetime'] = date
        all_factor.append(df_pro)

        # 添加时间列

    
    factor_final = pd.concat(all_factor)

    return factor_final


if __name__ == '__main__':
    ####################### 建立参数集合
    #设置回测参数
    param = {}
    param['start_date'] = '2018-03-15' #回测开始时间
    param['end_date'] = '2024-02-29' #回测结束时间
    param['market_value_neutral'] = True  # 市值中性化

    #  导入收盘价
    df = pd.read_csv('../data/18-24标准数据.csv',index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    df_close = df.pivot(index='date', columns='code', values='close')
    df_ltsz = df.pivot(index='date', columns='code', values='ltsz')

    # 导入因子
    factor = pd.read_csv('../data/factor158_未中性化.csv')

    # 导入行业
    indus = pd.read_csv('../data/沪深300成分股行业哑变量.csv',index_col=0)
    indus = indus.set_index('code')

    hh = fun(factor,df_ltsz,indus)
    hh.to_csv('../data/factor_158_中性化.csv')