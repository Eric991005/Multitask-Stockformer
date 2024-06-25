# Import the necessary packages
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

warnings.filterwarnings('ignore')  # Ignore matched warnings
matplotlib.rc("font", family='Kaiti')  # Set Chinese font
matplotlib.rcParams['axes.unicode_minus'] = False  # Correctly display plus/minus signs

def fun(factor, df_ltsz, df_indus):    

    # Store all factors
    all_factor = []

    # Get industries
    indus = df_indus
    # Get all stock codes
    codes = indus.index.to_list()

    factor_1 = factor.drop(columns=['LABEL0'])
    grouped_dfs = [group_df for _, group_df in factor_1.groupby('datetime')]

    # Factors for each date
    for dfs in grouped_dfs:

        dfs = dfs.rename(columns={'instrument':'code'})
        dfs = dfs.set_index(['code'])
        # Get the date
        date = dfs['datetime'][0]

        # Get market value
        ltsz = np.log(df_ltsz.loc[date].to_frame())

        # Get factor names
        lst_f = dfs.columns[2:].tolist()

        # Record factors for each date
        all_factor_temp = []
        for x in lst_f:

            # x is the factor name, y is the factor value corresponding to codes
            y = dfs[x].to_frame(name=date)

            # Check if all are Na
            if (~y.isna().values).sum() == 0:
                break

            # Concatenate
            df = pd.concat([y, ltsz, indus], axis=1)
            df.columns = range(df.shape[1])
            # Regression
            model = sm.OLS(df.iloc[:,0].to_frame(), df.drop(0,axis=1),missing='drop')
            results = model.fit()
            # Residuals as the neutralized factor
            y = (y.loc[results.fittedvalues.index] - results.fittedvalues.to_frame().rename(columns={0:date}))
            y = y.reindex(y.index)
            y = y.rename(columns={date:x})
            # Normalize factors
            y = y.replace(np.inf,np.nan).replace(-np.inf,np.nan)
            y = (y - y.mean()) / y.std()

            all_factor_temp.append(y)

        df_pro = pd.concat(all_factor_temp,axis=1).reindex(codes)
        df_pro['datetime'] = date
        all_factor.append(df_pro)
        # Add time column

    factor_final = pd.concat(all_factor)

    return factor_final


if __name__ == '__main__':
    ####################### Establish parameter set
    # Set backtest parameters
    param = {}
    param['start_date'] = '2018-03-15' # Start date of the backtest
    param['end_date'] = '2024-02-29' # End date of the backtest
    param['market_value_neutral'] = True  # Market value neutralization

    # Import closing price data
    df = pd.read_csv('../data/18-24 Standard Data.csv', index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    df_close = df.pivot(index='date', columns='code', values='close')
    df_ltsz = df.pivot(index='date', columns='code', values='ltsz')

    # Import factors
    factor = pd.read_csv('../data/factor158_unneutralized.csv')

    # Import industries
    indus = pd.read_csv('../data/CSI 300 Constituent Stocks Industry Dummy Variables.csv', index_col=0)
    indus = indus.set_index('code')

    hh = fun(factor, df_ltsz, indus)
    hh.to_csv('../data/factor_158_neutralized.csv')
