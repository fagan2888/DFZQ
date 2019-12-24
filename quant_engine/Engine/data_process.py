import pandas as pd
import numpy as np
import math

class DataProcess:
    @staticmethod
    def remove_outlier(series):
        median = np.median(series)
        MAD = np.median(abs(series - median))
        ceil = median + 3 * 1.4826 * MAD
        floor = median - 3 * 1.4826 * MAD
        series.clip(floor,ceil,inplace=True)
        return series

    @staticmethod
    def Z_standardize(series):
        mean = np.mean(series)
        std = np.std(series)
        if std == 0:
            series.clip(0,0,inplace=True)
        else:
            series = (series-mean)/std
        return series

    @staticmethod
    def rank_standardize(series):
        rank = series.rank()
        mean = np.mean(rank)
        std = np.std(rank)
        if std == 0:
            rank[rank != 0] = 0
        else:
            rank = (rank-mean)/std
        return rank

    @staticmethod
    def calc_ann_return(series):
        return (series.iloc[-1]/series.iloc[0] -1)**(250/series.shape[0])

    @staticmethod
    def calc_alpha_ann_return(series1,series2):
        ret1 = (series1.iloc[-1]/series1.iloc[0] -1)**(250/series1.shape[0])
        ret2 = (series2.iloc[-1]/series2.iloc[0] -1)**(250/series2.shape[0])
        return ret1-ret2

    @staticmethod
    def calc_max_draw_down(series):
        index_low = np.argmax(np.maximum.accumulate(series) - series)
        index_high = np.argmax(series[:index_low])
        return (series[index_high]-series[index_low])/series[index_high]

    @staticmethod
    def calc_sharpe(series):
        return math.sqrt(252) * series.pct_change().mean()/series.pct_change().std()

    @staticmethod
    def calc_alpha_sharpe(series1,series2):
        ret_series1 = series1.pct_change()
        ret_series2 = series2.pct_change()
        alpha = ret_series1 - ret_series2
        return math.sqrt(252) * alpha.mean() / alpha.std()

    @staticmethod
    def get_next_date(calendar,today,days):
        return {today:calendar[calendar>today].iloc[days-1]}

    @staticmethod
    def get_next_period_return(mkt_data,calendar,days):
        # 默认index是日期
        idxs = mkt_data.index.unique()
        next_date_dict = {}
        for idx in idxs:
            next_date_dict.update(DataProcess.get_next_date(calendar,idx,days))
        field = 'next_' + str(days) + '_date'
        mkt_data[field] = mkt_data.apply(lambda row: next_date_dict[row.name],axis=1)
        mkt_data.index.names = ['date']
        mkt_data.reset_index(inplace=True)
        fq_close = mkt_data.loc[:,['date','code','adj_factor','close']].copy()
        fq_close['next_fq_close'] = fq_close['adj_factor'] * fq_close['close']
        fq_close = fq_close.loc[:,['date','code','next_fq_close']]
        fq_close.rename(columns={'date':field},inplace=True)
        mkt_data = pd.merge(mkt_data,fq_close,how='left',on=[field,'code'])
        mkt_data['next_period_return'] = mkt_data['next_fq_close'] / mkt_data['adj_factor'] / mkt_data['close'] - 1
        mkt_data = mkt_data.set_index('date')
        return mkt_data