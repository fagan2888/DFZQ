import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
from joblib import Parallel, delayed, parallel_backend


class DataProcess:
    @staticmethod
    def remove_outlier(series):
        median = np.median(series)
        MAD = np.median(abs(series - median))
        ceil = median + 3 * 1.4826 * MAD
        floor = median - 3 * 1.4826 * MAD
        series.clip(floor, ceil, inplace=True)
        return series

    @staticmethod
    def Z_standardize(series):
        mean = np.mean(series)
        std = np.std(series)
        if std == 0:
            series.clip(0, 0, inplace=True)
        else:
            series = (series - mean) / std
        return series

    @staticmethod
    def rank_standardize(series):
        rank = series.rank()
        mean = np.mean(rank)
        std = np.std(rank)
        if std == 0:
            rank[rank != 0] = 0
        else:
            rank = (rank - mean) / std
        return rank

    @staticmethod
    def calc_ann_return(series):
        return (series.iloc[-1] / series.iloc[0] - 1) ** (250 / series.shape[0])

    @staticmethod
    def calc_alpha_ann_return(series1, series2):
        ret1 = (series1.iloc[-1] / series1.iloc[0] - 1) ** (250 / series1.shape[0])
        ret2 = (series2.iloc[-1] / series2.iloc[0] - 1) ** (250 / series2.shape[0])
        return ret1 - ret2

    @staticmethod
    def calc_max_draw_down(series):
        index_low = np.argmax(np.maximum.accumulate(series) - series)
        index_high = np.argmax(series[:index_low])
        return (series[index_high] - series[index_low]) / series[index_high]

    @staticmethod
    def calc_sharpe(series):
        return math.sqrt(252) * series.pct_change().mean() / series.pct_change().std()

    @staticmethod
    def calc_alpha_sharpe(series1, series2):
        ret_series1 = series1.pct_change()
        ret_series2 = series2.pct_change()
        alpha = ret_series1 - ret_series2
        return math.sqrt(252) * alpha.mean() / alpha.std()

    @staticmethod
    def get_next_date(calendar, today, days):
        return {today: calendar[calendar > today].iloc[days - 1]}

    @staticmethod
    # 返回的date在columns里
    def add_next_period_return(mkt_data, calendar, days):
        # 默认index是日期
        idxs = mkt_data.index.unique()
        next_date_dict = {}
        for idx in idxs:
            next_date_dict.update(DataProcess.get_next_date(calendar, idx, days))
        field = 'next_' + str(days) + '_date'
        mkt_data[field] = mkt_data.apply(lambda row: next_date_dict[row.name], axis=1)
        mkt_data.index.names = ['date']
        mkt_data.reset_index(inplace=True)
        fq_close = mkt_data.loc[:, ['date', 'code', 'adj_factor', 'close']].copy()
        fq_close['next_fq_close'] = fq_close['adj_factor'] * fq_close['close']
        fq_close = fq_close.loc[:, ['date', 'code', 'next_fq_close']]
        fq_close.rename(columns={'date': field}, inplace=True)
        mkt_data = pd.merge(mkt_data, fq_close, how='left', on=[field, 'code'])
        mkt_data['next_period_return'] = mkt_data['next_fq_close'] / mkt_data['adj_factor'] / mkt_data['close'] - 1
        return mkt_data

    @staticmethod
    # 获取行业哑变量
    def get_industry_dummies(mkt_data, industry_field):
        industry_data = pd.get_dummies(mkt_data[industry_field])
        industry_data = pd.concat([mkt_data['code'], industry_data], axis=1)
        # 过滤掉没有行业信息的数据
        industry_data = industry_data.loc[~(industry_data == 0).all(axis=1), :]
        return industry_data

    @staticmethod
    # 市值行业中性化
    # 返回的date在columns里
    def neutralize(factor_data, factor_field, mkt_data, size_data, industry_field='improved_lv1',
                    size_field='ln_market_cap'):
        industry = DataProcess.get_industry_dummies(mkt_data, industry_field)
        industry.index.names = ['date']
        factor_data.index.names = ['date']
        size_data.index.names = ['date']
        size_data.rename(columns={size_field: 'size'}, inplace=True)
        factor = factor_data.reset_index().loc[:, ['date', 'code', factor_field]].copy()
        industry = industry.reset_index()
        size = size_data.reset_index().loc[:, ['date', 'code', 'size']].copy()
        factor = pd.merge(factor, industry, on=['date', 'code'])
        factor = pd.merge(factor, size, on=['date', 'code'])
        dates = factor['date'].unique()
        split_dates = np.array_split(dates, 10)
        with parallel_backend('multiprocessing', n_jobs=4):
            parallel_res = Parallel()(delayed(DataProcess.JOB_cross_section_remove_outlier)
                                      (factor, factor_field, dates) for dates in split_dates)
        factor = pd.concat(parallel_res)
        dates = factor['date'].unique()
        split_dates = np.array_split(dates, 10)
        with parallel_backend('multiprocessing', n_jobs=4):
            parallel_res = Parallel()(delayed(DataProcess.JOB_cross_section_Z_score)
                                      (factor, factor_field, dates) for dates in split_dates)
        factor = pd.concat(parallel_res)
        dates = factor['date'].unique()
        split_dates = np.array_split(dates, 10)
        with parallel_backend('multiprocessing', n_jobs=4):
            parallel_res = Parallel()(delayed(DataProcess.JOB_cross_section_Z_score)
                                      (factor, 'size', dates) for dates in split_dates)
        factor = pd.concat(parallel_res)
        dates = factor['date'].unique()
        split_dates = np.array_split(dates, 10)
        with parallel_backend('multiprocessing', n_jobs=4):
            parallel_res = Parallel()(delayed(DataProcess.JOB_neutralize)
                                      (factor, factor_field, dates) for dates in split_dates)
        neutral_factor = pd.concat(parallel_res)
        return neutral_factor

    # ----------------JOB开头的函数是其他函数多进程时调用的工具函数---------------------
    # ----------------data中的date在columns里---------------------
    @staticmethod
    def JOB_cross_section_Z_score(data, field, dates):
        res = []
        for date in dates:
            day_data = data.loc[data['date'] == date, :].copy()
            # 滤去周末出财报造成周末有因子的情况
            if day_data.shape[0] < 100:
                pass
            else:
                day_data.loc[:, field] = DataProcess.Z_standardize(day_data[field])
                res.append(day_data)
        dates_data = pd.concat(res)
        return dates_data

    @staticmethod
    def JOB_cross_section_rank_Z(data, field, dates):
        res = []
        for date in dates:
            day_data = data.loc[data['date'] == date, :].copy()
            # 滤去周末出财报造成周末有因子的情况
            if day_data.shape[0] < 100:
                pass
            else:
                day_data.loc[:, field] = DataProcess.rank_standardize(day_data[field])
                res.append(day_data)
        dates_data = pd.concat(res)
        return dates_data

    @staticmethod
    def JOB_cross_section_remove_outlier(data, field, dates):
        res = []
        for date in dates:
            day_data = data.loc[data['date'] == date, :].copy()
            # 滤去周末出财报造成周末有因子的情况
            if day_data.shape[0] < 100:
                pass
            else:
                day_data.loc[:, field] = DataProcess.remove_outlier(day_data[field])
                res.append(day_data)
        dates_data = pd.concat(res)
        return dates_data

    @staticmethod
    def JOB_neutralize(data, factor_field, dates):
        res = []
        for date in dates:
            day_code = data.loc[data['date'] == date, 'code']
            day_factor = data.loc[data['date'] == date, factor_field]
            day_idsty_size = data.loc[data['date'] == date, data.columns.difference(['date', 'code', factor_field])]
            OLS_est = sm.OLS(day_factor, day_idsty_size).fit()
            day_neutral_factor = OLS_est.resid
            day_neutral_factor.name = factor_field
            # 得到正交化后的因子值
            day_neutral_factor = pd.concat([day_code, day_neutral_factor], axis=1)
            day_neutral_factor['date'] = date
            res.append(day_neutral_factor)
        res_data = pd.concat(res)
        return res_data


if __name__ == '__main__':
    from rdf_data import rdf_data
    from influxdb_data import influxdbData

    rdf = rdf_data()
    influx = influxdbData()
    mkt_data = influx.getDataMultiprocess('DailyData_Gus', 'marketData', 20190101, 20190501, None)
    size_data = influx.getDataMultiprocess('DailyFactor_Gus', 'Size', 20190101, 20190501, None)
    factor = influx.getDataMultiprocess('DailyFactor_Gus', 'Growth', 20190101, 20190501, ['code', 'EPcut_TTM_growthY'])
    a = DataProcess.neurtralize(factor, 'EPcut_TTM_growthY', mkt_data, size_data)
