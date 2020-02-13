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
        return (series.iloc[-1] / series.iloc[0] - 1) * 250 / series.shape[0]

    @staticmethod
    def calc_alpha_ann_return(series1, series2):
        ret1 = (series1.iloc[-1] / series1.iloc[0] - 1) * 250 / series1.shape[0]
        ret2 = (series2.iloc[-1] / series2.iloc[0] - 1) * 250 / series2.shape[0]
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
    def add_next_period_return(data, calendar, days, benchmark):
        # 默认index是日期
        bm_dict = {'IH': '000016.SH', 'IF': '000300.SH', 'IC': '000905.SH'}
        mkt_data = data.copy()
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
        bm_close = mkt_data.loc[mkt_data['code'] == bm_dict[benchmark], ['date', 'close', field]].copy()
        bm_close.rename(columns={'close': 'benchmark_close'}, inplace=True)
        nxt_bm_close = bm_close.loc[:, ['date', 'benchmark_close']].copy()
        nxt_bm_close.rename(columns={'date': field, 'benchmark_close': 'next_benchmark_close'}, inplace=True)
        bm_return = pd.merge(bm_close, nxt_bm_close, on=field, how='left')
        bm_return['next_benchmark_return'] = bm_return['next_benchmark_close'] / bm_return['benchmark_close'] - 1
        bm_return = bm_return.loc[:, ['date', 'next_benchmark_return']]
        mkt_data = pd.merge(mkt_data, bm_return, how='left', on='date')
        mkt_data['next_period_alpha'] = mkt_data['next_period_return'] - mkt_data['next_benchmark_return']
        return mkt_data

    @staticmethod
    # 获取行业哑变量
    def get_industry_dummies(mkt_data, industry_field='improved_lv1'):
        df = mkt_data.dropna(subset=[industry_field])
        industry_data = pd.get_dummies(df[industry_field])
        industry_data = pd.concat([df['code'], industry_data], axis=1)
        return industry_data

    @staticmethod
    # 去极值+标准化
    def remove_and_Z(factor_data, factor_field, index_is_date: bool, n_process=5):
        if index_is_date:
            factor_data.index.names = ['date']
            factor_data.reset_index(inplace=True)
        dates = factor_data['date'].unique()
        split_dates = np.array_split(dates, n_process)
        with parallel_backend('multiprocessing', n_jobs=n_process):
            parallel_res = Parallel()(delayed(DataProcess.JOB_cross_section_remove_and_Z)
                                      (factor_data, factor_field, dates) for dates in split_dates)
        factor = pd.concat(parallel_res)
        return factor

    @staticmethod
    # 市值行业中性化: 因子先去极值标准化，ln市值标准化，行业变换哑变量，回归完取残差
    # 返回的date在columns里
    def neutralize(factor_data, factor_field, industry_dummies, size_data, size_field='ln_market_cap', n_process=5):
        industry_dummies.index.names = ['date']
        factor_data.index.names = ['date']
        size_data.index.names = ['date']
        size_data.rename(columns={size_field: 'size'}, inplace=True)
        factor = factor_data.reset_index().loc[:, ['date', 'code', factor_field]].copy()
        industry = industry_dummies.reset_index()
        size = size_data.reset_index().loc[:, ['date', 'code', 'size']].copy()
        factor = pd.merge(factor, industry, on=['date', 'code'])
        factor = pd.merge(factor, size, on=['date', 'code'])
        dates = factor['date'].unique()
        split_dates = np.array_split(dates, n_process)
        with parallel_backend('multiprocessing', n_jobs=n_process):
            parallel_res = Parallel()(delayed(DataProcess.JOB_cross_section_remove_outlier)
                                      (factor, factor_field, dates) for dates in split_dates)
        factor = pd.concat(parallel_res)
        dates = factor['date'].unique()
        split_dates = np.array_split(dates, n_process)
        with parallel_backend('multiprocessing', n_jobs=n_process):
            parallel_res = Parallel()(delayed(DataProcess.JOB_cross_section_Z_score)
                                      (factor, factor_field, dates) for dates in split_dates)
        factor = pd.concat(parallel_res)
        dates = factor['date'].unique()
        split_dates = np.array_split(dates, n_process)
        with parallel_backend('multiprocessing', n_jobs=n_process):
            parallel_res = Parallel()(delayed(DataProcess.JOB_cross_section_Z_score)
                                      (factor, 'size', dates) for dates in split_dates)
        factor = pd.concat(parallel_res)
        dates = factor['date'].unique()
        split_dates = np.array_split(dates, n_process)
        with parallel_backend('multiprocessing', n_jobs=n_process):
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

    # 同时处理remove outlier 和 Z_score
    @staticmethod
    def JOB_cross_section_remove_and_Z(data, field, dates):
        res = []
        for date in dates:
            day_data = data.loc[data['date'] == date, :].copy()
            # 滤去周末出财报造成周末有因子的情况
            if day_data.shape[0] < 100:
                pass
            else:
                day_data.loc[:, field] = DataProcess.remove_outlier(day_data[field])
                day_data.loc[:, field] = DataProcess.Z_standardize(day_data[field])
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
