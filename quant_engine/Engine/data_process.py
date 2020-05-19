import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
from joblib import Parallel, delayed, parallel_backend
import warnings


class DataProcess:
    # -------------------------------------------------
    # 基础工具
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

    # -------------------------------------------------
    # 结果分析工具
    @staticmethod
    def calc_ann_return(series):
        return (series.iloc[-1] / series.iloc[0]) ** (250 / series.shape[0]) - 1

    @staticmethod
    def calc_alpha_ann_return(series1, series2):
        ret1 = (series1.iloc[-1] / series1.iloc[0]) ** (250 / series1.shape[0]) - 1
        ret2 = (series2.iloc[-1] / series2.iloc[0]) ** (250 / series2.shape[0]) - 1
        return ret1 - ret2

    @staticmethod
    def calc_accum_alpha(series1, series2):
        return (series1 - series2) / series2

    @staticmethod
    def calc_max_draw_down(series):
        warnings.filterwarnings("ignore")
        index_low = np.argmax(np.maximum.accumulate(series) - series)
        if index_low == 0:
            return 0
        else:
            index_high = np.argmax(series[:index_low])
            return (series[index_high] - series[index_low]) / series[index_high]

    @staticmethod
    def calc_alpha_max_draw_down(series1, series2):
        warnings.filterwarnings("ignore")
        accum_alpha = (series1 - series2) / series2
        index_low = np.argmax(np.maximum.accumulate(accum_alpha) - accum_alpha)
        value_low = accum_alpha[index_low]
        if index_low == 0:
            return 0
        else:
            value_high = np.max(accum_alpha[:index_low])
            return value_high - value_low

    @staticmethod
    def calc_sharpe(series):
        return math.sqrt(252) * series.pct_change().mean() / series.pct_change().std()

    @staticmethod
    def calc_alpha_sharpe(series1, series2):
        ret_series1 = series1.pct_change()
        ret_series2 = series2.pct_change()
        alpha = ret_series1 - ret_series2
        return math.sqrt(252) * alpha.mean() / alpha.std()

    # -------------------------------------------------
    # 数据处理工具
    @staticmethod
    def get_next_date(calendar, today, days):
        return {today: calendar[calendar > today].iloc[days - 1]}

    @staticmethod
    # 获取行业哑变量
    def get_industry_dummies(mkt_data, industry_field):
        df = mkt_data.dropna(subset=[industry_field])
        industry_data = pd.get_dummies(df[industry_field])
        industry_data = pd.concat([df['code'], industry_data], axis=1)
        return industry_data

    @staticmethod
    # 标准化
    def standardize(factor_data, factor_field, index_is_date: bool, n_process):
        if index_is_date:
            factor_data.index.names = ['date']
            factor_data.reset_index(inplace=True)
        dates = factor_data['date'].unique()
        split_dates = np.array_split(dates, n_process)
        with parallel_backend('multiprocessing', n_jobs=n_process):
            parallel_res = Parallel()(delayed(DataProcess.JOB_cross_section_Z_score)
                                      (factor_data, factor_field, dates) for dates in split_dates)
        factor = pd.concat(parallel_res)
        return factor

    @staticmethod
    # 去极值+标准化
    def remove_and_Z(factor_data, factor_field, index_is_date: bool, n_process):
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
    # 数据中不能有nan，否则答案全为nan
    # 返回的date在columns里
    def neutralize(factor_data, factor_field, industry_dummies, size_data, n_process=5):
        industry_dummies.index.names = ['date']
        factor_data.index.names = ['date']
        size_data.index.names = ['date']
        factor = factor_data.reset_index().loc[:, ['date', 'code', factor_field]].copy()
        industry = industry_dummies.reset_index()
        size = size_data.reset_index().loc[:, ['date', 'code', 'size']].copy()
        factor = pd.merge(factor, industry, on=['date', 'code'])
        factor = pd.merge(factor, size, on=['date', 'code'])
        factor = factor.dropna()
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
                                      (factor, 'size', dates) for dates in split_dates)
        factor = pd.concat(parallel_res)
        dates = factor['date'].unique()
        split_dates = np.array_split(dates, n_process)
        with parallel_backend('multiprocessing', n_jobs=n_process):
            parallel_res = Parallel()(delayed(DataProcess.JOB_neutralize)
                                      (factor, factor_field, dates) for dates in split_dates)
        factor = pd.concat(parallel_res)
        dates = factor['date'].unique()
        split_dates = np.array_split(dates, n_process)
        with parallel_backend('multiprocessing', n_jobs=n_process):
            parallel_res = Parallel()(delayed(DataProcess.JOB_cross_section_Z_score)
                                      (factor, factor_field, dates) for dates in split_dates)
        neutral_factor = pd.concat(parallel_res)
        return neutral_factor

    @staticmethod
    # 对所有风格因子和行业做中性
    # 数据中不能有nan，否则答案全为nan
    # 返回的date在columns里
    def neutralize_v2(factor_data, factor_field, risk_data, style='size', n_process=5):
        risk_data.index.names = ['date']
        factor_data.index.names = ['date']
        factor = factor_data.reset_index().loc[:, ['date', 'code', factor_field]].copy()
        # 只对 行业 和 市值 做中性化
        if style == 'size':
            risk_data = risk_data.loc[:, risk_data.columns.difference([
                'Beta', 'Cubic size', 'Growth', 'Liquidity', 'Market', 'SOE', 'Trend',
                'Uncertainty', 'Value', 'Volatility'])]
        # 对 行业 和 所有风格 做中性化
        else:
            risk_data = risk_data.loc[:, risk_data.columns.difference(['Market'])]
        factor = pd.merge(factor, risk_data.reset_index(), on=['date', 'code'])
        # 因子去极值
        dates = factor['date'].unique()
        split_dates = np.array_split(dates, n_process)
        with parallel_backend('multiprocessing', n_jobs=n_process):
            parallel_res = Parallel()(delayed(DataProcess.JOB_cross_section_remove_outlier)
                                      (factor, factor_field, dates) for dates in split_dates)
        factor = pd.concat(parallel_res)
        # 对 风格因子 做标准化
        dates = factor['date'].unique()
        split_dates = np.array_split(dates, n_process)
        # 如果 选股范围不为全市场 需要对风险矩阵重新标准化
        with parallel_backend('multiprocessing', n_jobs=n_process):
            parallel_res = Parallel()(delayed(DataProcess.JOB_neutralize)
                                      (factor, factor_field, dates) for dates in split_dates)
        factor = pd.concat(parallel_res)
        dates = factor['date'].unique()
        split_dates = np.array_split(dates, n_process)
        with parallel_backend('multiprocessing', n_jobs=n_process):
            parallel_res = Parallel()(delayed(DataProcess.JOB_cross_section_Z_score)
                                      (factor, factor_field, dates) for dates in split_dates)
        neutral_factor = pd.concat(parallel_res)
        return neutral_factor


    # -------------------------------------------------
    # JOB开头的函数是其他函数多进程时调用的工具函数---------------------
    # data中的date在columns里
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
        dts = []
        codes = []
        factors = []
        for date in dates:
            day_code = data.loc[data['date'] == date, 'code'].values
            dts.append([date] * day_code.shape[0])
            codes.append(day_code)
            day_factor = data.loc[data['date'] == date, factor_field].values
            day_idsty_size = data.loc[data['date'] == date,
                                      data.columns.difference(['date', 'code', factor_field])].values
            OLS_est = sm.OLS(day_factor, day_idsty_size).fit()
            day_neutral_factor = OLS_est.resid
            factors.append(day_neutral_factor)
        dts = np.concatenate(dts)
        codes = np.concatenate(codes)
        factors = np.concatenate(factors)
        res_data = pd.DataFrame({'date': dts, 'code': codes, factor_field: factors})
        return res_data


if __name__ == '__main__':
    from rdf_data import rdf_data
    from influxdb_data import influxdbData

    rdf = rdf_data()
    influx = influxdbData()
    mkt_data = influx.getDataMultiprocess('DailyData_Gus', 'marketData', 20190101, 20190501, None)
    size_data = influx.getDataMultiprocess('DailyFactor_Gus', 'Size', 20190101, 20190501, None)
    factor = influx.getDataMultiprocess('DailyFactor_Gus', 'Growth', 20190101, 20190501, ['code', 'EPcut_TTM_growthY'])
    a = DataProcess.neutralize(factor, 'EPcut_TTM_growthY', mkt_data, size_data)
