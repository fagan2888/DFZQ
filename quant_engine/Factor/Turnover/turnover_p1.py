#  动量因子 return_Xm, wgt_return_Xm, exp_wgt_return_Xm 的计算

from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
import dateutil.parser as dtparser
from joblib import Parallel, delayed, parallel_backend
from dateutil.relativedelta import relativedelta
import math


class turnover_p1(FactorBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def job_factors(codes, df):
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_df = code_df.sort_index()
            start = code_df.index[0]
            m1 = []
            m3 = []
            m6 = []
            y2 = []
            for idx, row in code_df.iterrows():
                str_idx = idx.strftime('%Y%m%d')
                m1_before_idx = (idx - relativedelta(months=1)).strftime('%Y%m%d')
                m3_before_idx = (idx - relativedelta(months=3)).strftime('%Y%m%d')
                m6_before_idx = (idx - relativedelta(months=6)).strftime('%Y%m%d')
                y2_before_idx = (idx - relativedelta(years=2)).strftime('%Y%m%d')
                # ------------------------------------------------------------------------------------------
                period_m1_df = code_df.loc[m1_before_idx:str_idx, :].copy()
                if not period_m1_df.empty:
                    turn_1m = period_m1_df['turnover'].mean()
                    float_turn_1m = period_m1_df['float_turnover'].mean()
                    free_turn_1m = period_m1_df['free_turnover'].mean()
                    std_turn_1m = period_m1_df['turnover'].std()
                    std_float_turn_1m = period_m1_df['float_turnover'].std()
                    std_free_turn_1m = period_m1_df['free_turnover'].std()
                    m1.append(pd.Series([idx, code, turn_1m, float_turn_1m, free_turn_1m, std_turn_1m,
                                         std_float_turn_1m, std_free_turn_1m],
                                        index=['date', 'code', 'turn_1m', 'float_turn_1m', 'free_turn_1m',
                                               'std_turn_1m', 'std_float_turn_1m', 'std_free_turn_1m']))
                # -----------------------------------------------------------------------------------------
                period_m3_df = code_df.loc[m3_before_idx:str_idx, :].copy()
                if not period_m3_df.empty:
                    turn_3m = period_m3_df['turnover'].mean()
                    float_turn_3m = period_m3_df['float_turnover'].mean()
                    free_turn_3m = period_m3_df['free_turnover'].mean()
                    std_turn_3m = period_m3_df['turnover'].std()
                    std_float_turn_3m = period_m3_df['float_turnover'].std()
                    std_free_turn_3m = period_m3_df['free_turnover'].std()
                    m3.append(pd.Series([idx, code, turn_3m, float_turn_3m, free_turn_3m, std_turn_3m,
                                         std_float_turn_3m, std_free_turn_3m],
                                        index=['date', 'code', 'turn_3m', 'float_turn_3m', 'free_turn_3m',
                                               'std_turn_3m', 'std_float_turn_3m', 'std_free_turn_3m']))
                # -----------------------------------------------------------------------------------------
                period_m6_df = code_df.loc[m6_before_idx:str_idx, :].copy()
                if not period_m6_df.empty:
                    turn_6m = period_m6_df['turnover'].mean()
                    float_turn_6m = period_m6_df['float_turnover'].mean()
                    free_turn_6m = period_m6_df['free_turnover'].mean()
                    std_turn_6m = period_m6_df['turnover'].std()
                    std_float_turn_6m = period_m6_df['float_turnover'].std()
                    std_free_turn_6m = period_m6_df['free_turnover'].std()
                    m6.append(pd.Series([idx, code, turn_6m, float_turn_6m, free_turn_6m, std_turn_6m,
                                         std_float_turn_6m, std_free_turn_6m],
                                        index=['date', 'code', 'turn_6m', 'float_turn_6m', 'free_turn_6m',
                                               'std_turn_6m', 'std_float_turn_6m', 'std_free_turn_6m']))
                # ---------------------------------------------------------------------------------------
                period_y2_df = code_df.loc[y2_before_idx:str_idx, :].copy()
                if period_y2_df.shape[0] < 200:
                    pass
                else:
                    turn_2y = period_y2_df['turnover'].mean()
                    float_turn_2y = period_y2_df['float_turnover'].mean()
                    free_turn_2y = period_y2_df['free_turnover'].mean()
                    std_turn_2y = period_y2_df['turnover'].std()
                    std_float_turn_2y = period_y2_df['float_turnover'].std()
                    std_free_turn_2y = period_y2_df['free_turnover'].std()
                    if not period_m1_df.empty:
                        bias_turn_1m = turn_1m / turn_2y - 1
                        bias_float_turn_1m = float_turn_1m / float_turn_2y - 1
                        bias_free_turn_1m = free_turn_1m / free_turn_2y - 1
                        bias_std_turn_1m = std_turn_1m / std_turn_2y - 1
                        bias_std_float_turn_1m = std_float_turn_1m / std_float_turn_2y - 1
                        bias_std_free_turn_1m = std_free_turn_1m / std_free_turn_2y - 1
                    else:
                        bias_turn_1m = np.nan
                        bias_float_turn_1m = np.nan
                        bias_free_turn_1m = np.nan
                        bias_std_turn_1m = np.nan
                        bias_std_float_turn_1m = np.nan
                        bias_std_free_turn_1m = np.nan
                    if not period_m3_df.empty:
                        bias_turn_3m = turn_3m / turn_2y - 1
                        bias_float_turn_3m = float_turn_3m / float_turn_2y - 1
                        bias_free_turn_3m = free_turn_3m / free_turn_2y - 1
                        bias_std_turn_3m = std_turn_3m / std_turn_2y - 1
                        bias_std_float_turn_3m = std_float_turn_3m / std_float_turn_2y - 1
                        bias_std_free_turn_3m = std_free_turn_3m / std_free_turn_2y - 1
                    else:
                        bias_turn_3m = np.nan
                        bias_float_turn_3m = np.nan
                        bias_free_turn_3m = np.nan
                        bias_std_turn_3m = np.nan
                        bias_std_float_turn_3m = np.nan
                        bias_std_free_turn_3m = np.nan
                    if not period_m6_df.empty:
                        bias_turn_6m = turn_6m / turn_2y - 1
                        bias_float_turn_6m = float_turn_6m / float_turn_2y - 1
                        bias_free_turn_6m = free_turn_6m / free_turn_2y - 1
                        bias_std_turn_6m = std_turn_6m / std_turn_2y - 1
                        bias_std_float_turn_6m = std_float_turn_6m / std_float_turn_2y - 1
                        bias_std_free_turn_6m = std_free_turn_6m / std_free_turn_2y - 1
                    else:
                        bias_turn_6m = np.nan
                        bias_float_turn_6m = np.nan
                        bias_free_turn_6m = np.nan
                        bias_std_turn_6m = np.nan
                        bias_std_float_turn_6m = np.nan
                        bias_std_free_turn_6m = np.nan
                    y2.append(pd.Series([idx, code, bias_turn_1m, bias_turn_3m, bias_turn_6m, bias_float_turn_1m,
                                         bias_float_turn_3m, bias_float_turn_6m, bias_free_turn_1m, bias_free_turn_3m,
                                         bias_free_turn_6m, bias_std_turn_1m, bias_std_turn_3m, bias_std_turn_6m,
                                         bias_std_float_turn_1m, bias_std_float_turn_3m, bias_std_float_turn_6m,
                                         bias_std_free_turn_1m, bias_std_free_turn_3m, bias_std_free_turn_6m],
                                        index=['date', 'code', 'bias_turn_1m', 'bias_turn_3m', 'bias_turn_6m',
                                               'bias_float_turn_1m', 'bias_float_turn_3m', 'bias_float_turn_6m',
                                               'bias_free_turn_1m', 'bias_free_turn_3m', 'bias_free_turn_6m',
                                               'bias_std_turn_1m', 'bias_std_turn_3m', 'bias_std_turn_6m',
                                               'bias_std_float_turn_1m', 'bias_std_float_turn_3m',
                                               'bias_std_float_turn_6m',
                                               'bias_std_free_turn_1m', 'bias_std_free_turn_3m',
                                               'bias_std_free_turn_6m']))
            if m1:
                code_res_m1 = pd.concat(m1, axis=1).T
                code_res_m1.set_index('date', inplace=True)
                m1_after_start = (start + relativedelta(months=1)).strftime('%Y%m%d')
                code_res_m1 = code_res_m1.loc[m1_after_start:, :]
                code_res_m1.reset_index(inplace=True)
                code_merge = code_res_m1
            if m3:
                code_res_m3 = pd.concat(m3, axis=1).T
                code_res_m3.set_index('date', inplace=True)
                m3_after_start = (start + relativedelta(months=3)).strftime('%Y%m%d')
                code_res_m3 = code_res_m3.loc[m3_after_start:, :]
                code_res_m3.reset_index(inplace=True)
                code_merge = pd.merge(code_merge, code_res_m3, how='outer', on=['date', 'code'])
            if m6:
                code_res_m6 = pd.concat(m6, axis=1).T
                code_res_m6.set_index('date', inplace=True)
                m6_after_start = (start + relativedelta(months=6)).strftime('%Y%m%d')
                code_res_m6 = code_res_m6.loc[m6_after_start:, :]
                code_res_m6.reset_index(inplace=True)
                code_merge = pd.merge(code_merge, code_res_m6, how='outer', on=['date', 'code'])
            if y2:
                code_res_y2 = pd.concat(y2, axis=1).T
                code_res_y2.set_index('date', inplace=True)
                y2_after_start = (start + relativedelta(years=2)).strftime('%Y%m%d')
                code_res_y2 = code_res_y2.loc[y2_after_start:, :]
                code_res_y2.reset_index(inplace=True)
                code_merge = pd.merge(code_merge, code_res_y2, how='outer', on=['date', 'code'])
            code_merge.set_index('date', inplace=True)
            code_merge[code_merge.columns.difference(['code'])] = \
                code_merge[code_merge.columns.difference(['code'])].astype('float')
            code_merge = code_merge.where(pd.notnull(code_merge), None)
            # save
            print('code: %s' % code)
            influx = influxdbData()
            influx.saveData(code_merge, 'DailyFactor_Gus', 'Turnover')

    def cal_factors(self, start, end):
        turnover = self.influx.getDataMultiprocess('DailyData_Gus', 'indicators', start, end,
                                                   ['code', 'turnover', 'float_turnover', 'free_turnover'])
        print('turnover got')
        codes = turnover['code'].unique()
        split_codes = np.array_split(codes, 10)
        with parallel_backend('multiprocessing', n_jobs=4):
            Parallel()(delayed(turnover_p1.job_factors)
                       (codes, turnover) for codes in split_codes)


if __name__ == '__main__':
    i = turnover_p1()
    i.cal_factors(20100101, 20190901)
