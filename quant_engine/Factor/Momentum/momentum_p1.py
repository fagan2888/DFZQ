#  动量因子 return_Xm, wgt_return_Xm, exp_wgt_return_Xm 的计算

from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
import dateutil.parser as dtparser
from joblib import Parallel, delayed, parallel_backend
from dateutil.relativedelta import relativedelta
import math


class momentum_p1(FactorBase):
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
            m12 = []
            for idx, row in code_df.iterrows():
                str_idx = idx.strftime('%Y%m%d')
                m1_before_idx = (idx - relativedelta(months=1)).strftime('%Y%m%d')
                m3_before_idx = (idx - relativedelta(months=3)).strftime('%Y%m%d')
                m6_before_idx = (idx - relativedelta(months=6)).strftime('%Y%m%d')
                m12_before_idx = (idx - relativedelta(months=12)).strftime('%Y%m%d')
                # ------------------------------------------------------------------------------------------
                period_m1_df = code_df.loc[m1_before_idx:str_idx, :].copy()
                if period_m1_df.shape[0] < 10:
                    pass
                else:
                    period_m1_df['n_days_to_idx'] = range(period_m1_df.shape[0] - 1, -1, -1)
                    period_m1_df['exp_wgt'] = period_m1_df['n_days_to_idx'].apply(lambda x: math.exp(x * -1 / 1 / 4))
                    rtn_m1 = (period_m1_df['close'].iloc[-1] * period_m1_df['adj_factor'].iloc[-1]) \
                             / (period_m1_df['preclose'].iloc[0] * period_m1_df['adj_factor'].iloc[0]) - 1
                    wgt_rtn_m1 = \
                        period_m1_df['wgt_return'].sum() / period_m1_df['turnover'].sum()
                    float_wgt_rtn_m1 = \
                        period_m1_df['float_wgt_return'].sum() / period_m1_df['float_turnover'].sum()
                    free_wgt_rtn_m1 = \
                        period_m1_df['free_wgt_return'].sum() / period_m1_df['free_turnover'].sum()
                    exp_wgt_rtn_m1 = \
                        (period_m1_df['return'] * period_m1_df['turnover'] * period_m1_df['exp_wgt']).sum() \
                        / (period_m1_df['turnover'] * period_m1_df['exp_wgt']).sum()
                    float_exp_wgt_rtn_m1 = \
                        (period_m1_df['return'] * period_m1_df['float_turnover'] * period_m1_df['exp_wgt']).sum() \
                        / (period_m1_df['float_turnover'] * period_m1_df['exp_wgt']).sum()
                    free_exp_wgt_rtn_m1 = \
                        (period_m1_df['return'] * period_m1_df['free_turnover'] * period_m1_df['exp_wgt']).sum() \
                        / (period_m1_df['free_turnover'] * period_m1_df['exp_wgt']).sum()
                    m1.append(pd.Series([idx, code, rtn_m1, wgt_rtn_m1, float_wgt_rtn_m1, free_wgt_rtn_m1,
                                         exp_wgt_rtn_m1, float_exp_wgt_rtn_m1, free_exp_wgt_rtn_m1],
                                        index=['date', 'code', 'rtn_m1', 'wgt_rtn_m1', 'float_wgt_rtn_m1',
                                               'free_wgt_rtn_m1', 'exp_wgt_rtn_m1', 'float_exp_wgt_rtn_m1',
                                               'free_exp_wgt_rtn_m1']))
                # -----------------------------------------------------------------------------------------
                period_m3_df = code_df.loc[m3_before_idx:str_idx, :].copy()
                if period_m3_df.shape[0] < 30:
                    pass
                else:
                    period_m3_df['n_days_to_idx'] = range(period_m3_df.shape[0] - 1, -1, -1)
                    period_m3_df['exp_wgt'] = period_m3_df['n_days_to_idx'].apply(lambda x: math.exp(x * -1 / 3 / 4))
                    rtn_m3 = (period_m3_df['close'].iloc[-1] * period_m3_df['adj_factor'].iloc[-1]) \
                             / (period_m3_df['preclose'].iloc[0] * period_m3_df['adj_factor'].iloc[0]) - 1
                    wgt_rtn_m3 = \
                        period_m3_df['wgt_return'].sum() / period_m3_df['turnover'].sum()
                    float_wgt_rtn_m3 = \
                        period_m3_df['float_wgt_return'].sum() / period_m3_df['float_turnover'].sum()
                    free_wgt_rtn_m3 = \
                        period_m3_df['free_wgt_return'].sum() / period_m3_df['free_turnover'].sum()
                    exp_wgt_rtn_m3 = \
                        (period_m3_df['return'] * period_m3_df['turnover'] * period_m3_df['exp_wgt']).sum() \
                        / (period_m3_df['turnover'] * period_m3_df['exp_wgt']).sum()
                    float_exp_wgt_rtn_m3 = \
                        (period_m3_df['return'] * period_m3_df['float_turnover'] * period_m3_df['exp_wgt']).sum() \
                        / (period_m3_df['float_turnover'] * period_m3_df['exp_wgt']).sum()
                    free_exp_wgt_rtn_m3 = \
                        (period_m3_df['return'] * period_m3_df['free_turnover'] * period_m3_df['exp_wgt']).sum() \
                        / (period_m3_df['free_turnover'] * period_m3_df['exp_wgt']).sum()
                    m3.append(pd.Series([idx, code, rtn_m3, wgt_rtn_m3, float_wgt_rtn_m3, free_wgt_rtn_m3,
                                         exp_wgt_rtn_m3, float_exp_wgt_rtn_m3, free_exp_wgt_rtn_m3],
                                        index=['date', 'code', 'rtn_m3', 'wgt_rtn_m3', 'float_wgt_rtn_m3',
                                               'free_wgt_rtn_m3', 'exp_wgt_rtn_m3', 'float_exp_wgt_rtn_m3',
                                               'free_exp_wgt_rtn_m3']))
                # -----------------------------------------------------------------------------------------
                period_m6_df = code_df.loc[m6_before_idx:str_idx, :].copy()
                if period_m6_df.shape[0] < 60:
                    pass
                else:
                    period_m6_df['n_days_to_idx'] = range(period_m6_df.shape[0] - 1, -1, -1)
                    period_m6_df['exp_wgt'] = period_m6_df['n_days_to_idx'].apply(lambda x: math.exp(x * -1 / 6 / 4))
                    rtn_m6 = (period_m6_df['close'].iloc[-1] * period_m6_df['adj_factor'].iloc[-1]) \
                             / (period_m6_df['preclose'].iloc[0] * period_m6_df['adj_factor'].iloc[0]) - 1
                    wgt_rtn_m6 = \
                        period_m6_df['wgt_return'].sum() / period_m6_df['turnover'].sum()
                    float_wgt_rtn_m6 = \
                        period_m6_df['float_wgt_return'].sum() / period_m6_df['float_turnover'].sum()
                    free_wgt_rtn_m6 = \
                        period_m6_df['free_wgt_return'].sum() / period_m6_df['free_turnover'].sum()
                    exp_wgt_rtn_m6 = \
                        (period_m6_df['return'] * period_m6_df['turnover'] * period_m6_df['exp_wgt']).sum() \
                        / (period_m6_df['turnover'] * period_m6_df['exp_wgt']).sum()
                    float_exp_wgt_rtn_m6 = \
                        (period_m6_df['return'] * period_m6_df['float_turnover'] * period_m6_df['exp_wgt']).sum() \
                        / (period_m6_df['float_turnover'] * period_m6_df['exp_wgt']).sum()
                    free_exp_wgt_rtn_m6 = \
                        (period_m6_df['return'] * period_m6_df['free_turnover'] * period_m6_df['exp_wgt']).sum() \
                        / (period_m6_df['free_turnover'] * period_m6_df['exp_wgt']).sum()
                    m6.append(pd.Series([idx, code, rtn_m6, wgt_rtn_m6, float_wgt_rtn_m6, free_wgt_rtn_m6,
                                         exp_wgt_rtn_m6, float_exp_wgt_rtn_m6, free_exp_wgt_rtn_m6],
                                        index=['date', 'code', 'rtn_m6', 'wgt_rtn_m6', 'float_wgt_rtn_m6',
                                               'free_wgt_rtn_m6', 'exp_wgt_rtn_m6', 'float_exp_wgt_rtn_m6',
                                               'free_exp_wgt_rtn_m6']))
                # ---------------------------------------------------------------------------------------
                period_m12_df = code_df.loc[m12_before_idx:str_idx, :].copy()
                if period_m12_df.shape[0] < 120:
                    pass
                else:
                    period_m12_df['n_days_to_idx'] = range(period_m12_df.shape[0] - 1, -1, -1)
                    period_m12_df['exp_wgt'] = period_m12_df['n_days_to_idx'].apply(lambda x: math.exp(x * -1 / 12 / 4))
                    rtn_m12 = (period_m12_df['close'].iloc[-1] * period_m12_df['adj_factor'].iloc[-1]) \
                              / (period_m12_df['preclose'].iloc[0] * period_m12_df['adj_factor'].iloc[0]) - 1
                    wgt_rtn_m12 = \
                        period_m12_df['wgt_return'].sum() / period_m12_df['turnover'].sum()
                    float_wgt_rtn_m12 = \
                        period_m12_df['float_wgt_return'].sum() / period_m12_df['float_turnover'].sum()
                    free_wgt_rtn_m12 = \
                        period_m12_df['free_wgt_return'].sum() / period_m12_df['free_turnover'].sum()
                    exp_wgt_rtn_m12 = \
                        (period_m12_df['return'] * period_m12_df['turnover'] * period_m12_df['exp_wgt']).sum() \
                        / (period_m12_df['turnover'] * period_m12_df['exp_wgt']).sum()
                    float_exp_wgt_rtn_m12 = \
                        (period_m12_df['return'] * period_m12_df['float_turnover'] * period_m12_df['exp_wgt']).sum() \
                        / (period_m12_df['float_turnover'] * period_m12_df['exp_wgt']).sum()
                    free_exp_wgt_rtn_m12 = \
                        (period_m12_df['return'] * period_m12_df['free_turnover'] * period_m12_df['exp_wgt']).sum() \
                        / (period_m12_df['free_turnover'] * period_m12_df['exp_wgt']).sum()
                    m12.append(pd.Series([idx, code, rtn_m12, wgt_rtn_m12, float_wgt_rtn_m12, free_wgt_rtn_m12,
                                          exp_wgt_rtn_m12, float_exp_wgt_rtn_m12, free_exp_wgt_rtn_m12],
                                         index=['date', 'code', 'rtn_m12', 'wgt_rtn_m12', 'float_wgt_rtn_m12',
                                                'free_wgt_rtn_m12', 'exp_wgt_rtn_m12', 'float_exp_wgt_rtn_m12',
                                                'free_exp_wgt_rtn_m12']))
            code_res_m1 = pd.concat(m1, axis=1).T
            code_res_m1.set_index('date', inplace=True)
            m1_after_start = (start + relativedelta(months=1)).strftime('%Y%m%d')
            code_res_m1 = code_res_m1.loc[m1_after_start:, :]
            code_res_m1.reset_index(inplace=True)
            code_res_m3 = pd.concat(m3, axis=1).T
            code_res_m3.set_index('date', inplace=True)
            m3_after_start = (start + relativedelta(months=3)).strftime('%Y%m%d')
            code_res_m3 = code_res_m3.loc[m3_after_start:, :]
            code_res_m3.reset_index(inplace=True)
            code_res_m6 = pd.concat(m6, axis=1).T
            code_res_m6.set_index('date', inplace=True)
            m6_after_start = (start + relativedelta(months=6)).strftime('%Y%m%d')
            code_res_m6 = code_res_m6.loc[m6_after_start:, :]
            code_res_m6.reset_index(inplace=True)
            code_res_m12 = pd.concat(m12, axis=1).T
            code_res_m12.set_index('date', inplace=True)
            m12_after_start = (start + relativedelta(months=12)).strftime('%Y%m%d')
            code_res_m12 = code_res_m12.loc[m12_after_start:, :]
            code_res_m12.reset_index(inplace=True)
            if not code_res_m1.empty:
                code_merge = code_res_m1
            if not code_res_m3.empty:
                code_merge = pd.merge(code_merge, code_res_m3, how='outer', on=['date', 'code'])
            if not code_res_m6.empty:
                code_merge = pd.merge(code_merge, code_res_m6, how='outer', on=['date', 'code'])
            if not code_res_m12.empty:
                code_merge = pd.merge(code_merge, code_res_m12, how='outer', on=['date', 'code'])
            code_merge.set_index('date')
            code_merge = code_merge.where(pd.notnull(code_merge), None)
            # save
            influx = influxdbData()
            influx.saveData(code_merge, 'DailyFactor_Gus', 'Momentum')

    def cal_factors(self, start, end):
        daily_data = self.influx.getDataMultiprocess('DailyData_Gus', 'marketData', start, end,
                                                     ['code', 'adj_factor', 'close', 'preclose'])
        print('daily data got')
        turnover = self.influx.getDataMultiprocess('DailyData_Gus', 'indicators', start, end,
                                                   ['code', 'turnover', 'float_turnover', 'free_turnover'])
        print('turnover got')
        daily_data.index.names = ['date']
        turnover.index.names = ['date']
        daily_data.reset_index(inplace=True)
        turnover.reset_index(inplace=True)
        merge = pd.merge(daily_data, turnover, on=['date', 'code'])
        merge['return'] = merge['close'] / merge['preclose'] - 1
        merge['wgt_return'] = merge['turnover'] * merge['return']
        merge['float_wgt_return'] = merge['float_turnover'] * merge['return']
        merge['free_wgt_return'] = merge['free_turnover'] * merge['return']
        merge.set_index('date', inplace=True)
        codes = merge['code'].unique()
        split_codes = np.array_split(codes, 10)
        with parallel_backend('multiprocessing', n_jobs=4):
            Parallel()(delayed(momentum_p1.job_factors)
                       (codes, merge) for codes in split_codes)


if __name__ == '__main__':
    i = momentum_p1()
    i.cal_factors(20100101, 20190901)
