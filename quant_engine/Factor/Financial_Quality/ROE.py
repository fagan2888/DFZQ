# 盈利能力因子 ROE,ROE_Q,ROE_ddt,ROE_ddt_Q的计算

from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
import datetime
from global_constant import N_JOBS
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData


class ROE_series(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'ROE'

    @staticmethod
    def JOB_factors(codes, df, cur_net_equity_field, pre_net_equity_field, net_profit_field, result_field, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_df[[cur_net_equity_field, pre_net_equity_field]] = \
                code_df[[cur_net_equity_field, pre_net_equity_field]].fillna(method='ffill', axis=1)
            code_df[result_field] = \
                code_df[net_profit_field] / (code_df[cur_net_equity_field] + code_df[pre_net_equity_field]) * 2
            code_df.set_index('date', inplace=True)
            code_df = code_df.loc[:, ['code', result_field]]
            code_df = code_df.dropna(subset=[result_field])
            print('code: %s' % code)
            if code_df.empty:
                continue
            r = influx.saveData(code_df, db, measure)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('%s Error: %s' % (result_field, r))
        return save_res

    def cal_factors(self, start, end, n_jobs):
        pd.set_option('mode.use_inf_as_na', True)
        net_equity = self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_equity', start, end,
                                                     ['code', 'net_equity', 'net_equity_last1Q', 'net_equity_lastY'])
        net_equity.index.names = ['date']
        net_equity.reset_index(inplace=True)
        fail_list = []

        # ROE_Q
        net_profit_Q = self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_profit_Q', start, end,
                                                       ['code', 'net_profit_Q'])
        net_profit_Q.index.names = ['date']
        net_profit_Q.reset_index(inplace=True)
        ROE_Q = pd.merge(net_equity.loc[:, ['date', 'code', 'net_equity', 'net_equity_last1Q']], net_profit_Q,
                         on=['date', 'code'])
        codes = ROE_Q['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(ROE_series.JOB_factors)
                             (codes, ROE_Q, 'net_equity', 'net_equity_last1Q', 'net_profit_Q', 'ROE_Q',
                              self.db, self.measure) for codes in split_codes)
        print('ROE_Q finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)
        
        # ROE
        net_profit_TTM = self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_profit_TTM', start, end,
                                                         ['code', 'net_profit_TTM'])
        net_profit_TTM.index.names = ['date']
        net_profit_TTM.reset_index(inplace=True)
        ROE = pd.merge(net_equity.loc[:, ['date', 'code', 'net_equity', 'net_equity_lastY']], net_profit_TTM,
                       on=['date', 'code'])
        codes = ROE['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(ROE_series.JOB_factors)
                             (codes, ROE, 'net_equity', 'net_equity_lastY', 'net_profit_TTM', 'ROE',
                              self.db, self.measure) for codes in split_codes)
        print('ROE finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)
        
        # ROE_ddt_Q
        net_profit_ddt_Q = self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_profit_ddt_Q', start, end,
                                                           ['code', 'net_profit_ddt_Q'])
        net_profit_ddt_Q.index.names = ['date']
        net_profit_ddt_Q.reset_index(inplace=True)
        ROE_ddt_Q = pd.merge(net_equity.loc[:, ['date', 'code', 'net_equity', 'net_equity_last1Q']],
                             net_profit_ddt_Q, on=['date', 'code'])
        codes = ROE_ddt_Q['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(ROE_series.JOB_factors)
                             (codes, ROE_ddt_Q, 'net_equity', 'net_equity_last1Q', 'net_profit_ddt_Q', 'ROE_ddt_Q',
                              self.db, self.measure) for codes in split_codes)
        print('ROE_ddt_Q finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)

        # ROE_ddt
        net_profit_ddt_TTM = self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_profit_ddt_TTM', start, end,
                                                             ['code', 'net_profit_ddt_TTM'])
        net_profit_ddt_TTM.index.names = ['date']
        net_profit_ddt_TTM.reset_index(inplace=True)
        ROE_ddt = pd.merge(net_equity.loc[:, ['date', 'code', 'net_equity', 'net_equity_lastY']],
                           net_profit_ddt_TTM,
                           on=['date', 'code'])
        codes = ROE_ddt['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(ROE_series.JOB_factors)
                             (codes, ROE_ddt, 'net_equity', 'net_equity_lastY', 'net_profit_ddt_TTM', 'ROE_ddt',
                              self.db, self.measure) for codes in split_codes)
        print('ROE_ddt finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)

        return fail_list


if __name__ == '__main__':
    roe = ROE_series()
    r = roe.cal_factors(20100101, 20200205, N_JOBS)
    print('task finish')
    print(r)
    print(datetime.datetime.now())