from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
import datetime
from global_constant import N_JOBS
from joblib import Parallel, delayed, parallel_backend

class EPandEPcut(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'

    def cal_factors(self, start, end, n_jobs):
        net_profit_Q = self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_profit_Q', start, end, 
                                                       ['code', 'net_profit_Q', 'report_period'])
        net_profit_TTM = self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_profit_TTM', start, end,
                                                         ['code', 'net_profit_TTM', 'report_period'])
        net_profit_ddt_TTM = self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_profit_ddt_TTM', start, end,
                                                             ['code', 'net_profit_ddt_TTM', 'report_period'])
        market_cap = self.influx.getDataMultiprocess('DailyFactors_Gus', 'Size', start, end,
                                                     ['code', 'market_cap'])
        net_profit_Q.index.names = ['date']
        net_profit_Q.reset_index(inplace=True)
        net_profit_TTM.index.names = ['date']
        net_profit_TTM.reset_index(inplace=True)
        net_profit_ddt_TTM.index.names = ['date']
        net_profit_ddt_TTM.reset_index(inplace=True)
        market_cap.index.names = ['date']
        market_cap.reset_index(inplace=True)
        # ----------------------------------------------------------
        EP_Q = pd.merge(net_profit_Q, market_cap, on=['date', 'code'])
        EP_Q.set_index('date', inplace=True)
        EP_Q['EP_Q'] = EP_Q['net_profit_Q'] / EP_Q['market_cap'] / 10000
        EP_Q = EP_Q.loc[:, ['code', 'EP_Q', 'report_period']]
        EP_Q = EP_Q.dropna(subset=['EP_Q'])
        codes = EP_Q['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (EP_Q, 'code', codes, self.db, 'EP_Q') for codes in split_codes)
        print('EP_Q finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        # ----------------------------------------------------------
        # market_cap 的单位为万元
        EP = pd.merge(net_profit_TTM, market_cap, on=['date', 'code'])
        EP.set_index('date', inplace=True)
        EP['EP_TTM'] = EP['net_profit_TTM'] / EP['market_cap'] / 10000
        EP = EP.loc[:, ['code', 'EP_TTM', 'report_period']]
        EP = EP.dropna(subset=['EP_TTM'])
        codes = EP['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (EP, 'code', codes, self.db, 'EP') for codes in split_codes)
        print('EP_TTM finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        # ----------------------------------------------------------
        EPcut = pd.merge(net_profit_ddt_TTM, market_cap, on=['date', 'code'])
        EPcut.set_index('date', inplace=True)
        EPcut['EPcut_TTM'] = EPcut['net_profit_ddt_TTM'] / EPcut['market_cap'] / 10000
        EPcut = EPcut.loc[:, ['code', 'EPcut_TTM', 'report_period']]
        EPcut = EPcut.dropna(subset=['EPcut_TTM'])
        codes = EPcut['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (EPcut, 'code', codes, self.db, 'EPcut') for codes in split_codes)
        print('EPcut_TTM finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)
        return fail_list

if __name__ == '__main__':
    print(datetime.datetime.now())
    ep = EPandEPcut()
    r = ep.cal_factors(20090101, 20200601, N_JOBS)
    print('task finish')
    print(r)
    print(datetime.datetime.now())