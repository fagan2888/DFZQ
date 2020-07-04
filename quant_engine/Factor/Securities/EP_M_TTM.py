from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
import datetime
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS

class EP_M(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'EP_M_TTM'

    def cal_factors(self, start, end, n_jobs):
        net_profit = self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_profit_M', start, end,
                                                     ['code', 'net_profit_M_TTM', 'report_period'])
        market_cap = self.influx.getDataMultiprocess('DailyFactors_Gus', 'Size', start, end,
                                                     ['code', 'market_cap', 'float_market_cap', 'free_market_cap'])
        net_profit.index.names = ['date']
        net_profit.reset_index(inplace=True)
        market_cap.index.names = ['date']
        market_cap.reset_index(inplace=True)
        # market_cap 的单位为万元
        EP = pd.merge(net_profit, market_cap, on=['date', 'code'])
        EP.set_index('date', inplace=True)
        EP['EP_M_TTM'] = EP['net_profit_M_TTM'] / EP['market_cap'] / 10000
        EP['EP_M_TTM_float'] = EP['net_profit_M_TTM'] / EP['float_market_cap'] / 10000
        EP['EP_M_TTM_free'] = EP['net_profit_M_TTM'] / EP['free_market_cap'] / 10000
        cols = ['EP_M_TTM', 'EP_M_TTM_float', 'EP_M_TTM_free']
        EP = EP.loc[np.any(pd.notnull(EP[cols]), axis=1), ['code', 'report_period'] + cols]
        EP = EP.where(pd.notnull(EP), None)
        codes = EP['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (EP, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('EP_M finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list

if __name__ == '__main__':
    print(datetime.datetime.now())
    ep = EP_M()
    r = ep.cal_factors(20100101, 20200629, N_JOBS)
    print('task finish')
    print(r)
    print(datetime.datetime.now())