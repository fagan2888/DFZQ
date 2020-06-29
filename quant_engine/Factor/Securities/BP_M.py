from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
import datetime
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS

class BP_M(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'BP_M'

    def cal_factors(self, start, end, n_jobs):
        net_equity = self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_equity_M', start, end,
                                                     ['code', 'net_equity_M', 'report_period'])
        market_cap = self.influx.getDataMultiprocess('DailyFactors_Gus', 'Size', start, end,
                                                     ['code', 'market_cap'])
        net_equity.index.names = ['date']
        net_equity.reset_index(inplace=True)
        market_cap.index.names = ['date']
        market_cap.reset_index(inplace=True)
        # market_cap 的单位为万元
        BP = pd.merge(net_equity, market_cap, on=['date', 'code'])
        BP.set_index('date', inplace=True)
        BP['BP_M'] = BP['net_equity_M'] / BP['market_cap'] / 10000
        BP = BP.loc[:, ['code', 'BP_M', 'report_period']]
        BP = BP.dropna(subset=['BP_M'])
        codes = BP['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (BP, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('BP_M finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list

if __name__ == '__main__':
    print(datetime.datetime.now())
    bp = BP_M()
    r = bp.cal_factors(20100101, 20200629, N_JOBS)
    print('task finish')
    print(r)
    print(datetime.datetime.now())