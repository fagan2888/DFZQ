from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
import datetime
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS

class SP_M(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'SP_M_TTM'

    def cal_factors(self, start, end, n_jobs):
        oper_rev = self.influx.getDataMultiprocess('FinancialReport_Gus', 'oper_rev_M', start, end,
                                                     ['code', 'oper_rev_M_TTM', 'report_period'])
        market_cap = self.influx.getDataMultiprocess('DailyFactors_Gus', 'Size', start, end,
                                                     ['code', 'market_cap'])
        oper_rev.index.names = ['date']
        oper_rev.reset_index(inplace=True)
        market_cap.index.names = ['date']
        market_cap.reset_index(inplace=True)
        # market_cap 的单位为万元
        SP = pd.merge(oper_rev, market_cap, on=['date', 'code'])
        SP.set_index('date', inplace=True)
        SP['SP_M_TTM'] = SP['oper_rev_M_TTM'] / SP['market_cap'] / 10000
        SP = SP.loc[:, ['code', 'SP_M_TTM', 'report_period']]
        SP = SP.dropna(subset=['SP_M_TTM'])
        codes = SP['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (SP, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('SP_M finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list

if __name__ == '__main__':
    print(datetime.datetime.now())
    SP = SP_M()
    r = SP.cal_factors(20100101, 20200629, N_JOBS)
    print('task finish')
    print(r)
    print(datetime.datetime.now())