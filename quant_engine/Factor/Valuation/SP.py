from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
import datetime
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS

class SP(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'SP'

    def cal_factors(self, start, end, n_jobs):
        oper_rev_Q = self.influx.getDataMultiprocess('FinancialReport_Gus', 'oper_rev_Q', start, end,
                                                     ['code', 'oper_rev_Q', 'report_period'])
        oper_rev_TTM = self.influx.getDataMultiprocess('FinancialReport_Gus', 'oper_rev_TTM', start, end,
                                                       ['code', 'oper_rev_TTM', 'report_period'])
        market_cap = self.influx.getDataMultiprocess('DailyFactors_Gus', 'Size', start, end,
                                                     ['code', 'market_cap'])
        oper_rev_Q.index.names = ['date']
        oper_rev_Q.reset_index(inplace=True)
        oper_rev_TTM.index.names = ['date']
        oper_rev_TTM.reset_index(inplace=True)
        market_cap.index.names = ['date']
        market_cap.reset_index(inplace=True)
        fail_list = []

        # 计算sp_Q
        SP_Q = pd.merge(oper_rev_Q, market_cap, on=['date', 'code'])
        SP_Q.set_index('date', inplace=True)
        # market_cap 的单位为万元
        SP_Q['SP_Q'] = SP_Q['oper_rev_Q'] / SP_Q['market_cap'] / 10000
        SP_Q = SP_Q.loc[:, ['code', 'SP_Q', 'report_period']]
        codes = SP_Q['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (SP_Q, 'code', codes, self.db, self.measure) for codes in split_codes)
        for r in res:
            fail_list.extend(r)
        print('SP_Q finish')
        print('-' * 30)

        # 计算SP
        SP = pd.merge(oper_rev_TTM, market_cap, on=['date', 'code'])
        SP.set_index('date', inplace=True)
        # market_cap 的单位为万元
        SP['SP'] = SP['oper_rev_TTM'] / SP['market_cap'] / 10000
        SP = SP.loc[:, ['code', 'SP', 'report_period']]
        codes = SP['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (SP, 'code', codes, self.db, self.measure) for codes in split_codes)
        for r in res:
            fail_list.extend(r)
        print('SP finish')
        print('-' * 30)
        return fail_list

if __name__ == '__main__':
    print(datetime.datetime.now())
    SP = SP()
    r = SP.cal_factors(20100101, 20200315, N_JOBS)
    print('task finish')
    print(r)
    print(datetime.datetime.now())