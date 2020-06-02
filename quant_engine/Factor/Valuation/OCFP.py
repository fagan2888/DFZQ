from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
import datetime
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS


class OCFP(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'OCFP'

    def cal_factors(self, start, end, n_jobs):
        net_OCF = self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_OCF_TTM', start, end,
                                                 ['code', 'net_OCF_TTM', 'report_period'])
        market_cap = self.influx.getDataMultiprocess('DailyFactors_Gus', 'Size', start, end,
                                                     ['code', 'market_cap'])
        net_OCF.index.names = ['date']
        net_OCF.reset_index(inplace=True)
        market_cap.index.names = ['date']
        market_cap.reset_index(inplace=True)
        # market_cap 的单位为万元
        OCFP = pd.merge(net_OCF, market_cap, on=['date', 'code'])
        OCFP.set_index('date', inplace=True)
        OCFP['OCFP'] = OCFP['net_OCF_TTM'] / OCFP['market_cap'] / 10000
        OCFP = OCFP.loc[:, ['code', 'OCFP', 'report_period']]
        OCFP = OCFP.dropna(subset=['OCFP'])
        codes = OCFP['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (OCFP, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('OCFP finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    print(datetime.datetime.now())
    OCFP = OCFP()
    r = OCFP.cal_factors(20100101, 20200205, N_JOBS)
    print('task finish')
    print(r)
    print(datetime.datetime.now())
