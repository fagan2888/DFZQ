from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
import datetime
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS


class NCFP(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'NCFP'

    def cal_factors(self, start, end, n_jobs):
        net_CF = self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_CF_TTM', start, end,
                                                 ['code', 'net_CF_TTM', 'report_period'])
        market_cap = self.influx.getDataMultiprocess('DailyFactors_Gus', 'Size', start, end,
                                                     ['code', 'market_cap'])
        net_CF.index.names = ['date']
        net_CF.reset_index(inplace=True)
        market_cap.index.names = ['date']
        market_cap.reset_index(inplace=True)
        # market_cap 的单位为万元
        NCFP = pd.merge(net_CF, market_cap, on=['date', 'code'])
        NCFP.set_index('date', inplace=True)
        NCFP['NCFP'] = NCFP['net_CF_TTM'] / NCFP['market_cap'] / 10000
        NCFP = NCFP.loc[:, ['code', 'NCFP', 'report_period']]
        NCFP = NCFP.dropna(subset=['NCFP'])
        codes = NCFP['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (NCFP, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('NCFP finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    print(datetime.datetime.now())
    NCFP = NCFP()
    r = NCFP.cal_factors(20100101, 20200205, N_JOBS)
    print('task finish')
    print(r)
    print(datetime.datetime.now())
