from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
import datetime
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS

class NPL_leverage(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'NPL_leverage'

    def cal_factors(self, start, end, n_jobs):
        NPL = self.influx.getDataMultiprocess('DailyFactors_Gus', 'NPL', start, end, ['code', 'NPL'])
        CA_ratio = self.influx.getDataMultiprocess('DailyFactors_Gus', 'CA_ratio', start, end, ['code', 'CA_ratio'])
        NPL.index.names = ['date']
        NPL.reset_index(inplace=True)
        CA_ratio.index.names = ['date']
        CA_ratio.reset_index(inplace=True)
        NPL_leverage = pd.merge(NPL, CA_ratio, on=['date', 'code'])
        NPL_leverage.set_index('date', inplace=True)
        NPL_leverage['NPL_leverage'] = NPL_leverage['NPL'] / NPL_leverage['CA_ratio']
        NPL_leverage = NPL_leverage.loc[:, ['code', 'NPL_leverage']]
        codes = NPL_leverage['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (NPL_leverage, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('NPL_leverage finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list

if __name__ == '__main__':
    print(datetime.datetime.now())
    NPL_leverage = NPL_leverage()
    r = NPL_leverage.cal_factors(20100101, 20200520, N_JOBS)
    print('task finish')
    print(r)
    print(datetime.datetime.now())