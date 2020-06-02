from factor_base import FactorBase
import pandas as pd
import numpy as np
import datetime
from influxdb_data import influxdbData
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS


class provision_cov_growth(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'

    def cal_factors(self, start, end, n_jobs):
        fail_list = []
        pc_growth = self.influx.getDataMultiprocess(
            'FinancialReport_Gus', 'provision_cov', start, end,
            ['code', 'report_period', 'provision_cov', 'provision_cov_last1Q'])
        pc_growth['provision_cov_growthQ'] = pc_growth['provision_cov'] / pc_growth['provision_cov_last1Q'] - 1
        pc_growth = pc_growth.loc[pd.notnull(pc_growth['provision_cov_growthQ']),
                                  ['code', 'report_period', 'provision_cov_growthQ']]
        pc_growth.where(pd.notnull(pc_growth), None)
        codes = pc_growth['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (pc_growth, 'code', codes, self.db, 'provision_cov_growthQ') for codes in split_codes)
            for r in res:
                fail_list.extend(r)
            print('provision_cov_growthQ finish')
            print('-' * 30)
        return fail_list


if __name__ == '__main__':
    pd.set_option('mode.use_inf_as_na', True)
    start_dt = datetime.datetime.now()
    diff = provision_cov_growth()
    r = diff.cal_factors(20100101, 20200522, N_JOBS)
    print(r)
    print('time token:', datetime.datetime.now()-start_dt)