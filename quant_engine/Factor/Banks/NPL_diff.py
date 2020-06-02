from factor_base import FactorBase
import pandas as pd
import numpy as np
import datetime
from influxdb_data import influxdbData
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS


class NPL_diff(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'

    def cal_factors(self, start, end, n_jobs):
        fail_list = []
        NPL_diff = self.influx.getDataMultiprocess('FinancialReport_Gus', 'NPL', start, end,
                                                   ['code', 'report_period', 'NPL', 'NPL_last1Q'])
        NPL_diff['NPL_diffQ'] = NPL_diff['NPL'] - NPL_diff['NPL_last1Q']
        NPL_diff = NPL_diff.loc[pd.notnull(NPL_diff['NPL_diffQ']), ['code', 'report_period', 'NPL_diffQ']]
        NPL_diff.where(pd.notnull(NPL_diff), None)
        codes = NPL_diff['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (NPL_diff, 'code', codes, self.db, 'NPL_diffQ')
                             for codes in split_codes)
        for r in res:
            fail_list.extend(r)
        print('NPL_diff finish')
        print('-' * 30)
        return fail_list


if __name__ == '__main__':
    pd.set_option('mode.use_inf_as_na', True)
    start_dt = datetime.datetime.now()
    diff = NPL_diff()
    r = diff.cal_factors(20100101, 20200522, N_JOBS)
    print(r)
    print('time token:', datetime.datetime.now()-start_dt)