from factor_base import FactorBase
import pandas as pd
import numpy as np
import datetime
from influxdb_data import influxdbData
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS


class RNOA_growth(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'RNOA_growth'

    @staticmethod
    def JOB_factors(codes, df, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_df['RNOA_Q_growthY'] = \
                code_df.apply(lambda row: FactorBase.cal_growth(row['RNOA_Q_lastY'], row['RNOA_Q']), axis=1)
            code_df = code_df.loc[:, ['code', 'RNOA_Q_growthY']]
            code_df = code_df.replace(np.inf, np.nan)
            code_df = code_df.dropna()
            if code_df.empty:
                continue
            print('code: %s' % code)
            r = influx.saveData(code_df, db, measure)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('RNOA_growth Error: %s' % r)
        return save_res

    def cal_factors(self, start, end, n_jobs):
        pd.set_option('mode.use_inf_as_na', True)
        RNOA = self.influx.getDataMultiprocess('DailyFactors_Gus', 'RNOA', start, end)
        codes = RNOA['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(RNOA_growth.JOB_factors)
                             (codes, RNOA, self.db, self.measure) for codes in split_codes)
        print('RNOA_growth finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    pd.set_option('mode.use_inf_as_na', True)
    start_dt = datetime.datetime.now()
    RNOAg = RNOA_growth()
    r = RNOAg.cal_factors(20100101, 20200402, N_JOBS)
    print(r)
    print('time token:', datetime.datetime.now()-start_dt)