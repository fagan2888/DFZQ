from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
import datetime
from influxdb_data import influxdbData
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS


class ROE_growth(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'ROE_growth'

    @staticmethod
    def JOB_factors(codes, df, db, measure):
        pd.set_option('mode.use_inf_as_na', True)
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_df['ROE_Q_growthQ'] = \
                code_df.apply(lambda row: FactorBase.cal_growth(row['ROE_Q_last1Q'], row['ROE_Q']), axis=1)
            code_df['ROE_ddt_Q_growthQ'] = \
                code_df.apply(lambda row: FactorBase.cal_growth(row['ROE_ddt_Q_last1Q'], row['ROE_ddt_Q']), axis=1)
            code_df['ROE_Q_growthY'] = \
                code_df.apply(lambda row: FactorBase.cal_growth(row['ROE_Q_lastY'], row['ROE_Q']), axis=1)
            code_df['ROE_ddt_Q_growthY'] = \
                code_df.apply(lambda row: FactorBase.cal_growth(row['ROE_ddt_Q_lastY'], row['ROE_ddt_Q']), axis=1)
            code_df['ROE_growthQ'] = \
                code_df.apply(lambda row: FactorBase.cal_growth(row['ROE_last1Q'], row['ROE']), axis=1)
            code_df['ROE_ddt_growthQ'] = \
                code_df.apply(lambda row: FactorBase.cal_growth(row['ROE_ddt_last1Q'], row['ROE_ddt']), axis=1)
            code_df = code_df.loc[:, ['code', 'ROE_Q_growthQ', 'ROE_ddt_Q_growthQ', 'ROE_Q_growthY',
                                      'ROE_ddt_Q_growthY', 'ROE_growthQ', 'ROE_ddt_growthQ']]
            code_df = \
                code_df.loc[
                    pd.notnull(code_df['ROE_Q_growthQ']) | pd.notnull(code_df['ROE_ddt_Q_growthQ']) |
                    pd.notnull(code_df['ROE_Q_growthY']) | pd.notnull(code_df['ROE_ddt_Q_growthY']) |
                    pd.notnull(code_df['ROE_growthQ']) | pd.notnull(code_df['ROE_ddt_growthQ']), :]
            if code_df.empty:
                continue
            code_df = code_df.where(pd.notnull(code_df), None)
            print('code: %s' % code)
            r = influx.saveData(code_df, db, measure)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('ROE_growth Error: %s' % r)
        return save_res

    def cal_factors(self, start, end, n_jobs):
        pd.set_option('mode.use_inf_as_na', True)
        ROE = self.influx.getDataMultiprocess('DailyFactors_Gus', 'ROE', start, end)
        codes = ROE['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(ROE_growth.JOB_factors)
                             (codes, ROE, self.db, self.measure) for codes in split_codes)
        print('ROE_growth finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    pd.set_option('mode.use_inf_as_na', True)
    start_dt = datetime.datetime.now()
    ROEg = ROE_growth()
    r = ROEg.cal_factors(20100101, 20200205, N_JOBS)
    print(r)
    print('time token:', datetime.datetime.now()-start_dt)