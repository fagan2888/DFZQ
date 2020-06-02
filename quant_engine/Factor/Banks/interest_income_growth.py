from factor_base import FactorBase
import pandas as pd
import numpy as np
import datetime
from influxdb_data import influxdbData
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS


class InterestIncome_growth(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'interest_income_growth'

    @staticmethod
    def JOB_factors(df, codes, factor, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_df['{0}_growthQ'.format(factor)] = \
                code_df.apply(lambda row: FactorBase.cal_growth(
                    row['{0}_last1Q'.format(factor)], row['{0}'.format(factor)]), axis=1)
            code_df['{0}_growthY'.format(factor)] = \
                code_df.apply(lambda row: FactorBase.cal_growth(
                    row['{0}_last4Q'.format(factor)], row['{0}'.format(factor)]), axis=1)
            code_df = \
                code_df.loc[:, ['code', '{0}_growthQ'.format(factor), '{0}_growthY'.format(factor), 'report_period']]
            code_df = code_df.replace(np.inf, np.nan)
            code_df = code_df.loc[pd.notnull(code_df['{0}_growthQ'.format(factor)]) |
                                  pd.notnull(code_df['{0}_growthY'.format(factor)]), :]
            if code_df.empty:
                continue
            code_df = code_df.where(pd.notnull(code_df), None)
            print('code: %s' % code)
            r = influx.saveData(code_df, db, measure)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('%s_growth Error: %s' % (factor, r))
        return save_res

    def cal_factors(self, start, end, n_jobs):
        pd.set_option('mode.use_inf_as_na', True)
        fail_list = []
        interest_inc = self.influx.getDataMultiprocess(
            'FinancialReport_Gus', 'interest_income', start, end,
            ['code', 'report_period', 'interest_income', 'interest_income_last1Q', 'interest_income_last4Q'])
        codes = interest_inc['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(InterestIncome_growth.JOB_factors)
                             (interest_inc, codes, 'interest_income', self.db, self.measure) for codes in split_codes)
        print('interest_income_growth finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    start_dt = datetime.datetime.now()
    growth = InterestIncome_growth()
    r = growth.cal_factors(20150101, 20160522, N_JOBS)
    print(r)
    print('time token:', datetime.datetime.now()-start_dt)