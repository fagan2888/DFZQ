from factor_base import FactorBase
import pandas as pd
import numpy as np
import datetime
from influxdb_data import influxdbData
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS


class NonInterestInc_growth(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'

    @staticmethod
    def JOB_factors(codes, df, factor, db, measure):
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
                code_df.loc[:, ['code', 'report_period', '{0}_growthQ'.format(factor), '{0}_growthY'.format(factor)]]
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
        # 计算 non_interest_income_Q 的 growth
        non_income = self.influx.getDataMultiprocess(
            'FinancialReport_Gus', 'non_interest_income_Q', start, end,
            ['code', 'non_interest_income_Q', 'non_interest_income_Q_last1Q', 'non_interest_income_Q_last4Q',
             'report_period'])
        codes = non_income['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(NonInterestInc_growth.JOB_factors)
                             (codes, non_income, 'non_interest_income_Q', self.db, 'non_interest_income_Q_growth')
                             for codes in split_codes)
        print('non_interest_income_Q_growth finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)
        # 计算 non_interest_income_TTM 的 growth
        non_income = self.influx.getDataMultiprocess(
            'FinancialReport_Gus', 'non_interest_income_TTM', start, end,
            ['code', 'non_interest_income_TTM', 'non_interest_income_TTM_last1Q', 'non_interest_income_TTM_last4Q',
             'report_period'])
        codes = non_income['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(NonInterestInc_growth.JOB_factors)
                             (codes, non_income, 'non_interest_income_TTM', self.db, 'non_interest_income_TTM_growth')
                             for codes in split_codes)
        print('non_interest_income_TTM_growth finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)

        return fail_list


if __name__ == '__main__':
    pd.set_option('mode.use_inf_as_na', True)
    start_dt = datetime.datetime.now()
    g = NonInterestInc_growth()
    r = g.cal_factors(20090101, 20200602, N_JOBS)
    print(r)
    print('time token:', datetime.datetime.now()-start_dt)