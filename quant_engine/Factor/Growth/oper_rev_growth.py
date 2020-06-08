from factor_base import FactorBase
import pandas as pd
import numpy as np
import datetime
from influxdb_data import influxdbData
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS


class oper_rev_growth(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'

    @staticmethod
    def JOB_Q_growth(codes, df, factor, db, measure):
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
            cols = ['{0}_growthQ'.format(factor), '{0}_growthY'.format(factor)]
            for i in range(1, 8):
                code_df['{0}_growthY_last{1}Q'.format(factor, i)] = \
                    code_df.apply(lambda row: FactorBase.cal_growth(
                        row['{0}_last{1}Q'.format(factor, i+4)], row['{0}_last{1}Q'.format(factor, i)]), axis=1)
                cols.append('{0}_growthY_last{1}Q'.format(factor, i))
            code_df = code_df.loc[:, ['code', 'report_period'] + cols]
            code_df = code_df.replace(np.inf, np.nan)
            code_df = code_df.loc[np.any(pd.notnull(code_df[cols]), axis=1), :]
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

    @staticmethod
    def JOB_TTM_growth(codes, df, factor, db, measure):
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
            cols = ['{0}_growthQ'.format(factor), '{0}_growthY'.format(factor)]
            for i in range(1, 8):
                code_df['{0}_growthQ_last{1}Q'.format(factor, i)] = \
                    code_df.apply(lambda row: FactorBase.cal_growth(
                        row['{0}_last{1}Q'.format(factor, i + 1)], row['{0}_last{1}Q'.format(factor, i)]), axis=1)
                cols.append('{0}_growthQ_last{1}Q'.format(factor, i))
            code_df = code_df.loc[:, ['code', 'report_period'] + cols]
            code_df = code_df.replace(np.inf, np.nan)
            code_df = code_df.loc[np.any(pd.notnull(code_df[cols]), axis=1), :]
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
        # 计算 oper_rev_Q 的 growth
        oper_rev = self.influx.getDataMultiprocess('FinancialReport_Gus', 'oper_rev_Q', start, end)
        codes = oper_rev['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(oper_rev_growth.JOB_Q_growth)
                             (codes, oper_rev, 'oper_rev_Q', self.db, 'oper_rev_Q_growth')
                             for codes in split_codes)
        print('oper_rev_Q_growth finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)
        # 计算 oper_rev 的 growth
        oper_rev = self.influx.getDataMultiprocess('FinancialReport_Gus', 'oper_rev_TTM', start, end)
        codes = oper_rev['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(oper_rev_growth.JOB_TTM_growth)
                             (codes, oper_rev, 'oper_rev_TTM', self.db, 'oper_rev_TTM_growth')
                             for codes in split_codes)
        print('oper_rev_TTM_growth finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)

        return fail_list


if __name__ == '__main__':
    pd.set_option('mode.use_inf_as_na', True)
    start_dt = datetime.datetime.now()
    oper_revg = oper_rev_growth()
    r = oper_revg.cal_factors(20090101, 20200605, N_JOBS)
    print(r)
    print('time token:', datetime.datetime.now()-start_dt)