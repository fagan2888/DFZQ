from factor_base import FactorBase
import pandas as pd
import numpy as np
import datetime
from influxdb_data import influxdbData
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS


class TotLiabGrowth(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'

    @staticmethod
    def JOB_factor(codes, df, factor, db, measure):
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
            for i in range(1, 11):
                code_df['{0}_growthQ_last{1}Q'.format(factor, i)] = \
                    code_df.apply(lambda row: FactorBase.cal_growth(
                        row['{0}_last{1}Q'.format(factor, i + 1)], row['{0}_last{1}Q'.format(factor, i)]), axis=1)
                cols.append('{0}_growthQ_last{1}Q'.format(factor, i))
            for i in range(1, 8):
                code_df['{0}_growthY_last{1}Q'.format(factor, i)] = \
                    code_df.apply(lambda row: FactorBase.cal_growth(
                        row['{0}_last{1}Q'.format(factor, i + 4)], row['{0}_last{1}Q'.format(factor, i)]), axis=1)
                cols.append('{0}_growthY_last{1}Q'.format(factor, i))
            code_df = code_df.loc[:, ['code', 'report_period'] + cols]
            code_df = code_df.replace(np.inf, np.nan)
            code_df = code_df.replace(-np.inf, np.nan)
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
        # 计算 tot_liab 的 growth
        tot_liab = self.influx.getDataMultiprocess('FinancialReport_Gus', 'tot_liab', start, end)
        codes = tot_liab['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(TotLiabGrowth.JOB_factor)
                             (codes, tot_liab, 'tot_liab', self.db, 'tot_liab_growth')
                             for codes in split_codes)
        print('tot_liab_growth finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    pd.set_option('mode.use_inf_as_na', True)
    start_dt = datetime.datetime.now()
    g = TotLiabGrowth()
    r = g.cal_factors(20090101, 20200609, N_JOBS)
    print(r)
    print('time token:', datetime.datetime.now()-start_dt)