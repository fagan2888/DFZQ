from factor_base import FactorBase
import pandas as pd
import numpy as np
import datetime
from influxdb_data import influxdbData
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS


class EBITMarginDelta(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'

    @staticmethod
    def JOB_factor(codes, df, factor_field, n_Qs, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            cols = []
            for i in range(n_Qs):
                code_df['{0}_deltaQ_last{1}Q'.format(factor_field, i)] = \
                    code_df['{0}_last{1}Q'.format(factor_field, i)] - code_df['{0}_last{1}Q'.format(factor_field, i+1)]
                cols.append('{0}_deltaQ_last{1}Q'.format(factor_field, i))
            for i in range(n_Qs-3):
                code_df['{0}_deltaY_last{1}Q'.format(factor_field, i)] = \
                    code_df['{0}_last{1}Q'.format(factor_field, i)] - code_df['{0}_last{1}Q'.format(factor_field, i+4)]
                cols.append('{0}_deltaY_last{1}Q'.format(factor_field, i))
            code_df = code_df.replace(np.inf, np.nan)
            code_df = code_df.replace(-np.inf, np.nan)
            code_df = code_df.loc[np.any(pd.notnull(code_df[cols]), axis=1), ['code', 'report_period'] + cols]
            if code_df.empty:
                continue
            code_df.rename(columns={'{0}_deltaQ_last0Q'.format(factor_field): '{0}_deltaQ'.format(factor_field),
                                    '{0}_deltaY_last0Q'.format(factor_field): '{0}_deltaY'.format(factor_field)},
                           inplace=True)
            code_df = code_df.where(pd.notnull(code_df), None)
            print('code: %s' % code)
            r = influx.saveData(code_df, db, measure)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('%s Error: %s' % (measure, r))
        return save_res

    def cal_factors(self, start, end, n_jobs):
        fail_list = []
        # get EBITMargin_Q
        save_db = 'EBITmargin_Q_delta'
        margin = self.influx.getDataMultiprocess('DailyFactors_Gus', 'EBITmargin_Q', start, end)
        margin.rename(columns={'EBITmargin_Q': 'EBITmargin_Q_last0Q'}, inplace=True)
        codes = margin['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(EBITMarginDelta.JOB_factor)
                             (codes, margin, 'EBITmargin_Q', 9, self.db, save_db)
                             for codes in split_codes)
        print('EBITmargin_Q_delta finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)
        # -------------------------------------------------------------------------------
        # get EBITMargin_TTM
        save_db = 'EBITmargin_TTM_delta'
        margin = self.influx.getDataMultiprocess('DailyFactors_Gus', 'EBITmargin_TTM', start, end)
        margin.rename(columns={'EBITmargin_TTM': 'EBITmargin_TTM_last0Q'}, inplace=True)
        codes = margin['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(EBITMarginDelta.JOB_factor)
                             (codes, margin, 'EBITmargin_TTM', 6, self.db, save_db)
                             for codes in split_codes)
        print('EBITmargin_TTM_delta finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)
        # -------------------------------------------------------------------------------
        # get EBITMargin_latest
        save_db = 'EBITmargin_latest_delta'
        margin = self.influx.getDataMultiprocess('DailyFactors_Gus', 'EBITmargin_latest', start, end)
        margin.rename(columns={'EBITmargin_latest': 'EBITmargin_latest_last0Q'}, inplace=True)
        codes = margin['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(EBITMarginDelta.JOB_factor)
                             (codes, margin, 'EBITmargin_latest', 10, self.db, save_db)
                             for codes in split_codes)
        print('EBITmargin_latest_delta finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)

        return fail_list


if __name__ == '__main__':
    start_dt = datetime.datetime.now()
    delta = EBITMarginDelta()
    r = delta.cal_factors(20090101, 20200609, N_JOBS)
    print(r)
    print('time token:', datetime.datetime.now()-start_dt)