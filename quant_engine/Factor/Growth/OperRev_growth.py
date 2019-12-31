#  营业收入同比 的计算

from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
from joblib import Parallel, delayed, parallel_backend


class OperRev_growth(FactorBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def job_tot_OR(codes, df):
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_df = code_df.sort_values('date')
            code_df['tot_oper_rev_Q_growthY'] = \
                code_df.apply(lambda row: OperRev_growth.cal_growth(row['tot_oper_rev_Q_lastY'], row['tot_oper_rev_Q']),
                              axis=1)
            code_df = code_df.loc[:, ['date', 'code', 'tot_oper_rev_Q_growthY']]
            code_df.set_index('date', inplace=True)
            code_df = code_df.where(pd.notnull(code_df), None)
            code_df.dropna(subset=['tot_oper_rev_Q_growthY'], inplace=True)
            if code_df.empty:
                continue
            else:
                print('code: %s' % code)
                influx = influxdbData()
                influx.saveData(code_df, 'DailyFactor_Gus', 'Growth')

    @staticmethod
    def job_OR(codes, df):
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_df = code_df.sort_values('date')
            code_df['oper_rev_Q_growthY'] = \
                code_df.apply(lambda row: OperRev_growth.cal_growth(row['oper_rev_Q_lastY'], row['oper_rev_Q']),
                              axis=1)
            code_df = code_df.loc[:, ['date', 'code', 'oper_rev_Q_growthY']]
            code_df.set_index('date', inplace=True)
            code_df = code_df.where(pd.notnull(code_df), None)
            code_df.dropna(subset=['oper_rev_Q_growthY'], inplace=True)
            if code_df.empty:
                continue
            else:
                print('code: %s' % code)
                influx = influxdbData()
                influx.saveData(code_df, 'DailyFactor_Gus', 'Growth')

    def cal_factors(self, start, end):
        tot_oper_rev = self.influx.getDataMultiprocess('Financial_Report_Gus', 'tot_oper_rev', start, end,
                                                       ['code', 'tot_oper_rev_Q', 'tot_oper_rev_Q_lastY'])
        print('TOT OPER REV data got')
        tot_oper_rev.index.names = ['date']
        tot_oper_rev.reset_index(inplace=True)
        codes = tot_oper_rev['code'].unique()
        split_codes = np.array_split(codes, 10)
        with parallel_backend('multiprocessing', n_jobs=4):
            Parallel()(delayed(OperRev_growth.job_tot_OR)
                       (codes, tot_oper_rev) for codes in split_codes)
        # ----------------------------------------------------------------------------------------------
        oper_rev = self.influx.getDataMultiprocess('Financial_Report_Gus', 'oper_rev', start, end,
                                                   ['code', 'oper_rev_Q', 'oper_rev_Q_lastY'])
        print('OPER REV data got')
        oper_rev.index.names = ['date']
        oper_rev.reset_index(inplace=True)
        codes = oper_rev['code'].unique()
        split_codes = np.array_split(codes, 10)
        with parallel_backend('multiprocessing', n_jobs=4):
            Parallel()(delayed(OperRev_growth.job_OR)
                       (codes, oper_rev) for codes in split_codes)


if __name__ == '__main__':
    i = OperRev_growth()
    i.cal_factors(20100101, 20190901)
