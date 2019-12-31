#  assets_turn 的计算

from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
from joblib import Parallel, delayed, parallel_backend


class MA_p1(FactorBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def cal_assets_turn(oper_rev_Q, tot_assets, tot_assets_last1Q):
        if tot_assets < 0:
            return np.nan
        else:
            try:
                if pd.notnull(tot_assets) & pd.notnull(tot_assets_last1Q):
                    return oper_rev_Q * 2 / (tot_assets + tot_assets_last1Q)
                elif pd.notnull(tot_assets) & pd.isnull(tot_assets_last1Q):
                    return oper_rev_Q / tot_assets
                elif pd.isnull(tot_assets) & pd.notnull(tot_assets_last1Q):
                    return oper_rev_Q / tot_assets_last1Q
                else:
                    return np.nan
            except ZeroDivisionError:
                return np.nan

    @staticmethod
    def job_factors(codes, df):
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_df = code_df.sort_values('date')
            code_df['assets_turn_Q'] = \
                code_df.apply(lambda row: MA_p1.cal_assets_turn(row['tot_oper_rev_Q'], row['tot_assets'],
                                                                row['tot_assets_last1Q']), axis=1)
            code_df = code_df.loc[:, ['date', 'code', 'assets_turn_Q']]
            code_df.set_index('date', inplace=True)
            code_df = code_df.where(pd.notnull(code_df), None)
            code_df.dropna(subset=['assets_turn_Q'],inplace=True)
            if code_df.empty:
                continue
            else:
                print('code: %s' % code)
                influx = influxdbData()
                influx.saveData(code_df, 'DailyFactor_Gus', 'FinancialQuality')

    def cal_factors(self, start, end):
        oper_rev = self.influx.getDataMultiprocess('Financial_Report_Gus', 'tot_oper_rev', start, end,
                                                   ['code', 'tot_oper_rev_Q'])
        total_assets = self.influx.getDataMultiprocess('Financial_Report_Gus', 'tot_assets', start, end,
                                                       ['code', 'tot_assets', 'tot_assets_last1Q'])
        print('raw data got')
        oper_rev.index.names = ['date']
        oper_rev.reset_index(inplace=True)
        total_assets.index.names = ['date']
        total_assets.reset_index(inplace=True)
        merge = pd.merge(oper_rev, total_assets, on=['date', 'code'])
        codes = merge['code'].unique()
        split_codes = np.array_split(codes, 10)
        with parallel_backend('multiprocessing', n_jobs=4):
            Parallel()(delayed(MA_p1.job_factors)
                       (codes, merge) for codes in split_codes)


if __name__ == '__main__':
    i = MA_p1()
    i.cal_factors(20100101, 20190901)
