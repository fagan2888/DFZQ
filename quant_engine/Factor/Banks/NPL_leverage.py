from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
import datetime
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS

class NPL_leverage(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'NPL_leverage'

    @staticmethod
    def JOB_factors(df, codes, save_db, save_msr):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_df = code_df.sort_values(['date', 'report_period'])
            code_df['CA_ratio'] = code_df['CA_ratio'].fillna(method='ffill')
            code_df['CA_ratio_last1Q'] = code_df['CA_ratio_last1Q'].fillna(method='ffill')
            code_df = code_df.drop_duplicates(['date'], 'last')
            code_df['NPL_leverage'] = code_df['NPL'] / code_df['CA_ratio']
            code_df['NPL_leverage_last1Q'] = code_df['NPL_last1Q'] / code_df['CA_ratio_last1Q']
            code_df = code_df.loc[:, ['date', 'code', 'report_period', 'NPL_leverage', 'NPL_leverage_last1Q']]
            code_df.set_index('date', inplace=True)
            code_df = code_df.where(pd.notnull(code_df), None)
            print('code: %s' % code)
            r = influx.saveData(code_df, save_db, save_msr)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('NPL_leverage Error: %s' % r)
        return save_res

    def cal_factors(self, start, end, n_jobs):
        NPL = self.influx.getDataMultiprocess('FinancialReport_Gus', 'NPL', start, end,
                                              ['code', 'NPL', 'NPL_last1Q', 'report_period'])
        CA_ratio = self.influx.getDataMultiprocess('FinancialReport_Gus', 'CA_ratio', start, end,
                                                   ['code', 'CA_ratio', 'CA_ratio_last1Q', 'report_period'])
        NPL.index.names = ['date']
        NPL.reset_index(inplace=True)
        CA_ratio.index.names = ['date']
        CA_ratio.reset_index(inplace=True)
        merge_df = pd.merge(NPL, CA_ratio, how='outer', on=['date', 'code', 'report_period'])
        codes = merge_df['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(NPL_leverage.JOB_factors)
                             (merge_df, codes, self.db, self.measure) for codes in split_codes)
        print('NPL_leverage finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list

if __name__ == '__main__':
    print(datetime.datetime.now())
    NPL_leverage = NPL_leverage()
    r = NPL_leverage.cal_factors(20100101, 20200522, N_JOBS)
    print('task finish')
    print(r)
    print(datetime.datetime.now())