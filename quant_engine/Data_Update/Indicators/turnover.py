from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
from joblib import Parallel, delayed, parallel_backend

class turnover(FactorBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def job_factors(codes, df):
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            # 成交量单位是手，股本单位是万股，除完正好是百分比
            code_df['turnover'] = code_df['volume'] / code_df['tot_shares']
            code_df['float_turnover'] = code_df['volume'] / code_df['float_shares']
            code_df['free_turnover'] = code_df['volume'] / code_df['free_shares']
            code_df.set_index('date', inplace=True)
            code_df = code_df.loc[pd.notnull(code_df['turnover']) | pd.notnull(code_df['float_turnover']) |
                                  pd.notnull(code_df['free_turnover']),
                                  ['code', 'turnover', 'float_turnover', 'free_turnover']]
            code_df = code_df.where(pd.notnull(code_df), None)
            print('code: %s' % code)
            influx = influxdbData()
            influx.saveData(code_df, 'DailyData_Gus', 'indicators')

    def run(self, start, end):
        mkt = self.influx.getDataMultiprocess('DailyData_Gus', 'marketData', start, end, ['code', 'volume'])
        shares = self.influx.getDataMultiprocess('DailyData_Gus', 'indicators', start, end, None)
        print('raw_data got')
        mkt.index.names = ['date']
        shares.index.names = ['date']
        mkt.reset_index('date', inplace=True)
        shares.reset_index('date', inplace=True)
        merge = pd.merge(mkt, shares, on=['date', 'code'])
        codes = merge['code'].unique()
        split_codes = np.array_split(codes, 10)
        with parallel_backend('multiprocessing', n_jobs=4):
            Parallel()(delayed(turnover.job_factors)
                       (codes, merge) for codes in split_codes)


if __name__ == '__main__':
    i = turnover()
    i.run(20130101, 20130901)
