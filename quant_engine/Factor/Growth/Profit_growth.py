#  利润同比 的计算

from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
from joblib import Parallel, delayed, parallel_backend


class Profit_growth(FactorBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def job_factors(codes, df, field):
        cur_field = field + '_Q'
        lastY_field = field + '_lastY'
        result_field = field + '_Q_growthY'
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_df = code_df.sort_values('date')
            code_df[result_field] = \
                code_df.apply(lambda row: Profit_growth.cal_growth(row[lastY_field], row[cur_field]), axis=1)
            code_df = code_df.loc[:, ['date', 'code', result_field]]
            code_df.set_index('date', inplace=True)
            code_df = code_df.where(pd.notnull(code_df), None)
            code_df.dropna(subset=[result_field], inplace=True)
            if code_df.empty:
                continue
            else:
                print('code: %s' % code)
                influx = influxdbData()
                influx.saveData(code_df, 'DailyFactor_Gus', 'Growth')

    def cal_factors(self, start, end):
        net_profit = self.influx.getDataMultiprocess('Financial_Report_Gus', 'net_profit', start, end,
                                                     ['code', 'net_profit_Q', 'net_profit_lastY'])
        print('net profit data got')
        net_profit.index.names = ['date']
        net_profit.reset_index(inplace=True)
        codes = net_profit['code'].unique()
        split_codes = np.array_split(codes, 10)
        with parallel_backend('multiprocessing', n_jobs=4):
            Parallel()(delayed(Profit_growth.job_factors)
                       (codes, net_profit, 'net_profit') for codes in split_codes)
        # ----------------------------------------------------------------------------------------------
        net_profit_ddt = self.influx.getDataMultiprocess('Financial_Report_Gus', 'net_profit_ddt', start, end,
                                                         ['code', 'net_profit_ddt_Q', 'net_profit_ddt_lastY'])
        print('net profit ddt data got')
        net_profit_ddt.index.names = ['date']
        net_profit_ddt.reset_index(inplace=True)
        codes = net_profit_ddt['code'].unique()
        split_codes = np.array_split(codes, 10)
        with parallel_backend('multiprocessing', n_jobs=4):
            Parallel()(delayed(Profit_growth.job_factors)
                       (codes, net_profit_ddt, 'net_profit_ddt') for codes in split_codes)


if __name__ == '__main__':
    i = Profit_growth()
    i.cal_factors(20100101, 20190901)
