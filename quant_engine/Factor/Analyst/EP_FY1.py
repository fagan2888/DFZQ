import pandas as pd
import numpy as np
from factor_base import FactorBase
import datetime
from global_constant import N_JOBS
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData


class EP_FY1(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'Analyst'

    def cal_factors(self, start, end, n_jobs):
        # 获取一致预期净利润
        net_profit_FY1 = self.influx.getDataMultiprocess('DailyFactors_Gus', 'Analyst', start, end,
                                                         ['code', 'net_profit_FY1'])
        net_profit_FY1.index.names = ['date']
        net_profit_FY1.reset_index(inplace=True)
        market_cap = self.influx.getDataMultiprocess('DailyFactors_Gus', 'Size', start, end, ['code', 'market_cap'])
        market_cap.index.names = ['date']
        market_cap.reset_index(inplace=True)
        merge_df = pd.merge(market_cap, net_profit_FY1, how='inner', on=['date', 'code'])
        # 市值单位为 万元
        merge_df['EP_FY1'] = merge_df['net_profit_FY1'] / merge_df['market_cap'] / 10000
        merge_df.set_index('date', inplace=True)
        merge_df = merge_df.loc[:, ['code', 'EP_FY1']]
        merge_df = merge_df.replace(np.inf, np.nan)
        merge_df = merge_df.replace(-np.inf, np.nan)
        merge_df = merge_df.dropna()
        # 储存数据
        codes = merge_df['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (merge_df, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('consensus net profit finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    time_start = datetime.datetime.now()
    af = EP_FY1()
    f = af.cal_factors(20100101, 20200402, N_JOBS)
    print(f)
    time_end = datetime.datetime.now()
    print('Time token:', time_end - time_start)
