#  翻转投机因子 MaxRet 的计算

from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
import dateutil.parser as dtparser
from joblib import Parallel, delayed, parallel_backend
from dateutil.relativedelta import relativedelta
import math
from global_constant import N_JOBS


class MaxRet(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'MaxRet'

    @staticmethod
    def JOB_factors(mkt_data, codes, start, period, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_mkt = mkt_data.loc[mkt_data['code'] == code, ['code', 'return']].copy()
            code_mkt['max_return_{0}'.format(period)] = \
                code_mkt['return'].rolling(period, min_periods=round(period * 0.6)).apply(
                    lambda x: np.sort(x)[-3:].mean())
            code_mkt = code_mkt.dropna(subset=['max_return_{0}'.format(period)])
            code_mkt = code_mkt.loc[str(start):, ['code', 'max_return_{0}'.format(period)]]
            if code_mkt.empty:
                continue
            print('code: %s' % code)
            r = influx.saveData(code_mkt, db, measure)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('%s Error: %s' % (measure, r))
        return save_res


    def cal_factors(self, start, end, n_jobs):
        self.period = 60
        data_start = (dtparser.parse(str(start)) - relativedelta(years=1)).strftime('%Y%m%d')
        mkt_data = self.influx.getDataMultiprocess('DailyMarket_Gus', 'market', data_start, end,
                                                   ['code', 'status', 'close', 'preclose'])
        mkt_data = mkt_data.loc[mkt_data['status'] != '停牌', ['code', 'close', 'preclose']]
        mkt_data.index.names = ['date']
        mkt_data['return'] = mkt_data['close'] / mkt_data['preclose'] - 1
        mkt_data = mkt_data.loc[:, ['code', 'return']]
        codes = mkt_data['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(MaxRet.JOB_factors)
                             (mkt_data, codes, start, self.period, self.db, self.measure)
                             for codes in split_codes)
        print('MaxReturn finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list

if __name__ == '__main__':
    i = MaxRet()
    r = i.cal_factors(20100101, 20200506, N_JOBS)
    print(r)
