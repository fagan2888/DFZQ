#  翻转投机因子 Period_Ret 的计算

from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
import dateutil.parser as dtparser
from joblib import Parallel, delayed, parallel_backend
from dateutil.relativedelta import relativedelta
import math
from global_constant import N_JOBS


class PeriodRet(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'PeriodRet'

    @staticmethod
    def JOB_factors(mkt_data, codes, start, period, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_mkt = mkt_data.loc[mkt_data['code'] == code, ['code', 'fq_close', 'fq_preclose']].copy()
            code_mkt['period_preclose'] = code_mkt['fq_preclose'].shift(period)
            code_mkt['ret_{0}'.format(period)] = code_mkt['fq_close'] / code_mkt['period_preclose'] - 1
            code_mkt = code_mkt.loc[str(start):, ['code', 'ret_{0}'.format(period)]]
            code_mkt = code_mkt.dropna()
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
        self.period = 20
        data_start = (dtparser.parse(str(start)) - relativedelta(years=1)).strftime('%Y%m%d')
        mkt_data = self.influx.getDataMultiprocess('DailyMarket_Gus', 'market', data_start, end,
                                                   ['code', 'status', 'adj_factor', 'close', 'preclose'])
        mkt_data = mkt_data.loc[mkt_data['status'] != '停牌', ['code', 'adj_factor', 'close', 'preclose']]
        mkt_data.index.names = ['date']
        mkt_data['fq_close'] = mkt_data['adj_factor'] * mkt_data['close']
        mkt_data['fq_preclose'] = mkt_data['adj_factor'] * mkt_data['preclose']
        codes = mkt_data['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(PeriodRet.JOB_factors)
                             (mkt_data, codes, start, self.period, self.db, self.measure)
                             for codes in split_codes)
        print('PeriodReturn finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list

if __name__ == '__main__':
    i = PeriodRet()
    r = i.cal_factors(20100101, 20200508, N_JOBS)
    print(r)
