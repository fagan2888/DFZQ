#  非流动性因子 ln_turnover_60 的计算

from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
import dateutil.parser as dtparser
from joblib import Parallel, delayed, parallel_backend
from dateutil.relativedelta import relativedelta
import math
from global_constant import N_JOBS


class LnTurnover(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'ln_ma_turnover'

    @staticmethod
    def JOB_factors(turnover, codes, start, period, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_to = turnover.loc[turnover['code'] == code, :].copy()
            code_to['turnover_{0}'.format(period)] = \
                code_to['float_turnover'].rolling('{0}d'.format(period), min_periods=round(period/2)).mean()
            code_to['ln_turnover_{0}'.format(period)] = np.log(code_to['turnover_{0}'.format(period)].values)
            code_to = code_to.loc[str(start):, ['code', 'ln_turnover_{0}'.format(period)]].dropna()
            if code_to.empty:
                continue
            print('code: %s' % code)
            r = influx.saveData(code_to, db, measure)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('%s Error: %s' % (measure, r))
        return save_res


    def cal_factors(self, start, end, n_jobs):
        self.period = 60
        data_start = (dtparser.parse(str(start)) - relativedelta(years=1)).strftime('%Y%m%d')
        turnover = self.influx.getDataMultiprocess('DailyMarket_Gus', 'shares_turnover', data_start, end,
                                                   ['code', 'float_turnover'])
        turnover.index.names = ['date']
        turnover = turnover.loc[turnover['float_turnover'] > 0, :]
        codes = turnover['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(LnTurnover.JOB_factors)
                             (turnover, codes, start, self.period, self.db, self.measure)
                             for codes in split_codes)
        print('ln_ma_turnover finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list

if __name__ == '__main__':
    i = LnTurnover()
    r = i.cal_factors(20100101, 20200506, N_JOBS)
    print(r)
