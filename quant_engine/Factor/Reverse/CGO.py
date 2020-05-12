#  翻转投机因子 未实现收益 CGO 的计算

from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
import dateutil.parser as dtparser
from joblib import Parallel, delayed, parallel_backend
from dateutil.relativedelta import relativedelta
import math
from global_constant import N_JOBS


class CGO(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'CGO'

    @staticmethod
    def JOB_factors(mkt_data, codes, start, period, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_mkt = mkt_data.loc[mkt_data['code'] == code, :].copy()
            # 昨日买入，今日没有unturn数据
            code_mkt['to_1'] = code_mkt['float_turnover'].shift(1)
            code_mkt['price_1'] = code_mkt['fq_vwap'].shift(1) * code_mkt['to_1']
            to_cols = ['to_1']
            hp_cols = ['price_1']
            for p in range(2, period+1):
                turnover = code_mkt['float_turnover'].shift(p)
                prod_unturn = \
                    code_mkt['unturn'].rolling(p-1, min_periods=p-1).apply(lambda x: np.product(x)).shift(1)
                code_mkt['to_{0}'.format(p)] = turnover * prod_unturn
                code_mkt['price_{0}'.format(p)] = code_mkt['fq_vwap'].shift(p) * code_mkt['to_{0}'.format(p)]
                to_cols.append('to_{0}'.format(p))
                hp_cols.append('price_{0}'.format(p))
            code_mkt = code_mkt.dropna()
            code_mkt['multi'] = 1 / code_mkt[to_cols].sum(axis=1)
            code_mkt['price'] = code_mkt[hp_cols].sum(axis=1) * code_mkt['multi']
            code_mkt['CGO_{0}'.format(period)] = \
                (code_mkt['fq_vwap'] - code_mkt['price']) / code_mkt['price']
            code_mkt = code_mkt.loc[str(start):, ['code', 'CGO_{0}'.format(period)]]
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
        self.period = 60
        data_start = (dtparser.parse(str(start)) - relativedelta(years=1)).strftime('%Y%m%d')
        mkt_data = self.influx.getDataMultiprocess('DailyMarket_Gus', 'market', data_start, end,
                                                   ['code', 'status', 'vwap', 'adj_factor'])
        mkt_data = mkt_data.loc[mkt_data['status'] != '停牌', ['code', 'vwap', 'adj_factor']]
        mkt_data['fq_vwap'] = mkt_data['vwap'] * mkt_data['adj_factor']
        mkt_data.index.names = ['date']
        turnover = self.influx.getDataMultiprocess('DailyMarket_Gus', 'shares_turnover', data_start, end,
                                                   ['code', 'float_turnover'])
        turnover['float_turnover'] = turnover['float_turnover'] / 100
        turnover.index.names = ['date']
        mkt_data = pd.merge(mkt_data.reset_index(), turnover.reset_index(), on=['date', 'code'])
        mkt_data.set_index('date', inplace=True)
        mkt_data['unturn'] = 1 - mkt_data['float_turnover']
        mkt_data = mkt_data.loc[:, ['code', 'fq_vwap', 'float_turnover', 'unturn']]
        codes = mkt_data['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(CGO.JOB_factors)
                             (mkt_data, codes, start, self.period, self.db, self.measure)
                             for codes in split_codes)
        print('CGO finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list

if __name__ == '__main__':
    i = CGO()
    r = i.cal_factors(20200101, 20200508, N_JOBS)
    print(r)
