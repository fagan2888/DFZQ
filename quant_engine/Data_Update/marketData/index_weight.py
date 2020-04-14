# 回测所用日线数据维护
# 数据包含：指数权重

from influxdb_data import influxdbData
from rdf_data import rdf_data
from Index_comp_sql import IndexCompSQL
import pandas as pd
import numpy as np
import datetime
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS


class index_weight:
    def __init__(self):
        self.idx_comp_sql = IndexCompSQL()
        self.influx = influxdbData()
        self.db = 'DailyMarket_Gus'
        self.measure = 'index_weight'

    def process_data(self, start, end, n_jobs):
        # 获取50权重
        weight_50 = self.idx_comp_sql.get_IndexComp(50, start, end)
        weight_50['index_code'] = '000016.SH'
        # 获取300权重
        weight_300 = self.idx_comp_sql.get_IndexComp(300, start, end)
        weight_300['index_code'] = '000300.SH'
        # 获取300权重
        weight_500 = self.idx_comp_sql.get_IndexComp(500, start, end)
        weight_500['index_code'] = '000905.SH'

        weight = pd.concat([weight_50, weight_300, weight_500])
        weight['weight'] = weight['weight'].astype('float')
        codes = weight['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (weight, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('IndexWeight finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    print(datetime.datetime.now())
    btd = index_weight()
    start = 20100101
    end = 20200410
    btd.process_data(start, end, n_jobs=N_JOBS)
    print("start: %i ~ end: %i is finish!" % (start, end))
    print(datetime.datetime.now())
