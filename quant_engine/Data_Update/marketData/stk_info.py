# 回测所用日线数据维护
# 数据包含：上市日期，退市日期，股票简称

from influxdb_data import influxdbData
from rdf_data import rdf_data
import pandas as pd
import numpy as np
import datetime
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS


class StkInfo:
    def __init__(self):
        self.rdf = rdf_data()
        self.influx = influxdbData()
        self.db = 'DailyMarket_Gus'
        self.measure = 'stk_info'

    def process_data(self, n_jobs):
        # 获取除权除息信息
        query = "select S_INFO_WINDCODE, S_INFO_NAME, S_INFO_LISTDATE, S_INFO_DELISTDATE " \
                "from wind_filesync.AShareDescription " \
                "where (S_INFO_WINDCODE like '0%' or S_INFO_WINDCODE like '3%' or S_INFO_WINDCODE like '6%') "
        self.rdf.curs.execute(query)
        stk_info = pd.DataFrame(self.rdf.curs.fetchall(), columns=['code', 'name', 'list_date', 'delist_date'])
        stk_info['date'] = datetime.datetime.now().strftime('%Y%m%d')
        stk_info['date'] = pd.to_datetime(stk_info['date'])
        stk_info.set_index('date', inplace=True)
        codes = stk_info['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (stk_info, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('stk info finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    print(datetime.datetime.now())
    btd = StkInfo()
    btd.process_data(n_jobs=N_JOBS)
    print(datetime.datetime.now())
