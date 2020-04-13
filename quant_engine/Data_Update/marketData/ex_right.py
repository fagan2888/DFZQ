# 回测所用日线数据维护
# 数据包含：分红、送股、转增、配股

from influxdb_data import influxdbData
from rdf_data import rdf_data
import pandas as pd
import numpy as np
import datetime
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS


class ex_right:
    def __init__(self):
        self.rdf = rdf_data()
        self.influx = influxdbData()
        self.db = 'DailyMarket_Gus'
        self.measure = 'exright'

    def process_data(self, start, end, n_jobs):
        # 获取除权除息信息
        query = "select EX_DATE, S_INFO_WINDCODE, CASH_DIVIDEND_RATIO, BONUS_SHARE_RATIO, RIGHTSISSUE_RATIO, " \
                "RIGHTSISSUE_PRICE, CONVERSED_RATIO, CONSOLIDATE_SPLIT_RATIO, SEO_PRICE, SEO_RATIO " \
                "from wind_filesync.AShareEXRightDividendRecord " \
                "where (S_INFO_WINDCODE like '0%' or S_INFO_WINDCODE like '3%' or S_INFO_WINDCODE like '6%') " \
                "and EX_DATE >= {0} and EX_DATE <= {1}".format(start, end)
        self.rdf.curs.execute(query)
        ex_right = pd.DataFrame(self.rdf.curs.fetchall(),
                                columns=['date', 'code', 'cash_dvd_ratio', 'bonus_share_ratio',
                                         'rightissue_ratio', 'rightissue_price', 'conversed_ratio',
                                         'split_ratio', 'seo_price', 'seo_ratio'])
        ex_right['date'] = pd.to_datetime(ex_right['date'])
        ex_right.set_index('date', inplace=True)
        ex_right = ex_right.where(pd.notnull(ex_right), None)
        codes = ex_right['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (ex_right, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('Daily ex_right finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    print(datetime.datetime.now())
    btd = ex_right()
    start = 20100101
    end = 20200410
    btd.process_data(start, end, n_jobs=N_JOBS)
    print("start: %i ~ end: %i is finish!" % (start, end))
    print(datetime.datetime.now())
