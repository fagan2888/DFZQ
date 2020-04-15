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
        self.rdf = rdf_data()
        self.influx = influxdbData()
        self.db = 'DailyMarket_Gus'
        self.measure = 'index_weight'

    def process_data(self, start, end, n_jobs):
        calendar = self.rdf.get_trading_calendar()
        calendar = calendar[(calendar >= str(start)) & (calendar <= str(end))]
        # 获取50权重
        weight_50 = self.idx_comp_sql.get_IndexComp(50, start, end)
        weight_50['index_code'] = '000016.SH'
        miss_dates = set(calendar) - set(weight_50.index.unique())
        if miss_dates:
            miss_dates = pd.DatetimeIndex(miss_dates).strftime('%Y%m%d')
            query = "select TRADE_DT,S_INFO_WINDCODE,S_CON_WINDCODE,weight " \
                    "from wind_filesync.AIndexSSE50Weight " \
                    "where TRADE_DT in " + str(tuple(miss_dates))
            self.rdf.curs.execute(query)
            miss_df = pd.DataFrame(self.rdf.curs.fetchall(), columns=['date', 'index_code', 'code', 'weight'])
            miss_df['date'] = pd.to_datetime(miss_df['date'])
            miss_df.set_index('date', inplace=True)
            weight_50 = pd.concat([weight_50, miss_df])
        # 获取300权重
        weight_300 = self.idx_comp_sql.get_IndexComp(300, start, end)
        weight_300['index_code'] = '000300.SH'
        miss_dates = set(calendar) - set(weight_300.index.unique())
        if miss_dates:
            dates_before_miss = {}
            for d in miss_dates:
                dates_before_miss[calendar[calendar < d].iloc[-1].strftime('%Y%m%d')] = d.strftime('%Y%m%d')
            query = "select TRADE_DT,S_INFO_WINDCODE,S_CON_WINDCODE,i_weight " \
                    "from wind_filesync.AIndexHS300Weight " \
                    "where TRADE_DT in " + str(tuple(dates_before_miss.keys()))
            self.rdf.curs.execute(query)
            miss_df = pd.DataFrame(self.rdf.curs.fetchall(), columns=['last_date', 'index_code', 'code', 'weight'])
            miss_df['date'] = miss_df['last_date'].map(dates_before_miss)
            miss_df['date'] = pd.to_datetime(miss_df['date'])
            miss_df.drop('last_date', axis=1, inplace=True)
            miss_df.set_index('date', inplace=True)
            weight_300 = pd.concat([weight_300, miss_df])
        # 获取500权重
        weight_500 = self.idx_comp_sql.get_IndexComp(500, start, end)
        weight_500['index_code'] = '000905.SH'
        miss_dates = set(calendar) - set(weight_500.index.unique())
        if miss_dates:
            miss_dates = pd.DatetimeIndex(miss_dates).strftime('%Y%m%d')
            query = "select TRADE_DT,S_INFO_WINDCODE,S_CON_WINDCODE,weight " \
                    "from wind_filesync.AIndexCSI500Weight " \
                    "where TRADE_DT in " + str(tuple(miss_dates))
            self.rdf.curs.execute(query)
            miss_df = pd.DataFrame(self.rdf.curs.fetchall(), columns=['date', 'index_code', 'code', 'weight'])
            miss_df['date'] = pd.to_datetime(miss_df['date'])
            miss_df.set_index('date', inplace=True)
            weight_500 = pd.concat([weight_500, miss_df])
        ########################################################################
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
    end = 20200413
    btd.process_data(start, end, n_jobs=N_JOBS)
    print("start: %i ~ end: %i is finish!" % (start, end))
    print(datetime.datetime.now())
