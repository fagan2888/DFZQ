# 回测所用日线数据维护
# 数据包含：高开低收

from influxdb_data import influxdbData
from rdf_data import rdf_data
import pandas as pd
import numpy as np
import datetime
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS


class mkt_ohlc:
    def __init__(self):
        self.rdf = rdf_data()
        self.influx = influxdbData()
        self.db = 'DailyMarket_Gus'
        self.measure = 'market'

    def process_data(self, start, end, n_jobs):
        # 获取股票高开低收量价状态
        query = "select TRADE_DT,S_INFO_WINDCODE,S_DQ_ADJFACTOR,S_DQ_PRECLOSE,S_DQ_OPEN,S_DQ_HIGH,S_DQ_LOW," \
                "S_DQ_CLOSE,S_DQ_VOLUME,S_DQ_AMOUNT,S_DQ_TRADESTATUS,S_DQ_AVGPRICE " \
                "from wind_filesync.AShareEODPrices " \
                "where (S_INFO_WINDCODE like '0%' or S_INFO_WINDCODE like '3%' or S_INFO_WINDCODE like '6%') " \
                "and TRADE_DT >= {0} and TRADE_DT <= {1}".format(str(start), str(end))
        self.rdf.curs.execute(query)
        stk_ohlc = pd.DataFrame(self.rdf.curs.fetchall(),
                                columns=['date', 'code', 'adj_factor', 'preclose', 'open', 'high', 'low',
                                         'close', 'volume', 'amount', 'status', 'vwap'])
        stk_ohlc['date'] = pd.to_datetime(stk_ohlc['date'], format="%Y%m%d")
        # 获取股票高开低收量价状态
        query = "select TRADE_DT,S_INFO_WINDCODE,S_DQ_PRECLOSE,S_DQ_OPEN,S_DQ_HIGH,S_DQ_LOW,S_DQ_CLOSE," \
                "S_DQ_VOLUME,S_DQ_AMOUNT " \
                "from wind_filesync.AIndexEODPrices " \
                "where S_INFO_WINDCODE in ('000016.SH','000300.SH','000905.SH','000906.SH','000852.SH') " \
                "and TRADE_DT >= {0} and TRADE_DT <= {1}".format(str(start), str(end))
        self.rdf.curs.execute(query)
        idx_ohlc = pd.DataFrame(self.rdf.curs.fetchall(),
                                columns=['date', 'code', 'preclose', 'open', 'high', 'low', 'close',
                                         'volume', 'amount'])
        idx_ohlc['adj_factor'] = 1
        idx_ohlc['status'] = '交易'
        idx_ohlc['date'] = pd.to_datetime(idx_ohlc['date'], format="%Y%m%d")
        # 合并
        ohlc = pd.concat([stk_ohlc, idx_ohlc])
        ohlc.set_index('date', inplace=True)
        # 存数据前整理
        ohlc = ohlc.where(pd.notnull(ohlc), None)
        codes = ohlc['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (ohlc, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('Daily OHLC finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list

if __name__ == '__main__':
    print(datetime.datetime.now())
    btd = mkt_ohlc()
    start = 20100101
    end = 20200410
    res = btd.process_data(start=start, end=end, n_jobs=N_JOBS)
    print(res)
    print("start: %i ~ end: %i is finish!" % (start, end))
    print(datetime.datetime.now())
