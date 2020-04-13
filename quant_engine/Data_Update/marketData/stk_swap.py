# 回测所用日线数据维护
# 数据包含：分红、送股、转增、配股、吸收合并

from influxdb_data import influxdbData
from rdf_data import rdf_data
import pandas as pd
import numpy as np
import datetime
import dateutil.parser as dtparser
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS


class stk_swap:
    def __init__(self):
        self.rdf = rdf_data()
        self.influx = influxdbData()
        self.db = 'DailyMarket_Gus'
        self.measure = 'swap'

    def process_data(self, start, end, n_jobs):
        # 获取转股信息
        query = "select TRANSFERER_WINDCODE, TARGETCOMP_WINDCODE, CONVERSIONRATIO, ANNCEDATE, " \
                "LASTTRADEDATE, LISTDATE " \
                "from wind_filesync.AShareStockSwap " \
                "where (TRANSFERER_WINDCODE like '0%' or TRANSFERER_WINDCODE like '3%' or " \
                "TRANSFERER_WINDCODE like '6%') " \
                "and PROGRESS = 3 and anncedate >= 20090101"
        self.rdf.curs.execute(query)
        swap = pd.DataFrame(self.rdf.curs.fetchall(),
                            columns=['code', 'swap_code', 'swap_ratio', 'ann_dt', 'last_trade_dt', 'swap_dt'])
        swap = swap.loc[(swap['ann_dt'] <= str(end)) & (swap['swap_dt'] >= str(start)) &
                        (swap['last_trade_dt'] >= '20100101'), :]
        # 填充日期
        calendar = self.rdf.get_trading_calendar()
        trade_day_values = []
        swap_day_values = []
        code_values = []
        swap_code_values = []
        swap_ratio_values = []
        for idx, row in swap.iterrows():
            trade_days = list(calendar[(calendar >= row['ann_dt']) & (calendar <= row['swap_dt'])])
            trade_day_values.extend(trade_days)
            swap_day_values.extend([row['swap_dt']] * len(trade_days))
            code_values.extend([row['code']] * len(trade_days))
            swap_code_values.extend([row['swap_code']] * len(trade_days))
            swap_ratio_values.extend([row['swap_ratio']] * len(trade_days))
        swap = pd.DataFrame({'date': trade_day_values, 'swap_date': swap_day_values, 'code': code_values,
                             'swap_code': swap_code_values, 'swap_ratio': swap_ratio_values})
        swap.set_index('date', inplace=True)
        swap = swap.where(pd.notnull(swap), None)
        codes = swap['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (swap, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('Daily swap finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    print(datetime.datetime.now())
    btd = stk_swap()
    start = 20100101
    end = 20200410
    btd.process_data(start, end, n_jobs=N_JOBS)
    print("start: %i ~ end: %i is finish!" % (start, end))
    print(datetime.datetime.now())
