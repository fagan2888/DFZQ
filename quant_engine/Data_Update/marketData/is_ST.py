# 回测所用日线数据维护
# 数据包含：是否st

from influxdb_data import influxdbData
from rdf_data import rdf_data
import pandas as pd
import numpy as np
import datetime
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS


class isst:
    def __init__(self):
        self.rdf = rdf_data()
        self.influx = influxdbData()
        self.db = 'DailyMarket_Gus'
        self.measure = 'isST'

    def process_data(self, start, end, n_jobs):
        sql_sentence = "select s_info_windcode,ENTRY_DT,REMOVE_DT " \
                       "from wind_filesync.AShareST"
        self.rdf.curs.execute(sql_sentence)
        st = pd.DataFrame(self.rdf.curs.fetchall(), columns=['code', 'entry_date', 'exit_date'])
        st = st.loc[(st['entry_date'] <= str(end)) &
                    ((st['exit_date'] >= str(start)) | pd.isnull(st['exit_date'])), :]
        calendar = self.rdf.get_trading_calendar()
        trade_day_values = []
        st_values = []
        code_values = []
        for idx, row in st.iterrows():
            s_date = max(row['entry_date'], str(start))
            if pd.isnull(row['exit_date']):
                e_date = str(end)
            else:
                e_date = min(row['exit_date'], str(end))
            trade_days = list(calendar[(calendar >= s_date) & (calendar <= e_date)])
            if not trade_days:
                continue
            trade_day_values.extend(trade_days)
            st_values.extend([True] * len(trade_days))
            code_values.extend([row['code']] * len(trade_days))
        df = pd.DataFrame({'date': trade_day_values, 'isST': st_values, 'code': code_values})
        df.set_index('date', inplace=True)
        # 存数据前整理
        df = df.where(pd.notnull(df), None)
        codes = df['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (df, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('Daily isST finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    st = isst()
    st.process_data(20100101, 20200410, N_JOBS)