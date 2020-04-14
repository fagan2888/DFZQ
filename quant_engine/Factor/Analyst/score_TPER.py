from gogoal_data import GoGoal_data
import pandas as pd
import numpy as np
from factor_base import FactorBase
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
import datetime
from global_constant import N_JOBS
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData


class score_TPER(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'Analyst'
        self.gogoal = GoGoal_data()

    def cal_factors(self, start, end, n_jobs):
        fail_list = []
        # 获取一致预期数据
        query = "SELECT [CON_DATE], [STOCK_CODE], [TARGET_PRICE], [TARGET_PRICE_TYPE], [SCORE], [SCORE_TYPE] " \
                "FROM [{0}].[dbo].[CON_FORECAST_SCHEDULE] " \
                "WHERE [CON_DATE] >= '{1}' and [CON_DATE] <= '{2}' " \
                "order by [CON_DATE] " \
            .format(self.gogoal.database,
                    (dtparser.parse(str(start)) - relativedelta(months=1)).strftime('%Y%m%d'), end)
        self.gogoal.cur.execute(query)
        consensus = pd.DataFrame(self.gogoal.cur.fetchall(),
                                 columns=['date', 'code', 'target_price', 'target_price_type', 'score', 'score_type'])
        # 筛选出A股
        consensus['if_A'] = np.where((consensus['code'].str.len() == 6) &
                                     ((consensus['code'].str[0] == '0') | (consensus['code'].str[0] == '3') |
                                      (consensus['code'].str[0] == '6')), 1, 0)
        consensus = consensus.loc[consensus['if_A'] == 1, :]
        # 朝阳永续的code不带exchange，修改统一格式
        consensus['code'] = \
            np.where(consensus['code'].str[0] == '6', consensus['code'] + '.SH', consensus['code'] + '.SZ')
        # 计算score
        consensus['score'] = \
            np.where(consensus['score_type'].values == 1, consensus['score'].values, np.nan)
        score = consensus.loc[:, ['date', 'code', 'score']].copy()
        score.set_index('date', inplace=True)
        score = score.dropna()
        codes = score['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (score, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('score finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)
        # 计算 TPER
        consensus['target_price'] = \
            np.where(consensus['target_price_type'].values == 1, consensus['target_price'].values, np.nan)
        target_price = consensus.loc[:, ['date', 'code', 'target_price']].copy()
        target_price = target_price.dropna()
        # 获取股价
        curr_price = self.influx.getDataMultiprocess('DailyMarket_Gus', 'market', start, end, ['code', 'close'])
        curr_price.index.names = ['date']
        curr_price.reset_index(inplace=True)
        merge_df = pd.merge(curr_price, target_price, how='inner', on=['date', 'code'])
        merge_df['TPER'] = merge_df['target_price'] / merge_df['close'] - 1
        merge_df.drop('close', axis=1, inplace=True)
        merge_df.set_index('date', inplace=True)
        codes = merge_df['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (merge_df, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('TPER finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)

        return fail_list


if __name__ == '__main__':
    time_start = datetime.datetime.now()
    af = score_TPER()
    f = af.cal_factors(20100101, 20200402, N_JOBS)
    print(f)
    time_end = datetime.datetime.now()
    print('Time token:', time_end - time_start)
