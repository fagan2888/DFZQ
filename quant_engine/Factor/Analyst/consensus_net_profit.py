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


class consensus_net_profit(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'Analyst'
        self.gogoal = GoGoal_data()

    def cal_factors(self, start, end, n_jobs):
        self.n_jobs = n_jobs

        # 获取一致预期数据
        query = "SELECT [CON_DATE], [STOCK_CODE], [RPT_DATE], [C4], [C82]" \
                "FROM [{0}].[dbo].[CON_FORECAST_STK] " \
                "WHERE [CON_DATE] >= '{1}' and [CON_DATE] <= '{2}' and [RPT_TYPE] = 4 " \
                "order by [CON_DATE] " \
            .format(self.gogoal.database,
                    (dtparser.parse(str(start)) - relativedelta(months=1)).strftime('%Y%m%d'), end)
        self.gogoal.cur.execute(query)
        consensus = pd.DataFrame(self.gogoal.cur.fetchall(),
                                 columns=['date', 'code', 'year', 'consensus_net_profit', 'change_rate_3m'])
        # 筛选出A股
        consensus['if_A'] = np.where((consensus['code'].str.len() == 6) &
                                     ((consensus['code'].str[0] == '0') | (consensus['code'].str[0] == '3') |
                                      (consensus['code'].str[0] == '6')), 1, 0)
        consensus = consensus.loc[consensus['if_A'] == 1, :]
        consensus.dropna(subset=['year'], inplace=True)
        # 朝阳永续的code不带exchange，修改统一格式
        consensus['code'] = \
            np.where(consensus['code'].str[0] == '6', consensus['code'] + '.SH', consensus['code'] + '.SZ')
        # 原数据单位为万元
        consensus['consensus_net_profit'] = consensus['consensus_net_profit'] * 10000
        # 设定在每年的4月31日作为财年切换
        consensus.set_index('date', inplace=True)
        consensus['year_FY1'] = np.where(consensus.index.month < 5, consensus.index.year - 1, consensus.index.year)
        consensus['year_FY2'] = np.where(consensus.index.month < 5, consensus.index.year, consensus.index.year + 1)
        consensus['net_profit_FY1'] = np.where(consensus['year_FY1'].values == consensus['year'].values,
                                               consensus['consensus_net_profit'].values, np.nan)
        consensus['net_profit_FY2'] = np.where(consensus['year_FY2'].values == consensus['year'].values,
                                               consensus['consensus_net_profit'].values, np.nan)
        net_profit_FY1 = consensus.groupby(['date', 'code'])['net_profit_FY1'].max()
        net_profit_FY2 = consensus.groupby(['date', 'code'])['net_profit_FY2'].max()
        consensus = pd.merge(net_profit_FY1.reset_index(), net_profit_FY2.reset_index(),
                             on=['date', 'code'], how='outer')
        consensus.set_index('date', inplace=True)
        consensus['report_period'] = np.where(consensus.index.month < 5, consensus.index.year - 1, consensus.index.year)
        consensus['report_period'] = consensus['report_period'].astype('int').astype('str') + '1231'
        consensus.reset_index(inplace=True)
        # ---------------------------------------------------------------------------------------------
        # 获取财报数据
        announcement = self.influx.getDataMultiprocess(
            'FinancialReport_Gus', 'net_profit', start, end, ['code', 'net_profit', 'report_period'])
        announcement.index.names = ['date']
        announcement.reset_index(inplace=True)
        # ---------------------------------------------------------------------------------------------
        merge_df = pd.merge(consensus, announcement, how='left', on=['date', 'code', 'report_period'])
        merge_df['net_profit_FY1'] = np.where(pd.isnull(merge_df['net_profit'].values),
                                              merge_df['net_profit_FY1'].values, merge_df['net_profit'].values)
        merge_df.drop('net_profit', axis=1, inplace=True)
        merge_df.set_index('date', inplace=True)
        merge_df = merge_df.where(pd.notnull(merge_df), None)
        # ---------------------------------------------------------------------------------------------
        # 储存数据
        codes = merge_df['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (merge_df, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('consensus net profit finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    time_start = datetime.datetime.now()
    af = consensus_net_profit()
    f = af.cal_factors(20100101, 20200402, N_JOBS)
    print(f)
    time_end = datetime.datetime.now()
    print('Time token:', time_end - time_start)
