import pandas as pd
import numpy as np
from factor_base import FactorBase
import datetime
from global_constant import N_JOBS
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData


class PEG(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'PEG'

    # 计算两个报告期相差几个季度的工具函数
    @staticmethod
    def report_period_diff(former_rp, later_rp):
        dt_former = dtparser.parse(former_rp) + datetime.timedelta(days=1)
        dt_later = dtparser.parse(later_rp) + datetime.timedelta(days=1)
        Q_delta = relativedelta(months=3)
        if dt_former == dt_later:
            return 0
        elif dt_former < dt_later:
            n = 1
            dt_process = dt_former + Q_delta
            while dt_process < dt_later:
                n += 1
                dt_process += Q_delta
            if dt_former + n * Q_delta != dt_later:
                print('Report Period Error!')
                raise NameError
            return n
        else:
            n = -1
            dt_process = dt_former - Q_delta
            while dt_process > dt_later:
                n -= 1
                dt_process -= Q_delta
            if dt_former + n * Q_delta != dt_later:
                print('Report Period Error!')
                raise NameError
            return n

    def cal_factors(self, start, end, n_jobs):
        # 获取净利润数据
        con_net_profit = self.influx.getDataMultiprocess('DailyFactors_Gus', 'AnalystNetProfit', start, end,
                                                         ['code', 'net_profit_FY1', 'net_profit_FY2'])
        con_net_profit.index.names = ['date']
        con_net_profit.reset_index(inplace=True)
        net_profit_FY0 = self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_profit', start, end,
                                                         ['code', 'report_period', 'net_profit', 'net_profit_last1Q',
                                                          'net_profit_last2Q', 'net_profit_last3Q', 'net_profit_last4Q',
                                                          'net_profit_last5Q', 'net_profit_last6Q'])
        net_profit_FY0.index.names = ['date']
        net_profit_FY0.reset_index(inplace=True)
        merge_df = pd.merge(con_net_profit, net_profit_FY0, how='inner', on=['date', 'code'])
        merge_df.set_index('date', inplace=True)
        merge_df['FY0_report_period'] = np.where(merge_df.index.month < 5,
                                                 merge_df.index.year-2, merge_df.index.year-1)
        merge_df['FY0_report_period'] = merge_df['FY0_report_period'].astype('int').astype('str') + '1231'
        merge_df.reset_index(inplace=True)
        unique_rps = merge_df[['FY0_report_period', 'report_period']].drop_duplicates()
        vect_func = np.vectorize(PEG.report_period_diff)
        unique_rps['report_period_diff'] = \
            vect_func(unique_rps['FY0_report_period'].values, unique_rps['report_period'].values)
        merge_df = pd.merge(merge_df, unique_rps, how='left', on=['FY0_report_period', 'report_period'])
        # report_period_diff<0 的情况一般为未上市，>=7的情况相差太远，不考虑
        merge_df = merge_df.loc[(merge_df['report_period_diff'] >= 0) & (merge_df['report_period_diff'] < 7), :]
        merge_df = merge_df.dropna(subset=['report_period_diff'])
        conditions = [merge_df['report_period_diff'].values == 0,
                      merge_df['report_period_diff'].values == 1,
                      merge_df['report_period_diff'].values == 2,
                      merge_df['report_period_diff'].values == 3,
                      merge_df['report_period_diff'].values == 4,
                      merge_df['report_period_diff'].values == 5,
                      merge_df['report_period_diff'].values == 6]
        choices = [merge_df['net_profit'].values,
                   merge_df['net_profit_last1Q'].values,
                   merge_df['net_profit_last2Q'].values,
                   merge_df['net_profit_last3Q'].values,
                   merge_df['net_profit_last4Q'].values,
                   merge_df['net_profit_last5Q'].values,
                   merge_df['net_profit_last6Q'].values]
        merge_df['net_profit_FY0'] = np.select(conditions, choices, default=np.nan)
        merge_df = merge_df.loc[(merge_df['net_profit_FY0'] >= 0) & (merge_df['net_profit_FY1'] >= 0) &
                                (merge_df['net_profit_FY2'] >= 0),
                                ['date', 'code', 'net_profit_FY0', 'net_profit_FY1', 'net_profit_FY2']]
        market_cap = self.influx.getDataMultiprocess('DailyFactors_Gus', 'Size', start, end, ['code', 'market_cap'])
        market_cap.index.names = ['date']
        market_cap.reset_index(inplace=True)
        merge_df = pd.merge(merge_df, market_cap, how='inner', on=['date', 'code'])
        # market_cap 单位为万元
        merge_df['PEG'] = merge_df['market_cap'] * 10000 / merge_df['net_profit_FY1'] / \
                          (merge_df['net_profit_FY2'] / merge_df['net_profit_FY0'])
        merge_df['PEG2'] = merge_df['market_cap'] * 10000 / merge_df['net_profit_FY1'] / \
                           (100 * np.sqrt(merge_df['net_profit_FY2'] / merge_df['net_profit_FY0']) - 1)
        merge_df = merge_df.loc[:, ['date', 'code', 'PEG', 'PEG2']]
        merge_df.set_index('date', inplace=True)
        merge_df = merge_df.replace(np.inf, np.nan)
        merge_df = merge_df.replace(-np.inf, np.nan)
        merge_df = merge_df.loc[np.any(pd.notnull(merge_df[['PEG', 'PEG2']]), axis=1), :]
        codes = merge_df['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (merge_df, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('PEG finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)

        return fail_list


if __name__ == '__main__':
    time_start = datetime.datetime.now()
    af = PEG()
    f = af.cal_factors(20090101, 20200706, N_JOBS)
    print(f)
    time_end = datetime.datetime.now()
    print('Time token:', time_end - time_start)
