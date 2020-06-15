from influxdb_data import influxdbData
import numpy as np
import pandas as pd
from factor_base import FactorBase
from joblib import Parallel, delayed, parallel_backend
import global_constant
import datetime


class NetProfitTrend(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'net_profit_trend'

    @staticmethod
    def JOB_factors(dates, df, db, measure):
        influx = influxdbData()
        save_res = []
        for date in dates:
            day_df = df.loc[df['date'] == date, :].copy()
            if day_df.shape[0] < 50:
                continue
            indus = day_df['improved_lv1'].unique()
            dfs = []
            for indu in indus:
                day_indu_df = day_df.loc[day_df['improved_lv1'] == indu, :].copy()
                # 保险行业股票数过少
                if day_indu_df.shape[0] < 3:
                    day_indu_df['net_profit_trend'] = 5
                else:
                    day_indu_df['growth_group'] = \
                        pd.qcut(day_indu_df['net_profit_Q_growthY'], 3, labels=[1, 2, 3]).astype('int')
                    day_indu_df['acc_group'] = \
                        pd.qcut(day_indu_df['net_profit_Q_acc'], 3, labels=[1, 2, 3]).astype('int')
                    day_indu_df['net_profit_trend'] = 3 * (day_indu_df['growth_group'] - 1) + day_indu_df['acc_group']
                dfs.append(day_indu_df)
            day_df = pd.concat(dfs)
            day_df.set_index('date', inplace=True)
            day_df = day_df.loc[:, ['code', 'report_period', 'net_profit_trend']]
            print('date: %s' % date)
            r = influx.saveData(day_df, db, measure)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('%s Error: %s' % (measure, r))
        return save_res

    def cal_factors(self, start, end, n_jobs):
        fail_list = []
        # get growth
        growth = self.influx.getDataMultiprocess(global_constant.FACTOR_DB, 'net_profit_Q_growth', start, end,
                                                 ['code', 'net_profit_Q_growthY', 'report_period'])
        growth.index.names = ['date']
        # get acc
        acc = self.influx.getDataMultiprocess(global_constant.FACTOR_DB, 'net_profit_Q_acc', start, end,
                                              ['code', 'net_profit_Q_acc', 'report_period'])
        acc.index.names = ['date']
        # get industry
        indu = self.influx.getDataMultiprocess(global_constant.MARKET_DB, 'industry', start, end,
                                               ['code', 'improved_lv1'])
        indu.index.names = ['date']
        merge = pd.merge(growth.reset_index(), acc.reset_index(), on=['date', 'code', 'report_period'])
        merge = pd.merge(merge, indu.reset_index(), on=['date', 'code'])
        dates = merge['date'].unique()
        split_dates = np.array_split(dates, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(NetProfitTrend.JOB_factors)
                             (dates, merge, self.db, self.measure) for dates in split_dates)
        print('net_profit_trend finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    start_dt = datetime.datetime.now()
    trd = NetProfitTrend()
    r = trd.cal_factors(20090101, 20200610, global_constant.N_JOBS)
    print(r)
    print('time token:', datetime.datetime.now() - start_dt)
