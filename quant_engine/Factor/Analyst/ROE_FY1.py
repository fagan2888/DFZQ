# 盈利能力因子 ROE_FY1 的计算
# 对齐 report period

from factor_base import FactorBase
import pandas as pd
import numpy as np
import datetime
from global_constant import N_JOBS, FACTOR_DB
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData
from data_process import DataProcess


class ROE_FY1(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'

    @staticmethod
    def JOB_cur_ROE_TTM(codes, df, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            conditions = [code_df['FY0_rp'].values == code_df['equity_last0Q_rp'].values,
                          code_df['FY0_rp'].values == code_df['equity_last1Q_rp'].values,
                          code_df['FY0_rp'].values == code_df['equity_last2Q_rp'].values,
                          code_df['FY0_rp'].values == code_df['equity_last3Q_rp'].values,
                          code_df['FY0_rp'].values == code_df['equity_last4Q_rp'].values,
                          code_df['FY0_rp'].values == code_df['equity_last5Q_rp'].values,
                          code_df['FY0_rp'].values == code_df['equity_last6Q_rp'].values]
            choices = [code_df['net_equity'].values, code_df['net_equity_last1Q'].values,
                       code_df['net_equity_last2Q'].values, code_df['net_equity_last3Q'].values,
                       code_df['net_equity_last4Q'].values, code_df['net_equity_last5Q'].values,
                       code_df['net_equity_last6Q'].values]
            code_df['ROE_equity'] = np.select(conditions, choices, default=np.nan)
            # 用最近的非nan值填充ROE_equity
            code_df[['net_equity_last6Q', 'net_equity_last5Q', 'net_equity_last4Q', 'net_equity_last3Q',
                     'net_equity_last2Q', 'net_equity_last1Q', 'net_equity', 'ROE_equity']] = \
                code_df[['net_equity_last6Q', 'net_equity_last5Q', 'net_equity_last4Q', 'net_equity_last3Q',
                         'net_equity_last2Q', 'net_equity_last1Q', 'net_equity', 'ROE_equity']].fillna(
                    method='ffill', axis=1)
            # 计算 ROE_FY1
            code_df['ROE_FY1'] = code_df['net_profit_FY1'] / code_df['ROE_equity']
            code_df.set_index('date', inplace=True)
            code_df = code_df.loc[:, ['code', 'ROE_FY1', 'report_period']]
            code_df = code_df.replace(np.inf, np.nan)
            code_df = code_df.replace(-np.inf, np.nan)
            code_df = code_df.dropna()
            print('code: %s' % code)
            if code_df.empty:
                continue
            r = influx.saveData(code_df, db, measure)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('ROE_FY1 Error: %s' % r)
        return save_res

    def cal_ROE_TTM(self):
        save_measure = 'ROE_FY1'
        # get profit
        profit_df = self.influx.getDataMultiprocess(FACTOR_DB, 'AnalystNetProfit', self.start, self.end,
                                                    ['code', 'net_profit_FY1', 'report_period'])
        profit_df.index.names = ['date']
        profit_df.reset_index(inplace=True)
        # --------------------------------------------------------------------------------------
        # 计算 ROE_FY1
        cur_rps = []
        former_rps = []
        for rp in profit_df['report_period'].unique():
            cur_rps.append(rp)
            former_rps.append(DataProcess.get_former_RP(rp, 4))
        rp_dict = dict(zip(cur_rps, former_rps))
        profit_df['FY0_rp'] = profit_df['report_period'].map(rp_dict)
        equity_df = self.net_equity.copy()
        for i in range(1, 7):
            cur_rps = []
            former_rps = []
            for rp in equity_df['report_period'].unique():
                cur_rps.append(rp)
                former_rps.append(DataProcess.get_former_RP(rp, i))
            rp_dict = dict(zip(cur_rps, former_rps))
            equity_df['equity_last{0}Q_rp'.format(i)] = equity_df['report_period'].map(rp_dict)
        equity_df.rename(columns={'report_period': 'equity_last0Q_rp'}, inplace=True)
        ROE_df = pd.merge(profit_df, equity_df, how='inner', on=['date', 'code'])
        ROE_df = ROE_df.sort_values(['date', 'code', 'report_period'])
        codes = ROE_df['code'].unique()
        split_codes = np.array_split(codes, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            res = Parallel()(delayed(ROE_FY1.JOB_cur_ROE_TTM)
                             (codes, ROE_df, self.db, save_measure) for codes in split_codes)
        print('cur ROE_TTM finish')
        print('-' * 30)
        for r in res:
            self.fail_list.extend(r)


    def cal_factors(self, start, end, n_jobs):
        pd.set_option('mode.use_inf_as_na', True)
        self.start = start
        self.end = end
        self.n_jobs = n_jobs
        self.fail_list = []
        # get net equity
        self.net_equity = \
            self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_equity', start, end)
        self.net_equity.index.names = ['date']
        self.net_equity.reset_index(inplace=True)
        self.cal_ROE_TTM()
        return self.fail_list


if __name__ == '__main__':
    roe = ROE_FY1()
    r = roe.cal_factors(20200101, 20200525, N_JOBS)
    print('task finish')
    print(r)
    print(datetime.datetime.now())