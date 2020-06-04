from factor_base import FactorBase
from data_process import DataProcess
import pandas as pd
import numpy as np
import datetime
from influxdb_data import influxdbData
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS


class NetProfitFY1_growth(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'net_profit_FY1_growth'

    @staticmethod
    def JOB_factors(codes, df, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            conditions = \
                [code_df['FY0_rp'].values == code_df['last0Q_rp'].values,
                 code_df['FY0_rp'].values == code_df['last1Q_rp'].values,
                 code_df['FY0_rp'].values == code_df['last2Q_rp'].values,
                 code_df['FY0_rp'].values == code_df['last3Q_rp'].values,
                 code_df['FY0_rp'].values == code_df['last4Q_rp'].values,
                 code_df['FY0_rp'].values == code_df['last5Q_rp'].values,
                 code_df['FY0_rp'].values == code_df['last6Q_rp'].values]
            choices = \
                [code_df['net_profit_TTM'].values, code_df['net_profit_TTM_last1Q'].values,
                 code_df['net_profit_TTM_last2Q'].values, code_df['net_profit_TTM_last3Q'].values,
                 code_df['net_profit_TTM_last4Q'].values, code_df['net_profit_TTM_last5Q'].values,
                 code_df['net_profit_TTM_last6Q'].values]
            code_df['net_profit_FY0'] = np.select(conditions, choices, default=np.nan)
            code_df['net_profit_FY1_growthY'] = code_df['net_profit_FY1'] / code_df['net_profit_FY0'] - 1
            code_df = code_df.loc[:, ['code', 'net_profit_FY1_growthY', 'report_period']]
            code_df = code_df.replace(np.inf, np.nan)
            code_df = code_df.replace(-np.inf, np.nan)
            code_df = code_df.where(pd.notnull(code_df), None)
            if code_df.empty:
                continue
            print('code: %s' % code)
            r = influx.saveData(code_df, db, measure)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('NetProfitFY1_growth Error: %s' % r)
        return save_res

    def cal_factors(self, start, end, n_jobs):
        pd.set_option('mode.use_inf_as_na', True)
        net_profit_FY1 = self.influx.getDataMultiprocess('DailyFactors_Gus', 'AnalystNetProfit', start, end,
                                                         ['code', 'net_profit_FY1', 'report_period'])
        net_profit_FY1.index.names = ['date']
        rps = net_profit_FY1['report_period'].unique()
        values = []
        for rp in rps:
            values.append(DataProcess.get_former_RP(rp, 4))
        former_rp_dict = dict(zip(rps, values))
        net_profit_FY1['FY0_rp'] = net_profit_FY1['report_period'].map(former_rp_dict)
        net_profit_TTM = self.influx.getDataMultiprocess(
            'FinancialReport_Gus', 'net_profit_TTM', start, end,
            ['code', 'report_period', 'net_profit_TTM', 'net_profit_TTM_last1Q', 'net_profit_TTM_last2Q',
             'net_profit_TTM_last3Q', 'net_profit_TTM_last4Q', 'net_profit_TTM_last5Q', 'net_profit_TTM_last6Q'])
        net_profit_TTM.index.names = ['date']
        net_profit_TTM.rename(columns={'report_period': 'last0Q_rp'}, inplace=True)
        rps = net_profit_TTM['last0Q_rp'].unique()
        for i in range(1, 7):
            values = []
            for rp in rps:
                values.append(DataProcess.get_former_RP(rp, i))
                former_rp_dict = dict(zip(rps, values))
                net_profit_TTM['last{0}Q_rp'.format(i)] = net_profit_TTM['last0Q_rp'].map(former_rp_dict)
        merge = pd.merge(net_profit_FY1.reset_index(), net_profit_TTM.reset_index(), how='left', on=['date', 'code'])
        merge.set_index('date', inplace=True)
        codes = merge['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(NetProfitFY1_growth.JOB_factors)
                             (codes, merge, self.db, self.measure) for codes in split_codes)
        print('net_profit_FY1_growth finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    pd.set_option('mode.use_inf_as_na', True)
    start_dt = datetime.datetime.now()
    g = NetProfitFY1_growth()
    r = g.cal_factors(20090101, 20200602, N_JOBS)
    print(r)
    print('time token:', datetime.datetime.now()-start_dt)