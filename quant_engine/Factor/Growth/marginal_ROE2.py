# 因子 边际股权回报率 的计算
# 对齐 report period

from factor_base import FactorBase
import pandas as pd
import numpy as np
import datetime
from global_constant import N_JOBS
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData
from data_process import DataProcess


class MarginalROE(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'marginal_ROE'

    @staticmethod
    def JOB_factors(codes, df, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            conditions = [code_df['profit_last4Q_rp'].values == code_df['equity_last4Q_rp'].values,
                          code_df['profit_last5Q_rp'].values == code_df['equity_last4Q_rp'].values]
            choices = [code_df['net_profit_TTM_last4Q'].values,
                       code_df['net_profit_TTM_last5Q'].values]
            code_df['former_profit'] = np.select(conditions, choices, default=np.nan)
            conditions = [code_df['profit_last0Q_rp'].values == code_df['report_period'].values,
                          code_df['profit_last1Q_rp'].values == code_df['report_period'].values]
            choices = [code_df['net_profit_TTM'].values,
                       code_df['net_profit_TTM_last1Q'].values]
            code_df['later_profit'] = np.select(conditions, choices, default=np.nan)
            code_df['delta_equity'] = code_df['net_equity'] - code_df['net_equity_last4Q']
            code_df['delta_profit'] = code_df['later_profit'] - code_df['former_profit']
            code_df['marginal_ROE'] = code_df['delta_profit'] / code_df['delta_equity']
            code_df.set_index('date', inplace=True)
            code_df = code_df.loc[:, ['code', 'report_period', 'marginal_ROE']]
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
                save_res.append('marginal ROE Error: %s' % r)
        return save_res

    def cal_marginal_ROE(self):
        # get net profit
        net_profit = self.influx.getDataMultiprocess(
            'FinancialReport_Gus', 'net_profit_TTM', self.start, self.end,
            ['code', 'report_period', 'net_profit_TTM', 'net_profit_TTM_last1Q', 'net_profit_TTM_last4Q',
             'net_profit_TTM_last5Q'])
        net_profit.index.names = ['date']
        net_profit.reset_index(inplace=True)
        for i in [0, 1, 4, 5]:
            cur_rps = []
            former_rps = []
            for rp in net_profit['report_period'].unique():
                cur_rps.append(rp)
                former_rps.append(DataProcess.get_former_RP(rp, i))
            rp_dict = dict(zip(cur_rps, former_rps))
            net_profit['profit_last{0}Q_rp'.format(i)] = net_profit['report_period'].map(rp_dict)
        net_profit.drop('report_period', axis=1, inplace=True)
        # get net equity
        net_equity = self.influx.getDataMultiprocess(
            'FinancialReport_Gus', 'net_equity', self.start, self.end,
            ['code', 'report_period', 'net_equity', 'net_equity_last4Q'])
        net_equity.index.names = ['date']
        net_equity.reset_index(inplace=True)
        cur_rps = []
        former_rps = []
        for rp in net_equity['report_period'].unique():
            cur_rps.append(rp)
            former_rps.append(DataProcess.get_former_RP(rp, 4))
        rp_dict = dict(zip(cur_rps, former_rps))
        net_equity['equity_last4Q_rp'.format(i)] = net_equity['report_period'].map(rp_dict)
        # --------------------------------------------------------------------------------------
        ROE_df = pd.merge(net_profit, net_equity, how='outer', on=['date', 'code'])
        ROE_df = ROE_df.sort_values(['date', 'code', 'report_period'])
        codes = ROE_df['code'].unique()
        split_codes = np.array_split(codes, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            res = Parallel()(delayed(MarginalROE.JOB_factors)
                             (codes, ROE_df, self.db, self.measure) for codes in split_codes)
        print('marginal ROE finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list

    def cal_factors(self, start, end, n_jobs):
        pd.set_option('mode.use_inf_as_na', True)
        self.start = start
        self.end = end
        self.n_jobs = n_jobs
        fail_list = self.cal_marginal_ROE()
        return fail_list


if __name__ == '__main__':
    roe = MarginalROE()
    r = roe.cal_factors(20090101, 20200609, N_JOBS)
    print('task finish')
    print(r)
    print(datetime.datetime.now())