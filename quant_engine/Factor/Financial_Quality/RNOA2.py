# 盈利能力因子 RNOA2 的计算
# 对齐 report period

from factor_base import FactorBase
import pandas as pd
import numpy as np
import datetime
from global_constant import N_JOBS
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData
from dateutil.relativedelta import relativedelta


class RNOA2(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'

    @staticmethod
    def get_lastQ_RP(rp, n_Qs):
        rp_dt = pd.to_datetime(rp) + datetime.timedelta(days=1) - \
                relativedelta(months=3 * n_Qs) - datetime.timedelta(days=1)
        return rp_dt.strftime('%Y%m%d')

    @staticmethod
    def JOB_cur_RNOA_Q(codes, df, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_df['NOA'] = code_df['NOA'].fillna(method='ffill')
            code_df[['NOA', 'NOA_last1Q']] = \
                code_df[['NOA', 'NOA_last1Q']].fillna(method='ffill', axis=1)
            code_df = code_df.drop_duplicates(['date'], 'last')
            code_df['RNOA_Q'] = \
                code_df['net_profit_Q'] / (code_df['NOA'] + code_df['NOA_last1Q']) * 2
            code_df.set_index('date', inplace=True)
            code_df = code_df.loc[:, ['code', 'RNOA_Q', 'report_period']]
            code_df = code_df.replace(np.inf, np.nan)
            code_df = code_df.replace(-np.inf, np.nan)
            code_df = code_df.dropna(subset=['RNOA_Q'])
            print('code: %s' % code)
            if code_df.empty:
                continue
            r = influx.saveData(code_df, db, measure)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('cur RNOA_Q Error: %s' % r)
        return save_res

    @staticmethod
    def JOB_cur_RNOA_TTM(codes, df, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_df['later_NOA'] = code_df['NOA'].fillna(method='ffill')
            conditions = [code_df['profit_last4Q_rp'].values == code_df['NOA_last4Q_rp'].values,
                          code_df['profit_last4Q_rp'].values == code_df['NOA_last3Q_rp'].values]
            choices = [code_df['NOA_last4Q'].values,
                       code_df['NOA_last3Q'].values]
            code_df['former_NOA'] = np.select(conditions, choices, default=np.nan)
            code_df[['later_NOA', 'former_NOA']] = \
                code_df[['later_NOA', 'former_NOA']].fillna(method='ffill', axis=1)
            code_df[['later_NOA', 'former_NOA']] = \
                code_df[['later_NOA', 'former_NOA']].fillna(method='bfill', axis=1)
            code_df['RNOA'] = code_df['net_profit_TTM'] / (code_df['later_NOA'] + code_df['former_NOA']) * 2
            code_df.set_index('date', inplace=True)
            code_df = code_df.loc[:, ['code', 'RNOA', 'report_period']]
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
                save_res.append('cur RNOA TTM Error: %s' % r)
        return save_res

    @staticmethod
    def JOB_hist_RNOA_Q(codes, df, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            dfs = []
            for i in range(1, 12):
                code_df = df.loc[df['code'] == code,
                                 ['date', 'code', 'net_profit_Q_last{0}Q'.format(i), 'profit_last{0}Q_rp'.format(i),
                                  'profit_last{0}Q_rp'.format(i+1), 'NOA_last{0}Q'.format(i-1),
                                  'NOA_last{0}Q'.format(i), 'NOA_last{0}Q'.format(i+1),
                                  'NOA_last{0}Q_rp'.format(i-1), 'NOA_last{0}Q_rp'.format(i),
                                  'NOA_last{0}Q_rp'.format(i+1)]].copy()
                conditions = [code_df['profit_last{0}Q_rp'.format(i)].values ==
                              code_df['NOA_last{0}Q_rp'.format(i)].values,
                              code_df['profit_last{0}Q_rp'.format(i)].values ==
                              code_df['NOA_last{0}Q_rp'.format(i-1)].values]
                choices = [code_df['NOA_last{0}Q'.format(i)].values,
                           code_df['NOA_last{0}Q'.format(i-1)].values]
                code_df['later_NOA'] = np.select(conditions, choices, default=np.nan)
                conditions = [code_df['profit_last{0}Q_rp'.format(i+1)].values ==
                              code_df['NOA_last{0}Q_rp'.format(i+1)].values,
                              code_df['profit_last{0}Q_rp'.format(i+1)].values ==
                              code_df['NOA_last{0}Q_rp'.format(i)].values]
                choices = [code_df['NOA_last{0}Q'.format(i+1)].values,
                           code_df['NOA_last{0}Q'.format(i)].values]
                code_df['former_NOA'] = np.select(conditions, choices, default=np.nan)
                code_df[['later_NOA', 'former_NOA']] = \
                    code_df[['later_NOA', 'former_NOA']].fillna(method='ffill', axis=1)
                code_df[['later_NOA', 'former_NOA']] = \
                    code_df[['later_NOA', 'former_NOA']].fillna(method='bfill', axis=1)
                code_df['RNOA_Q_last{0}Q'.format(i)] = code_df['net_profit_Q_last{0}Q'.format(i)] / \
                    (code_df['later_NOA'] + code_df['former_NOA']) * 2
                code_df.set_index(['date', 'code'], inplace=True)
                code_df = code_df.replace(np.inf, np.nan)
                code_df = code_df.replace(-np.inf, np.nan)
                code_df = code_df.loc[:, ['RNOA_Q_last{0}Q'.format(i)]].dropna()
                dfs.append(code_df)
            if not dfs:
                continue
            code_df = pd.concat(dfs, axis=1)
            code_df = code_df.reset_index().set_index('date')
            code_df = code_df.replace(np.inf, np.nan)
            code_df = code_df.replace(-np.inf, np.nan)
            code_df = code_df.where(pd.notnull(code_df), None)
            print('code: %s' % code)
            r = influx.saveData(code_df, db, measure)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('hist RNOA_Q Error: %s' % r)
        return save_res

    @staticmethod
    def JOB_hist_RNOA_TTM(codes, df, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            dfs = []
            for i in range(1, 9):
                code_df = df.loc[df['code'] == code,
                                 ['date', 'code',
                                  'net_profit_TTM_last{0}Q'.format(i),
                                  'profit_last{0}Q_rp'.format(i), 'profit_last{0}Q_rp'.format(i+4),
                                  'NOA_last{0}Q'.format(i-1), 'NOA_last{0}Q'.format(i),
                                  'NOA_last{0}Q'.format(i+3), 'NOA_last{0}Q'.format(i+4),
                                  'NOA_last{0}Q_rp'.format(i-1), 'NOA_last{0}Q_rp'.format(i),
                                  'NOA_last{0}Q_rp'.format(i+3), 'NOA_last{0}Q_rp'.format(i+4)]].copy()
                conditions = [code_df['profit_last{0}Q_rp'.format(i)].values ==
                              code_df['NOA_last{0}Q_rp'.format(i)].values,
                              code_df['profit_last{0}Q_rp'.format(i)].values ==
                              code_df['NOA_last{0}Q_rp'.format(i-1)].values]
                choices = [code_df['NOA_last{0}Q'.format(i)].values,
                           code_df['NOA_last{0}Q'.format(i-1)].values]
                code_df['later_NOA'] = np.select(conditions, choices, default=np.nan)
                conditions = [code_df['profit_last{0}Q_rp'.format(i+4)].values ==
                              code_df['NOA_last{0}Q_rp'.format(i+4)].values,
                              code_df['profit_last{0}Q_rp'.format(i+4)].values ==
                              code_df['NOA_last{0}Q_rp'.format(i+3)].values]
                choices = [code_df['NOA_last{0}Q'.format(i+4)].values,
                           code_df['NOA_last{0}Q'.format(i+3)].values]
                code_df['former_NOA'] = np.select(conditions, choices, default=np.nan)
                code_df[['later_NOA', 'former_NOA']] = \
                    code_df[['later_NOA', 'former_NOA']].fillna(method='ffill', axis=1)
                code_df[['later_NOA', 'former_NOA']] = \
                    code_df[['later_NOA', 'former_NOA']].fillna(method='bfill', axis=1)
                code_df['RNOA_last{0}Q'.format(i)] = code_df['net_profit_TTM_last{0}Q'.format(i)] / \
                    (code_df['later_NOA'] + code_df['former_NOA']) * 2
                code_df.set_index(['date', 'code'], inplace=True)
                code_df = code_df.replace(np.inf, np.nan)
                code_df = code_df.replace(-np.inf, np.nan)
                code_df = code_df.loc[:, ['RNOA_last{0}Q'.format(i)]].dropna()
                dfs.append(code_df)
            if not dfs:
                continue
            code_df = pd.concat(dfs, axis=1)
            code_df = code_df.reset_index().set_index('date')
            code_df = code_df.replace(np.inf, np.nan)
            code_df = code_df.replace(-np.inf, np.nan)
            code_df = code_df.where(pd.notnull(code_df), None)
            print('code: %s' % code)
            r = influx.saveData(code_df, db, measure)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('RNOA_Q Error: %s' % r)
        return save_res

    def cal_RNOA_Q(self):
        save_measure = 'RNOA2_Q'
        # get profit
        raw_profit = self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_profit_Q', self.start, self.end)
        raw_profit.index.names = ['date']
        raw_profit.reset_index(inplace=True)
        # --------------------------------------------------------------------------------------
        # 计算 cur RNOA_Q
        profit_df = raw_profit.loc[:, ['date', 'code', 'net_profit_Q', 'report_period']].copy()
        NOA_df = self.NOA.loc[:, ['date', 'code', 'NOA', 'NOA_last1Q', 'report_period']].copy()
        RNOA_df = pd.merge(profit_df, NOA_df, how='outer', on=['date', 'code', 'report_period'])
        RNOA_df = RNOA_df.sort_values(['date', 'code', 'report_period'])
        codes = RNOA_df['code'].unique()
        split_codes = np.array_split(codes, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            res = Parallel()(delayed(RNOA2.JOB_cur_RNOA_Q)
                             (codes, RNOA_df, self.db, save_measure) for codes in split_codes)
        print('cur RNOA_Q finish')
        print('-' * 30)
        for r in res:
            self.fail_list.extend(r)
        # --------------------------------------------------------------------------------------
        # 计算 history RNOA_Q
        profit_df = raw_profit.copy()
        for i in range(1, 13):
            cur_rps = []
            former_rps = []
            for rp in profit_df['report_period'].unique():
                cur_rps.append(rp)
                former_rps.append(RNOA2.get_lastQ_RP(rp, i))
            rp_dict = dict(zip(cur_rps, former_rps))
            profit_df['profit_last{0}Q_rp'.format(i)] = profit_df['report_period'].map(rp_dict)
        profit_df.drop('report_period', axis=1, inplace=True)
        NOA_df = self.NOA.copy()
        NOA_df.rename(columns={'NOA': 'NOA_last0Q'}, inplace=True)
        for i in range(0, 13):
            cur_rps = []
            former_rps = []
            for rp in NOA_df['report_period'].unique():
                cur_rps.append(rp)
                former_rps.append(RNOA2.get_lastQ_RP(rp, i))
            rp_dict = dict(zip(cur_rps, former_rps))
            NOA_df['NOA_last{0}Q_rp'.format(i)] = NOA_df['report_period'].map(rp_dict)
        NOA_df.drop('report_period', axis=1, inplace=True)
        RNOA_df = pd.merge(profit_df, NOA_df, how='outer', on=['date', 'code'])
        RNOA_df = RNOA_df.sort_values(['date', 'code'])
        codes = RNOA_df['code'].unique()
        split_codes = np.array_split(codes, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            res = Parallel()(delayed(RNOA2.JOB_hist_RNOA_Q)
                             (codes, RNOA_df, self.db, save_measure) for codes in split_codes)
        print('hist RNOA_Q finish')
        print('-' * 30)
        for r in res:
            self.fail_list.extend(r)


    def cal_RNOA_TTM(self):
        save_measure = 'RNOA2'
        # get profit
        raw_profit = self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_profit_TTM', self.start, self.end)
        raw_profit.index.names = ['date']
        raw_profit.reset_index(inplace=True)
        # --------------------------------------------------------------------------------------
        # 计算 cur RNOA
        profit_df = raw_profit.loc[:, ['date', 'code', 'net_profit_TTM', 'report_period']].copy()
        cur_rps = []
        former_rps = []
        for rp in profit_df['report_period'].unique():
            cur_rps.append(rp)
            former_rps.append(RNOA2.get_lastQ_RP(rp, 4))
        rp_dict = dict(zip(cur_rps, former_rps))
        profit_df['profit_last4Q_rp'] = profit_df['report_period'].map(rp_dict)
        NOA_df = self.NOA.loc[:, ['date', 'code', 'NOA', 'NOA_last3Q', 'NOA_last4Q', 'report_period']].copy()
        for i in [3, 4]:
            cur_rps = []
            former_rps = []
            for rp in NOA_df['report_period'].unique():
                cur_rps.append(rp)
                former_rps.append(RNOA2.get_lastQ_RP(rp, i))
            rp_dict = dict(zip(cur_rps, former_rps))
            NOA_df['NOA_last{0}Q_rp'.format(i)] = NOA_df['report_period'].map(rp_dict)
        NOA_df.drop('report_period', axis=1, inplace=True)
        RNOA_df = pd.merge(profit_df, NOA_df, how='outer', on=['date', 'code'])
        RNOA_df = RNOA_df.sort_values(['date', 'code', 'report_period'])
        codes = RNOA_df['code'].unique()
        split_codes = np.array_split(codes, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            res = Parallel()(delayed(RNOA2.JOB_cur_RNOA_TTM)
                             (codes, RNOA_df, self.db, save_measure) for codes in split_codes)
        print('cur RNOA_TTM finish')
        print('-' * 30)
        for r in res:
            self.fail_list.extend(r)
        # --------------------------------------------------------------------------------------
        # 计算 history RNOA
        profit_df = raw_profit.copy()
        for i in range(1, 13):
            cur_rps = []
            former_rps = []
            for rp in profit_df['report_period'].unique():
                cur_rps.append(rp)
                former_rps.append(RNOA2.get_lastQ_RP(rp, i))
            rp_dict = dict(zip(cur_rps, former_rps))
            profit_df['profit_last{0}Q_rp'.format(i)] = profit_df['report_period'].map(rp_dict)
        profit_df.drop('report_period', axis=1, inplace=True)
        NOA_df = self.NOA.copy()
        NOA_df.rename(columns={'NOA': 'NOA_last0Q'}, inplace=True)
        for i in range(0, 13):
            cur_rps = []
            former_rps = []
            for rp in NOA_df['report_period'].unique():
                cur_rps.append(rp)
                former_rps.append(RNOA2.get_lastQ_RP(rp, i))
            rp_dict = dict(zip(cur_rps, former_rps))
            NOA_df['NOA_last{0}Q_rp'.format(i)] = NOA_df['report_period'].map(rp_dict)
        NOA_df.drop('report_period', axis=1, inplace=True)
        RNOA_df = pd.merge(profit_df, NOA_df, how='outer', on=['date', 'code'])
        RNOA_df = RNOA_df.sort_values(['date', 'code'])
        codes = RNOA_df['code'].unique()
        split_codes = np.array_split(codes, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            res = Parallel()(delayed(RNOA2.JOB_hist_RNOA_TTM)
                             (codes, RNOA_df, self.db, save_measure) for codes in split_codes)
        print('hist RNOA_TTM finish')
        print('-' * 30)
        for r in res:
            self.fail_list.extend(r)


    def cal_factors(self, start, end, n_jobs):
        pd.set_option('mode.use_inf_as_na', True)
        self.start = start
        self.end = end
        self.n_jobs = n_jobs
        self.fail_list = []
        # get net NOA
        self.NOA = \
            self.influx.getDataMultiprocess('FinancialReport_Gus', 'NOA', start, end)
        self.NOA.index.names = ['date']
        self.NOA.reset_index(inplace=True)
        self.cal_RNOA_Q()
        self.cal_RNOA_TTM()

        return self.fail_list


if __name__ == '__main__':
    RNOA = RNOA2()
    r = RNOA.cal_factors(20090101, 20200525, N_JOBS)
    print('task finish')
    print(r)
    print(datetime.datetime.now())