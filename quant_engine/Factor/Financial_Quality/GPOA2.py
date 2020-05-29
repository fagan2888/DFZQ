# 盈利能力因子 GPOA2 的计算
# 对齐 report period

from factor_base import FactorBase
import pandas as pd
import numpy as np
import datetime
from global_constant import N_JOBS
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData
from dateutil.relativedelta import relativedelta


class GPOA2(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'

    @staticmethod
    def get_lastQ_RP(rp, n_Qs):
        rp_dt = pd.to_datetime(rp) + datetime.timedelta(days=1) - \
                relativedelta(months=3 * n_Qs) - datetime.timedelta(days=1)
        return rp_dt.strftime('%Y%m%d')

    @staticmethod
    def JOB_cur_GPOA_Q(codes, df, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_df['tot_assets'] = code_df['tot_assets'].fillna(method='ffill')
            code_df[['tot_assets', 'tot_assets_last1Q']] = \
                code_df[['tot_assets', 'tot_assets_last1Q']].fillna(method='ffill', axis=1)
            code_df = code_df.drop_duplicates(['date'], 'last')
            code_df['GPOA_Q'] = \
                code_df['tot_profit_Q'] / (code_df['tot_assets'] + code_df['tot_assets_last1Q']) * 2
            code_df.set_index('date', inplace=True)
            code_df = code_df.loc[:, ['code', 'GPOA_Q', 'report_period']]
            code_df = code_df.replace(np.inf, np.nan)
            code_df = code_df.replace(-np.inf, np.nan)
            code_df = code_df.dropna(subset=['GPOA_Q'])
            print('code: %s' % code)
            if code_df.empty:
                continue
            r = influx.saveData(code_df, db, measure)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('cur GPOA_Q Error: %s' % r)
        return save_res

    @staticmethod
    def JOB_cur_GPOA_TTM(codes, df, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_df['later_assets'] = code_df['tot_assets'].fillna(method='ffill')
            conditions = [code_df['profit_last4Q_rp'].values == code_df['assets_last4Q_rp'].values,
                          code_df['profit_last4Q_rp'].values == code_df['assets_last3Q_rp'].values]
            choices = [code_df['tot_assets_last4Q'].values,
                       code_df['tot_assets_last3Q'].values]
            code_df['former_assets'] = np.select(conditions, choices, default=np.nan)
            code_df[['later_assets', 'former_assets']] = \
                code_df[['later_assets', 'former_assets']].fillna(method='ffill', axis=1)
            code_df[['later_assets', 'former_assets']] = \
                code_df[['later_assets', 'former_assets']].fillna(method='bfill', axis=1)
            code_df['GPOA'] = code_df['tot_profit_TTM'] / (code_df['later_assets'] + code_df['former_assets']) * 2
            code_df.set_index('date', inplace=True)
            code_df = code_df.loc[:, ['code', 'GPOA', 'report_period']]
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
                save_res.append('cur GPOA TTM Error: %s' % r)
        return save_res

    @staticmethod
    def JOB_hist_GPOA_Q(codes, df, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            dfs = []
            for i in range(1, 12):
                code_df = df.loc[df['code'] == code,
                                 ['date', 'code', 'tot_profit_Q_last{0}Q'.format(i), 'profit_last{0}Q_rp'.format(i),
                                  'profit_last{0}Q_rp'.format(i+1), 'tot_assets_last{0}Q'.format(i-1),
                                  'tot_assets_last{0}Q'.format(i), 'tot_assets_last{0}Q'.format(i+1),
                                  'assets_last{0}Q_rp'.format(i-1), 'assets_last{0}Q_rp'.format(i),
                                  'assets_last{0}Q_rp'.format(i+1)]].copy()
                conditions = [code_df['profit_last{0}Q_rp'.format(i)].values ==
                              code_df['assets_last{0}Q_rp'.format(i)].values,
                              code_df['profit_last{0}Q_rp'.format(i)].values ==
                              code_df['assets_last{0}Q_rp'.format(i-1)].values]
                choices = [code_df['tot_assets_last{0}Q'.format(i)].values,
                           code_df['tot_assets_last{0}Q'.format(i-1)].values]
                code_df['later_assets'] = np.select(conditions, choices, default=np.nan)
                conditions = [code_df['profit_last{0}Q_rp'.format(i+1)].values ==
                              code_df['assets_last{0}Q_rp'.format(i+1)].values,
                              code_df['profit_last{0}Q_rp'.format(i+1)].values ==
                              code_df['assets_last{0}Q_rp'.format(i)].values]
                choices = [code_df['tot_assets_last{0}Q'.format(i+1)].values,
                           code_df['tot_assets_last{0}Q'.format(i)].values]
                code_df['former_assets'] = np.select(conditions, choices, default=np.nan)
                code_df[['later_assets', 'former_assets']] = \
                    code_df[['later_assets', 'former_assets']].fillna(method='ffill', axis=1)
                code_df[['later_assets', 'former_assets']] = \
                    code_df[['later_assets', 'former_assets']].fillna(method='bfill', axis=1)
                code_df['GPOA_Q_last{0}Q'.format(i)] = code_df['tot_profit_Q_last{0}Q'.format(i)] / \
                    (code_df['later_assets'] + code_df['former_assets']) * 2
                code_df.set_index(['date', 'code'], inplace=True)
                code_df = code_df.replace(np.inf, np.nan)
                code_df = code_df.replace(-np.inf, np.nan)
                code_df = code_df.loc[:, ['GPOA_Q_last{0}Q'.format(i)]].dropna()
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
                save_res.append('hist GPOA_Q Error: %s' % r)
        return save_res

    @staticmethod
    def JOB_hist_GPOA_TTM(codes, df, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            dfs = []
            for i in range(1, 9):
                code_df = df.loc[df['code'] == code,
                                 ['date', 'code',
                                  'tot_profit_TTM_last{0}Q'.format(i),
                                  'profit_last{0}Q_rp'.format(i), 'profit_last{0}Q_rp'.format(i+4),
                                  'tot_assets_last{0}Q'.format(i-1), 'tot_assets_last{0}Q'.format(i),
                                  'tot_assets_last{0}Q'.format(i+3), 'tot_assets_last{0}Q'.format(i+4),
                                  'assets_last{0}Q_rp'.format(i-1), 'assets_last{0}Q_rp'.format(i),
                                  'assets_last{0}Q_rp'.format(i+3), 'assets_last{0}Q_rp'.format(i+4)]].copy()
                conditions = [code_df['profit_last{0}Q_rp'.format(i)].values ==
                              code_df['assets_last{0}Q_rp'.format(i)].values,
                              code_df['profit_last{0}Q_rp'.format(i)].values ==
                              code_df['assets_last{0}Q_rp'.format(i-1)].values]
                choices = [code_df['tot_assets_last{0}Q'.format(i)].values,
                           code_df['tot_assets_last{0}Q'.format(i-1)].values]
                code_df['later_assets'] = np.select(conditions, choices, default=np.nan)
                conditions = [code_df['profit_last{0}Q_rp'.format(i+4)].values ==
                              code_df['assets_last{0}Q_rp'.format(i+4)].values,
                              code_df['profit_last{0}Q_rp'.format(i+4)].values ==
                              code_df['assets_last{0}Q_rp'.format(i+3)].values]
                choices = [code_df['tot_assets_last{0}Q'.format(i+4)].values,
                           code_df['tot_assets_last{0}Q'.format(i+3)].values]
                code_df['former_assets'] = np.select(conditions, choices, default=np.nan)
                code_df[['later_assets', 'former_assets']] = \
                    code_df[['later_assets', 'former_assets']].fillna(method='ffill', axis=1)
                code_df[['later_assets', 'former_assets']] = \
                    code_df[['later_assets', 'former_assets']].fillna(method='bfill', axis=1)
                code_df['GPOA_last{0}Q'.format(i)] = code_df['tot_profit_TTM_last{0}Q'.format(i)] / \
                    (code_df['later_assets'] + code_df['former_assets']) * 2
                code_df.set_index(['date', 'code'], inplace=True)
                code_df = code_df.replace(np.inf, np.nan)
                code_df = code_df.replace(-np.inf, np.nan)
                code_df = code_df.loc[:, ['GPOA_last{0}Q'.format(i)]].dropna()
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
                save_res.append('GPOA_Q Error: %s' % r)
        return save_res

    def cal_GPOA_Q(self):
        save_measure = 'GPOA2_Q'
        # get profit
        raw_profit = self.influx.getDataMultiprocess('FinancialReport_Gus', 'tot_profit_Q', self.start, self.end)
        raw_profit.index.names = ['date']
        raw_profit.reset_index(inplace=True)
        # --------------------------------------------------------------------------------------
        # 计算 cur GPOA_Q
        profit_df = raw_profit.loc[:, ['date', 'code', 'tot_profit_Q', 'report_period']].copy()
        assets_df = self.tot_assets.loc[:, ['date', 'code', 'tot_assets', 'tot_assets_last1Q',
                                            'report_period']].copy()
        GPOA_df = pd.merge(profit_df, assets_df, how='outer', on=['date', 'code', 'report_period'])
        GPOA_df = GPOA_df.sort_values(['date', 'code', 'report_period'])
        codes = GPOA_df['code'].unique()
        split_codes = np.array_split(codes, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            res = Parallel()(delayed(GPOA2.JOB_cur_GPOA_Q)
                             (codes, GPOA_df, self.db, save_measure) for codes in split_codes)
        print('cur GPOA_Q finish')
        print('-' * 30)
        for r in res:
            self.fail_list.extend(r)
        # --------------------------------------------------------------------------------------
        # 计算 history GPOA_Q
        profit_df = raw_profit.copy()
        for i in range(1, 13):
            cur_rps = []
            former_rps = []
            for rp in profit_df['report_period'].unique():
                cur_rps.append(rp)
                former_rps.append(GPOA2.get_lastQ_RP(rp, i))
            rp_dict = dict(zip(cur_rps, former_rps))
            profit_df['profit_last{0}Q_rp'.format(i)] = profit_df['report_period'].map(rp_dict)
        profit_df.drop('report_period', axis=1, inplace=True)
        assets_df = self.tot_assets.copy()
        assets_df.rename(columns={'tot_assets': 'tot_assets_last0Q'}, inplace=True)
        for i in range(0, 13):
            cur_rps = []
            former_rps = []
            for rp in assets_df['report_period'].unique():
                cur_rps.append(rp)
                former_rps.append(GPOA2.get_lastQ_RP(rp, i))
            rp_dict = dict(zip(cur_rps, former_rps))
            assets_df['assets_last{0}Q_rp'.format(i)] = assets_df['report_period'].map(rp_dict)
        assets_df.drop('report_period', axis=1, inplace=True)
        GPOA_df = pd.merge(profit_df, assets_df, how='outer', on=['date', 'code'])
        GPOA_df = GPOA_df.sort_values(['date', 'code'])
        codes = GPOA_df['code'].unique()
        split_codes = np.array_split(codes, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            res = Parallel()(delayed(GPOA2.JOB_hist_GPOA_Q)
                             (codes, GPOA_df, self.db, save_measure) for codes in split_codes)
        print('hist GPOA_Q finish')
        print('-' * 30)
        for r in res:
            self.fail_list.extend(r)


    def cal_GPOA_TTM(self):
        save_measure = 'GPOA2'
        # get profit
        raw_profit = self.influx.getDataMultiprocess('FinancialReport_Gus', 'tot_profit_TTM', self.start, self.end)
        raw_profit.index.names = ['date']
        raw_profit.reset_index(inplace=True)
        # --------------------------------------------------------------------------------------
        # 计算 cur GPOA
        profit_df = raw_profit.loc[:, ['date', 'code', 'tot_profit_TTM', 'report_period']].copy()
        cur_rps = []
        former_rps = []
        for rp in profit_df['report_period'].unique():
            cur_rps.append(rp)
            former_rps.append(GPOA2.get_lastQ_RP(rp, 4))
        rp_dict = dict(zip(cur_rps, former_rps))
        profit_df['profit_last4Q_rp'] = profit_df['report_period'].map(rp_dict)
        assets_df = self.tot_assets.loc[:, ['date', 'code', 'tot_assets', 'tot_assets_last3Q',
                                            'tot_assets_last4Q', 'report_period']].copy()
        for i in [3, 4]:
            cur_rps = []
            former_rps = []
            for rp in assets_df['report_period'].unique():
                cur_rps.append(rp)
                former_rps.append(GPOA2.get_lastQ_RP(rp, i))
            rp_dict = dict(zip(cur_rps, former_rps))
            assets_df['assets_last{0}Q_rp'.format(i)] = assets_df['report_period'].map(rp_dict)
        assets_df.drop('report_period', axis=1, inplace=True)
        GPOA_df = pd.merge(profit_df, assets_df, how='outer', on=['date', 'code'])
        GPOA_df = GPOA_df.sort_values(['date', 'code', 'report_period'])
        codes = GPOA_df['code'].unique()
        split_codes = np.array_split(codes, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            res = Parallel()(delayed(GPOA2.JOB_cur_GPOA_TTM)
                             (codes, GPOA_df, self.db, save_measure) for codes in split_codes)
        print('cur GPOA_TTM finish')
        print('-' * 30)
        for r in res:
            self.fail_list.extend(r)
        # --------------------------------------------------------------------------------------
        # 计算 history GPOA
        profit_df = raw_profit.copy()
        for i in range(1, 13):
            cur_rps = []
            former_rps = []
            for rp in profit_df['report_period'].unique():
                cur_rps.append(rp)
                former_rps.append(GPOA2.get_lastQ_RP(rp, i))
            rp_dict = dict(zip(cur_rps, former_rps))
            profit_df['profit_last{0}Q_rp'.format(i)] = profit_df['report_period'].map(rp_dict)
        profit_df.drop('report_period', axis=1, inplace=True)
        assets_df = self.tot_assets.copy()
        assets_df.rename(columns={'tot_assets': 'tot_assets_last0Q'}, inplace=True)
        for i in range(0, 13):
            cur_rps = []
            former_rps = []
            for rp in assets_df['report_period'].unique():
                cur_rps.append(rp)
                former_rps.append(GPOA2.get_lastQ_RP(rp, i))
            rp_dict = dict(zip(cur_rps, former_rps))
            assets_df['assets_last{0}Q_rp'.format(i)] = assets_df['report_period'].map(rp_dict)
        assets_df.drop('report_period', axis=1, inplace=True)
        GPOA_df = pd.merge(profit_df, assets_df, how='outer', on=['date', 'code'])
        GPOA_df = GPOA_df.sort_values(['date', 'code'])
        codes = GPOA_df['code'].unique()
        split_codes = np.array_split(codes, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            res = Parallel()(delayed(GPOA2.JOB_hist_GPOA_TTM)
                             (codes, GPOA_df, self.db, save_measure) for codes in split_codes)
        print('hist GPOA_TTM finish')
        print('-' * 30)
        for r in res:
            self.fail_list.extend(r)


    def cal_factors(self, start, end, n_jobs):
        pd.set_option('mode.use_inf_as_na', True)
        self.start = start
        self.end = end
        self.n_jobs = n_jobs
        self.fail_list = []
        # get tot assets
        self.tot_assets = \
            self.influx.getDataMultiprocess('FinancialReport_Gus', 'tot_assets', start, end)
        self.tot_assets.index.names = ['date']
        self.tot_assets.reset_index(inplace=True)
        self.cal_GPOA_Q()
        self.cal_GPOA_TTM()

        return self.fail_list


if __name__ == '__main__':
    GPOA = GPOA2()
    r = GPOA.cal_factors(20090101, 20200525, N_JOBS)
    print('task finish')
    print(r)
    print(datetime.datetime.now())