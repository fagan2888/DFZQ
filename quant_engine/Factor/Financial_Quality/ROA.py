# 盈利能力因子 ROA, ROA_Q 的计算

from factor_base import FactorBase
import pandas as pd
import numpy as np
import datetime
from global_constant import N_JOBS
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData


class ROA_series(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'ROA'

    @staticmethod
    def JOB_factors(codes, df, cur_tot_assets_field, pre_tot_assets_field, net_profit_field, result_field, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_df[[cur_tot_assets_field, pre_tot_assets_field]] = \
                code_df[[cur_tot_assets_field, pre_tot_assets_field]].fillna(method='ffill', axis=1)
            code_df[result_field] = \
                code_df[net_profit_field] / (code_df[cur_tot_assets_field] + code_df[pre_tot_assets_field]) * 2
            code_df.set_index('date', inplace=True)
            code_df = code_df.loc[:, ['code', result_field]]
            code_df = code_df.replace(np.inf, np.nan)
            code_df = code_df.replace(-np.inf, np.nan)
            code_df = code_df.dropna(subset=[result_field])
            print('code: %s' % code)
            if code_df.empty:
                continue
            r = influx.saveData(code_df, db, measure)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('%s Error: %s' % (result_field, r))
        return save_res

    def cal_ROA(self, profit_field, cur_tot_assets_field, pre_tot_assets_field, result_field):
        profit_df = self.raw_profit_data.loc[:, ['date', 'code', profit_field]].copy()
        ROA_df = pd.merge(self.tot_assets.loc[:, ['date', 'code', cur_tot_assets_field, pre_tot_assets_field]],
                          profit_df, on=['date', 'code'])
        codes = ROA_df['code'].unique()
        split_codes = np.array_split(codes, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            res = Parallel()(delayed(ROA_series.JOB_factors)
                             (codes, ROA_df, cur_tot_assets_field, pre_tot_assets_field, profit_field, result_field,
                              self.db, self.measure) for codes in split_codes)
        print('%s finish' %result_field)
        print('-' * 30)
        for r in res:
            self.fail_list.extend(r)

    def cal_factors(self, start, end, n_jobs):
        pd.set_option('mode.use_inf_as_na', True)
        self.n_jobs = n_jobs
        self.tot_assets = \
            self.influx.getDataMultiprocess('FinancialReport_Gus', 'tot_assets', start, end,
                                            ['code', 'tot_assets', 'tot_assets_last1Q', 'tot_assets_last2Q',
                                             'tot_assets_lastY', 'tot_assets_last5Q'])
        self.tot_assets.index.names = ['date']
        self.tot_assets.reset_index(inplace=True)
        self.fail_list = []
        # *****************************************************************************
        self.raw_profit_data = \
            self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_profit_Q', start, end,
                                            ['code', 'net_profit_Q', 'net_profit_Q_last1Q', 'net_profit_Q_lastY'])
        self.raw_profit_data.index.names = ['date']
        self.raw_profit_data.reset_index(inplace=True)
        # -----------------------------------ROA_Q-------------------------------------
        self.cal_ROA('net_profit_Q', 'tot_assets', 'tot_assets_last1Q', 'ROA_Q')
        # ------------------------------ROA_Q_last1Q-----------------------------------
        self.cal_ROA('net_profit_Q_last1Q', 'tot_assets_last1Q', 'tot_assets_last2Q', 'ROA_Q_last1Q')
        # ------------------------------ROA_Q_lastY------------------------------------
        self.cal_ROA('net_profit_Q_lastY', 'tot_assets_lastY', 'tot_assets_last5Q', 'ROA_Q_lastY')
        # *****************************************************************************
        self.raw_profit_data = \
            self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_profit_TTM', start, end,
                                            ['code', 'net_profit_TTM', 'net_profit_TTM_last1Q'])
        self.raw_profit_data.index.names = ['date']
        self.raw_profit_data.reset_index(inplace=True)
        # -----------------------------------ROA---------------------------------------
        self.cal_ROA('net_profit_TTM', 'tot_assets', 'tot_assets_lastY', 'ROA')
        # ------------------------------ROA_last1Q-------------------------------------
        self.cal_ROA('net_profit_TTM_last1Q', 'tot_assets_last1Q', 'tot_assets_last5Q', 'ROA_last1Q')

        return self.fail_list


if __name__ == '__main__':
    ROA = ROA_series()
    r = ROA.cal_factors(20100101, 20200428, N_JOBS)
    print('task finish')
    print(r)
    print(datetime.datetime.now())