# 盈利能力因子 ROE,ROE_Q,ROE_ddt,ROE_ddt_Q的计算

from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
import datetime
from global_constant import N_JOBS
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData


class ROE_series(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'ROE'

    @staticmethod
    def JOB_factors(codes, df, cur_net_equity_field, pre_net_equity_field, net_profit_field, result_field, db, measure):
        pd.set_option('mode.use_inf_as_na', True)
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_df[[cur_net_equity_field, pre_net_equity_field]] = \
                code_df[[cur_net_equity_field, pre_net_equity_field]].fillna(method='ffill', axis=1)
            code_df[result_field] = \
                code_df[net_profit_field] / (code_df[cur_net_equity_field] + code_df[pre_net_equity_field]) * 2
            code_df.set_index('date', inplace=True)
            code_df = code_df.loc[:, ['code', result_field]]
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

    def cal_ROE(self, profit_field, cur_net_equity_field, pre_net_equity_field, result_field):
        profit_df = self.raw_profit_data.loc[:, ['date', 'code', profit_field]].copy()
        ROE_df = pd.merge(self.net_equity.loc[:, ['date', 'code', cur_net_equity_field, pre_net_equity_field]],
                          profit_df, on=['date', 'code'])
        codes = ROE_df['code'].unique()
        split_codes = np.array_split(codes, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            res = Parallel()(delayed(ROE_series.JOB_factors)
                             (codes, ROE_df, cur_net_equity_field, pre_net_equity_field, profit_field, result_field,
                              self.db, self.measure) for codes in split_codes)
        print('%s finish' %result_field)
        print('-' * 30)
        for r in res:
            self.fail_list.extend(r)

    def cal_factors(self, start, end, n_jobs):
        pd.set_option('mode.use_inf_as_na', True)
        self.n_jobs = n_jobs
        self.net_equity = \
            self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_equity', start, end,
                                            ['code', 'net_equity', 'net_equity_last1Q', 'net_equity_last2Q',
                                             'net_equity_lastY', 'net_equity_last5Q'])
        self.net_equity.index.names = ['date']
        self.net_equity.reset_index(inplace=True)
        self.fail_list = []
        # *****************************************************************************
        self.raw_profit_data = \
            self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_profit_Q', start, end,
                                            ['code', 'net_profit_Q', 'net_profit_Q_last1Q', 'net_profit_Q_lastY'])
        self.raw_profit_data.index.names = ['date']
        self.raw_profit_data.reset_index(inplace=True)
        # -----------------------------------ROE_Q-------------------------------------
        self.cal_ROE('net_profit_Q', 'net_equity', 'net_equity_last1Q', 'ROE_Q')
        # ------------------------------ROE_Q_last1Q-----------------------------------
        self.cal_ROE('net_profit_Q_last1Q', 'net_equity_last1Q', 'net_equity_last2Q', 'ROE_Q_last1Q')
        # ------------------------------ROE_Q_lastY------------------------------------
        self.cal_ROE('net_profit_Q_lastY', 'net_equity_lastY', 'net_equity_last5Q', 'ROE_Q_lastY')
        # *****************************************************************************
        self.raw_profit_data = \
            self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_profit_TTM', start, end,
                                            ['code', 'net_profit_TTM', 'net_profit_TTM_last1Q'])
        # -----------------------------------ROE---------------------------------------
        self.cal_ROE('net_profit_TTM', 'net_equity', 'net_equity_lastY', 'ROE')
        # ------------------------------ROE_last1Q-------------------------------------
        self.cal_ROE('net_profit_TTM_last1Q', 'net_equity_last1Q', 'net_equity_last5Q', 'ROE_last1Q')
        # *****************************************************************************
        self.raw_profit_data = \
            self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_profit_ddt_Q', start, end,
                                            ['code', 'net_profit_ddt_Q', 'net_profit_ddt_Q_last1Q',
                                             'net_profit_ddt_Q_lastY'])
        # ------------------------------ROE_ddt_Q--------------------------------------
        self.cal_ROE('net_profit_ddt_Q', 'net_equity', 'net_equity_last1Q', 'ROE_ddt_Q')
        # ---------------------------ROE_ddt_Q_last1Q----------------------------------
        self.cal_ROE('net_profit_ddt_Q_last1Q', 'net_equity_last1Q', 'net_equity_last2Q', 'ROE_ddt_Q_last1Q')
        # ---------------------------ROE_ddt_Q_lastY-----------------------------------
        self.cal_ROE('net_profit_ddt_Q_lastY', 'net_equity_lastY', 'net_equity_last5Q', 'ROE_ddt_Q_lastY')
        # *****************************************************************************
        self.raw_profit_data = \
            self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_profit_ddt_TTM', start, end,
                                            ['code', 'net_profit_ddt_TTM', 'net_profit_ddt_TTM_last1Q'])
        # --------------------------------ROE_ddt--------------------------------------
        self.cal_ROE('net_profit_ddt_TTM', 'net_equity', 'net_equity_lastY', 'ROE_ddt')
        # ----------------------------ROE_ddt_last1Q-----------------------------------
        self.cal_ROE('net_profit_ddt_TTM_last1Q', 'net_equity_last1Q', 'net_equity_last5Q', 'ROE_ddt_last1Q')

        return self.fail_list


if __name__ == '__main__':
    roe = ROE_series()
    r = roe.cal_factors(20100101, 20200205, N_JOBS)
    print('task finish')
    print(r)
    print(datetime.datetime.now())