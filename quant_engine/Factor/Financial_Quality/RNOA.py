# 盈利能力因子 RONA, RNOA_Q的计算

from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
import datetime
from global_constant import N_JOBS
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData


class RNOA_series(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'RNOA'

    @staticmethod
    def JOB_factors(codes, df, cur_NOA_field, pre_NOA_field, oper_income_field, result_field, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_df[[cur_NOA_field, pre_NOA_field]] = \
                code_df[[cur_NOA_field, pre_NOA_field]].fillna(method='ffill', axis=1)
            code_df[result_field] = \
                code_df[oper_income_field] / (code_df[cur_NOA_field] + code_df[pre_NOA_field]) * 2
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

    def cal_RNOA(self, oper_income_field, cur_NOA_field, pre_NOA_field, result_field):
        oper_income_df = self.raw_oper_income_data.loc[:, ['date', 'code', oper_income_field]].copy()
        RNOA_df = pd.merge(self.NOA.loc[:, ['date', 'code', cur_NOA_field, pre_NOA_field]],
                          oper_income_df, on=['date', 'code'])
        codes = RNOA_df['code'].unique()
        split_codes = np.array_split(codes, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            res = Parallel()(delayed(RNOA_series.JOB_factors)
                             (codes, RNOA_df, cur_NOA_field, pre_NOA_field, oper_income_field, result_field,
                              self.db, self.measure) for codes in split_codes)
        print('%s finish' %result_field)
        print('-' * 30)
        for r in res:
            self.fail_list.extend(r)

    def cal_factors(self, start, end, n_jobs):
        pd.set_option('mode.use_inf_as_na', True)
        self.n_jobs = n_jobs
        self.NOA = \
            self.influx.getDataMultiprocess('FinancialReport_Gus', 'NOA', start, end,
                                            ['code', 'NOA', 'NOA_last1Q', 'NOA_last2Q',
                                             'NOA_lastY', 'NOA_last5Q'])
        self.NOA.index.names = ['date']
        self.NOA.reset_index(inplace=True)
        self.fail_list = []
        # *****************************************************************************
        self.raw_oper_income_data = \
            self.influx.getDataMultiprocess('FinancialReport_Gus', 'oper_income_Q', start, end,
                                            ['code', 'oper_income_Q', 'oper_income_Q_last1Q', 'oper_income_Q_lastY'])
        self.raw_oper_income_data.index.names = ['date']
        self.raw_oper_income_data.reset_index(inplace=True)
        # -----------------------------------RNOA_Q-------------------------------------
        self.cal_RNOA('oper_income_Q', 'NOA', 'NOA_last1Q', 'RNOA_Q')
        # ------------------------------RNOA_Q_last1Q-----------------------------------
        self.cal_RNOA('oper_income_Q_last1Q', 'NOA_last1Q', 'NOA_last2Q', 'RNOA_Q_last1Q')
        # ------------------------------RNOA_Q_lastY------------------------------------
        self.cal_RNOA('oper_income_Q_lastY', 'NOA_lastY', 'NOA_last5Q', 'RNOA_Q_lastY')
        # *****************************************************************************
        self.raw_oper_income_data = \
            self.influx.getDataMultiprocess('FinancialReport_Gus', 'oper_income_TTM', start, end,
                                            ['code', 'oper_income_TTM', 'oper_income_TTM_last1Q'])
        self.raw_oper_income_data.index.names = ['date']
        self.raw_oper_income_data.reset_index(inplace=True)
        # -----------------------------------RNOA---------------------------------------
        self.cal_RNOA('oper_income_TTM', 'NOA', 'NOA_lastY', 'RNOA')
        # ------------------------------RNOA_last1Q-------------------------------------
        self.cal_RNOA('oper_income_TTM_last1Q', 'NOA_last1Q', 'NOA_last5Q', 'RNOA_last1Q')
        return self.fail_list


if __name__ == '__main__':
    RNOA = RNOA_series()
    r = RNOA.cal_factors(20100101, 20200315, N_JOBS)
    print('task finish')
    print(r)
    print(datetime.datetime.now())