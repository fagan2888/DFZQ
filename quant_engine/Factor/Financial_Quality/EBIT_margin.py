# 盈利能力因子 EBIT_margin 的计算
# 对齐 report period

from factor_base import FactorBase
import pandas as pd
import numpy as np
import datetime
from global_constant import N_JOBS
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData
from data_process import DataProcess


class EBITMargin(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'

    @staticmethod
    def JOB_factor(codes, df, EBIT_field, rev_field, result_field, n_Qs, db, measure):
        influx = influxdbData()
        save_res = []
        for code in codes:
            cols = []
            code_df = df.loc[df['code'] == code, :].copy()
            for i in range(n_Qs):
                cols.append('{0}_last{1}Q'.format(result_field, i))
                # 由于业绩预告和业绩快报的存在，oper_rev 会领先于 EBIT
                conditions = [code_df['EBIT_last{0}Q_rp'.format(i)].values ==
                              code_df['rev_last{0}Q_rp'.format(i)].values,
                              code_df['EBIT_last{0}Q_rp'.format(i)].values ==
                              code_df['rev_last{0}Q_rp'.format(i + 1)].values,
                              code_df['EBIT_last{0}Q_rp'.format(i)].values ==
                              code_df['rev_last{0}Q_rp'.format(i + 2)].values]
                choices = [code_df['{0}_last{1}Q'.format(EBIT_field, i)].values /
                           code_df['{0}_last{1}Q'.format(rev_field, i)].values,
                           code_df['{0}_last{1}Q'.format(EBIT_field, i)].values /
                           code_df['{0}_last{1}Q'.format(rev_field, i + 1)].values,
                           code_df['{0}_last{1}Q'.format(EBIT_field, i)].values /
                           code_df['{0}_last{1}Q'.format(rev_field, i + 2)].values]
                code_df['{0}_last{1}Q'.format(result_field, i)] = np.select(conditions, choices, default=np.nan)
            code_df = code_df.replace(np.inf, np.nan)
            code_df = code_df.replace(-np.inf, np.nan)
            code_df = code_df.loc[np.any(pd.notnull(code_df[cols]), axis=1), ['code', 'report_period'] + cols]
            code_df.rename(columns={'{0}_last0Q'.format(result_field): result_field}, inplace=True)
            code_df = code_df.where(pd.notnull(code_df), None)
            print('code: %s' % code)
            r = influx.saveData(code_df, db, measure)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('%s Error: %s' % (result_field, r))
        return save_res

    def cal_EBITmargin_Q(self):
        # 计算 EBITmargin_Q
        save_measure = 'EBITmargin_Q'
        # get ebit
        ebit = self.influx.getDataMultiprocess('FinancialReport_Gus', 'EBIT_Q', self.start, self.end)
        ebit.index.names = ['date']
        ebit.reset_index(inplace=True)
        for i in range(12):
            cur_rps = []
            former_rps = []
            for rp in ebit['report_period'].unique():
                cur_rps.append(rp)
                former_rps.append(DataProcess.get_former_RP(rp, i))
            rp_dict = dict(zip(cur_rps, former_rps))
            ebit['EBIT_last{0}Q_rp'.format(i)] = ebit['report_period'].map(rp_dict)
        ebit.rename(columns={'EBIT_Q': 'EBIT_Q_last0Q'}, inplace=True)
        # get oper_rev
        oper_rev = self.influx.getDataMultiprocess('FinancialReport_Gus', 'oper_rev_Q', self.start, self.end)
        oper_rev.index.names = ['date']
        oper_rev.reset_index(inplace=True)
        for i in range(12):
            cur_rps = []
            former_rps = []
            for rp in oper_rev['report_period'].unique():
                cur_rps.append(rp)
                former_rps.append(DataProcess.get_former_RP(rp, i))
            rp_dict = dict(zip(cur_rps, former_rps))
            oper_rev['rev_last{0}Q_rp'.format(i)] = oper_rev['report_period'].map(rp_dict)
        oper_rev.drop('report_period', axis=1, inplace=True)
        oper_rev.rename(columns={'oper_rev_Q': 'oper_rev_Q_last0Q'}, inplace=True)
        em_df = pd.merge(ebit, oper_rev, how='outer', on=['date', 'code'])
        em_df = em_df.sort_values(['date', 'code', 'report_period'])
        em_df.set_index('date', inplace=True)
        codes = em_df['code'].unique()
        split_codes = np.array_split(codes, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            res = Parallel()(delayed(EBITMargin.JOB_factor)
                             (codes, em_df, 'EBIT_Q', 'oper_rev_Q', 'EBITmargin_Q', 10, self.db, save_measure)
                             for codes in split_codes)
        print('EBITMarginQ finish')
        print('-' * 30)
        for r in res:
            self.fail_list.extend(r)

    def cal_EBITmargin_TTM(self):
        # 计算 EBITmargin_TTM
        save_measure = 'EBITmargin_TTM'
        # get ebit
        ebit = self.influx.getDataMultiprocess('FinancialReport_Gus', 'EBIT_TTM', self.start, self.end)
        ebit.index.names = ['date']
        ebit.reset_index(inplace=True)
        for i in range(9):
            cur_rps = []
            former_rps = []
            for rp in ebit['report_period'].unique():
                cur_rps.append(rp)
                former_rps.append(DataProcess.get_former_RP(rp, i))
            rp_dict = dict(zip(cur_rps, former_rps))
            ebit['EBIT_last{0}Q_rp'.format(i)] = ebit['report_period'].map(rp_dict)
        ebit.rename(columns={'EBIT_TTM': 'EBIT_TTM_last0Q'}, inplace=True)
        # get oper_rev
        oper_rev = self.influx.getDataMultiprocess('FinancialReport_Gus', 'oper_rev_TTM', self.start, self.end)
        oper_rev.index.names = ['date']
        oper_rev.reset_index(inplace=True)
        for i in range(9):
            cur_rps = []
            former_rps = []
            for rp in oper_rev['report_period'].unique():
                cur_rps.append(rp)
                former_rps.append(DataProcess.get_former_RP(rp, i))
            rp_dict = dict(zip(cur_rps, former_rps))
            oper_rev['rev_last{0}Q_rp'.format(i)] = oper_rev['report_period'].map(rp_dict)
        oper_rev.drop('report_period', axis=1, inplace=True)
        oper_rev.rename(columns={'oper_rev_TTM': 'oper_rev_TTM_last0Q'}, inplace=True)
        em_df = pd.merge(ebit, oper_rev, how='outer', on=['date', 'code'])
        em_df = em_df.sort_values(['date', 'code', 'report_period'])
        em_df.set_index('date', inplace=True)
        codes = em_df['code'].unique()
        split_codes = np.array_split(codes, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            res = Parallel()(delayed(EBITMargin.JOB_factor)
                             (codes, em_df, 'EBIT_TTM', 'oper_rev_TTM', 'EBITmargin_TTM', 7, self.db, save_measure)
                             for codes in split_codes)
        print('EBITMarginTTM finish')
        print('-' * 30)
        for r in res:
            self.fail_list.extend(r)

    def cal_EBITmargin_latest(self):
        # 计算 EBITmargin_Q
        save_measure = 'EBITmargin_latest'
        # get ebit
        ebit = self.influx.getDataMultiprocess('FinancialReport_Gus', 'EBIT', self.start, self.end)
        ebit.index.names = ['date']
        ebit.reset_index(inplace=True)
        for i in range(13):
            cur_rps = []
            former_rps = []
            for rp in ebit['report_period'].unique():
                cur_rps.append(rp)
                former_rps.append(DataProcess.get_former_RP(rp, i))
            rp_dict = dict(zip(cur_rps, former_rps))
            ebit['EBIT_last{0}Q_rp'.format(i)] = ebit['report_period'].map(rp_dict)
        ebit.rename(columns={'EBIT': 'EBIT_last0Q'}, inplace=True)
        # get oper_rev
        oper_rev = self.influx.getDataMultiprocess('FinancialReport_Gus', 'oper_rev', self.start, self.end)
        oper_rev.index.names = ['date']
        oper_rev.reset_index(inplace=True)
        for i in range(13):
            cur_rps = []
            former_rps = []
            for rp in oper_rev['report_period'].unique():
                cur_rps.append(rp)
                former_rps.append(DataProcess.get_former_RP(rp, i))
            rp_dict = dict(zip(cur_rps, former_rps))
            oper_rev['rev_last{0}Q_rp'.format(i)] = oper_rev['report_period'].map(rp_dict)
        oper_rev.drop('report_period', axis=1, inplace=True)
        oper_rev.rename(columns={'oper_rev': 'oper_rev_last0Q'}, inplace=True)
        em_df = pd.merge(ebit, oper_rev, how='outer', on=['date', 'code'])
        em_df = em_df.sort_values(['date', 'code', 'report_period'])
        em_df.set_index('date', inplace=True)
        codes = em_df['code'].unique()
        split_codes = np.array_split(codes, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            res = Parallel()(delayed(EBITMargin.JOB_factor)
                             (codes, em_df, 'EBIT', 'oper_rev', 'EBITmargin_latest', 11, self.db, save_measure)
                             for codes in split_codes)
        print('EBITMargin latest finish')
        print('-' * 30)
        for r in res:
            self.fail_list.extend(r)

    def cal_factors(self, start, end, n_jobs):
        pd.set_option('mode.use_inf_as_na', True)
        self.start = start
        self.end = end
        self.n_jobs = n_jobs
        self.fail_list = []

        self.cal_EBITmargin_Q()
        self.cal_EBITmargin_TTM()
        self.cal_EBITmargin_latest()

        return self.fail_list


if __name__ == '__main__':
    em = EBITMargin()
    r = em.cal_factors(20090101, 20200610, N_JOBS)
    print('task finish')
    print(r)
    print(datetime.datetime.now())