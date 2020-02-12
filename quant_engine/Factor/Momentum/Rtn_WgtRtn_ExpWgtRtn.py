#  动量因子 return_Xm, wgt_return_Xm, exp_wgt_return_Xm 的计算

from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
import dateutil.parser as dtparser
from joblib import Parallel, delayed, parallel_backend
from dateutil.relativedelta import relativedelta
import math
from global_constant import N_JOBS


class Rtn_WgtRtn_ExpWgtRtn(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'Momentum'

    @staticmethod
    def cal_Rtn_series(code_df, idx, n_months_list):
        first_dt = code_df.index[0]
        str_idx = idx.strftime('%Y%m%d')
        n_months_dict = {}
        for n_months in n_months_list:
            n_months_dict[n_months] = (idx - relativedelta(months=n_months)).strftime('%Y%m%d')
        res = {}
        for n_months in n_months_dict.keys():
            if (idx - first_dt).days < 30 * n_months:
                continue
            months_before_idx = n_months_dict[n_months]
            period_df = code_df.loc[months_before_idx:str_idx, :].copy()
            if not period_df.shape[0] < 10 * n_months:
                tmp_res = {}
                tmp_res['rtn_m'+str(n_months)] = \
                    (period_df['close'].iloc[-1] * period_df['adj_factor'].iloc[-1]) / (
                            period_df['preclose'].iloc[0] * period_df['adj_factor'].iloc[0]) - 1
                if period_df['turnover'].sum() == 0:
                    tmp_res['wgt_rtn_m'+str(n_months)] = 0
                    tmp_res['float_wgt_rtn_m'+str(n_months)] = 0
                    tmp_res['free_wgt_rtn_m'+str(n_months)] = 0
                    tmp_res['exp_wgt_rtn_m'+str(n_months)] = 0
                    tmp_res['float_exp_wgt_rtn_m'+str(n_months)] = 0
                    tmp_res['free_exp_wgt_rtn_m'+str(n_months)] = 0
                else:
                    period_df['n_days_to_idx'] = range(period_df.shape[0] - 1, -1, -1)
                    period_df['exp_wgt'] = period_df['n_days_to_idx'].apply(
                        lambda x: math.exp(x * -1 / n_months / 4))
                    tmp_res['wgt_rtn_m'+str(n_months)] = \
                        period_df['wgt_return'].sum() / period_df['turnover'].sum()
                    tmp_res['float_wgt_rtn_m'+str(n_months)] = \
                        period_df['float_wgt_return'].sum() / period_df['float_turnover'].sum()
                    tmp_res['free_wgt_rtn_m'+str(n_months)] = \
                        period_df['free_wgt_return'].sum() / period_df['free_turnover'].sum()
                    tmp_res['exp_wgt_rtn_m'+str(n_months)] = \
                        (period_df['wgt_return'] * period_df['exp_wgt']).sum() / (
                                period_df['turnover'] * period_df['exp_wgt']).sum()
                    tmp_res['float_exp_wgt_rtn_m'+str(n_months)] = \
                        (period_df['float_wgt_return'] * period_df['exp_wgt']).sum() / (
                                period_df['float_turnover'] * period_df['exp_wgt']).sum()
                    tmp_res['free_exp_wgt_rtn_m'+str(n_months)] = \
                        (period_df['free_wgt_return'] * period_df['exp_wgt']).sum() / (
                                period_df['free_turnover'] * period_df['exp_wgt']).sum()
                    res.update(tmp_res)
            else:
                continue
        return res


    @staticmethod
    def JOB_factors(codes, df, months_list, start, db, measure):
        pd.set_option('mode.use_inf_as_na', True)
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_df = code_df.sort_index()
            res_dict = {}
            for idx, row in code_df.iterrows():
                r = Rtn_WgtRtn_ExpWgtRtn.cal_Rtn_series(code_df, idx, months_list)
                if not r:
                    continue
                else:
                    res_dict[idx] = r
            if not res_dict:
                continue
            else:
                res_df = pd.DataFrame(res_dict).T
                res_df['code'] = code
                res_df = res_df.loc[str(start):, :]
                res_df = res_df.where(pd.notnull(res_df), None)
                print('code: %s' % code)
                r = influx.saveData(res_df, db, measure)
                if r == 'No error occurred...':
                    pass
                else:
                    save_res.append('%s Error: %s' % (measure, r))
        return save_res


    def cal_factors(self, start, end, months_list, n_jobs):
        data_start = (dtparser.parse(str(start)) - relativedelta(years=1)).strftime('%Y%m%d')
        daily_data = self.influx.getDataMultiprocess('DailyData_Gus', 'marketData', data_start, end,
                                                     ['code', 'adj_factor', 'close', 'preclose'])
        daily_data.index.names = ['date']
        daily_data.reset_index(inplace=True)
        turnover = self.influx.getDataMultiprocess('DailyData_Gus', 'indicators', data_start, end,
                                                   ['code', 'turnover', 'float_turnover', 'free_turnover'])
        turnover.index.names = ['date']
        turnover.reset_index(inplace=True)
        merge = pd.merge(daily_data, turnover, on=['date', 'code'])
        merge['return'] = merge['close'] / merge['preclose'] - 1
        merge['wgt_return'] = merge['turnover'] * merge['return']
        merge['float_wgt_return'] = merge['float_turnover'] * merge['return']
        merge['free_wgt_return'] = merge['free_turnover'] * merge['return']
        merge.set_index('date', inplace=True)
        codes = merge['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(Rtn_WgtRtn_ExpWgtRtn.JOB_factors)
                             (codes, merge, months_list, start, self.db, self.measure) for codes in split_codes)
        print('Momentum finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list

if __name__ == '__main__':
    i = Rtn_WgtRtn_ExpWgtRtn()
    r = i.cal_factors(20100101, 20200210, [1,3,6], N_JOBS)
    print(r)
