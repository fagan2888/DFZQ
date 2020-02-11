#  动量因子 return_Xm, wgt_return_Xm, exp_wgt_return_Xm 的计算

from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
import dateutil.parser as dtparser
from joblib import Parallel, delayed, parallel_backend
from dateutil.relativedelta import relativedelta
from global_constant import N_JOBS

class Turn_StdTurn_BiasStdTurn(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'Daily_Factors'
        self.measure = 'Turnover'

    @staticmethod
    def JOB_factors(codes, df, start, db, measure):
        pd.set_option('mode.use_inf_as_na', True)
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_df = code_df.sort_index()
            first_dt = code_df.index[0]
            res_dict = {}
            for idx, row in code_df.iterrows():
                str_idx = idx.strftime('%Y%m%d')
                m1_before_idx = (idx - relativedelta(months=1)).strftime('%Y%m%d')
                m3_before_idx = (idx - relativedelta(months=3)).strftime('%Y%m%d')
                y2_before_idx = (idx - relativedelta(years=2)).strftime('%Y%m%d')
                # ------------------------------------------------------------------------------------------
                if (idx - first_dt).days < 30:
                    continue
                period_m1_df = code_df.loc[m1_before_idx:str_idx, :].copy()
                if not period_m1_df.shape[0] < 10:
                    res_dict[idx] = {}
                    res_dict[idx]['turn_1m'] = period_m1_df['turnover'].mean()
                    res_dict[idx]['float_turn_1m'] = period_m1_df['float_turnover'].mean()
                    res_dict[idx]['free_turn_1m'] = period_m1_df['free_turnover'].mean()
                    res_dict[idx]['std_turn_1m'] = period_m1_df['turnover'].std()
                    res_dict[idx]['std_float_turn_1m'] = period_m1_df['float_turnover'].std()
                    res_dict[idx]['std_free_turn_1m'] = period_m1_df['free_turnover'].std()
                else:
                    continue
                # -----------------------------------------------------------------------------------------
                if (idx - first_dt).days < 90:
                    continue
                period_m3_df = code_df.loc[m3_before_idx:str_idx, :].copy()
                if not period_m3_df.shape[0] < 30:
                    res_dict[idx]['turn_3m'] = period_m3_df['turnover'].mean()
                    res_dict[idx]['float_turn_3m'] = period_m3_df['float_turnover'].mean()
                    res_dict[idx]['free_turn_3m'] = period_m3_df['free_turnover'].mean()
                    res_dict[idx]['std_turn_3m'] = period_m3_df['turnover'].std()
                    res_dict[idx]['std_float_turn_3m'] = period_m3_df['float_turnover'].std()
                    res_dict[idx]['std_free_turn_3m'] = period_m3_df['free_turnover'].std()
                else:
                    continue
                # ---------------------------------------------------------------------------------------
                if (idx - first_dt).days < 700:
                    continue
                period_y2_df = code_df.loc[y2_before_idx:str_idx, :].copy()
                # 股票长期停牌的情况 eg. 000029.SZ
                if period_y2_df.shape[0] < 200:
                    continue
                else:
                    turn_2y = period_y2_df['turnover'].mean()
                    if turn_2y == 0:
                        continue
                    else:
                        float_turn_2y = period_y2_df['float_turnover'].mean()
                        free_turn_2y = period_y2_df['free_turnover'].mean()
                        std_turn_2y = period_y2_df['turnover'].std()
                        std_float_turn_2y = period_y2_df['float_turnover'].std()
                        std_free_turn_2y = period_y2_df['free_turnover'].std()
                        # -------------------------------------------------------------------
                        res_dict[idx]['bias_turn_1m'] = \
                            res_dict[idx]['turn_1m'] / turn_2y - 1
                        res_dict[idx]['bias_float_turn_1m'] = \
                            res_dict[idx]['float_turn_1m'] / float_turn_2y - 1
                        res_dict[idx]['bias_free_turn_1m'] = \
                            res_dict[idx]['free_turn_1m'] / free_turn_2y - 1
                        res_dict[idx]['bias_std_turn_1m'] = \
                            res_dict[idx]['std_turn_1m'] / std_turn_2y - 1
                        res_dict[idx]['bias_std_float_turn_1m'] = \
                            res_dict[idx]['std_float_turn_1m'] / std_float_turn_2y - 1
                        res_dict[idx]['bias_std_free_turn_1m'] = \
                            res_dict[idx]['std_free_turn_1m'] / std_free_turn_2y - 1
                        res_dict[idx]['bias_turn_3m'] = \
                            res_dict[idx]['turn_3m'] / turn_2y - 1
                        res_dict[idx]['bias_float_turn_3m'] = \
                            res_dict[idx]['float_turn_3m'] / float_turn_2y - 1
                        res_dict[idx]['bias_free_turn_3m'] = \
                            res_dict[idx]['free_turn_3m'] / free_turn_2y - 1
                        res_dict[idx]['bias_std_turn_3m'] = \
                            res_dict[idx]['std_turn_3m'] / std_turn_2y - 1
                        res_dict[idx]['bias_std_float_turn_3m'] = \
                            res_dict[idx]['std_float_turn_3m'] / std_float_turn_2y - 1
                        res_dict[idx]['bias_std_free_turn_3m'] = \
                            res_dict[idx]['std_free_turn_3m'] / std_free_turn_2y - 1
            if not res_dict:
                continue
            else:
                res_df = pd.DataFrame(res_dict).T
                res_df = res_df.loc[str(start):, ]
                res_df = res_df.where(pd.notnull(res_df), None)
                res_df['code'] = code
                # save
                print('code: %s' % code)
                r = influx.saveData(res_df, db, measure)
                if r == 'No error occurred...':
                    pass
                else:
                    save_res.append('%s Error: %s' % ('Turnover', r))
        return save_res


    def cal_factors(self, start, end, n_jobs):
        data_start = (dtparser.parse(str(start)) - relativedelta(years=2)).strftime('%Y%m%d')
        turnover = self.influx.getDataMultiprocess('DailyData_Gus', 'indicators', data_start, end,
                                                   ['code', 'turnover', 'float_turnover', 'free_turnover'])
        codes = turnover['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(Turn_StdTurn_BiasStdTurn.JOB_factors)
                             (codes, turnover, start, self.db, self.measure) for codes in split_codes)
        print('Turnover finish' )
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    i = Turn_StdTurn_BiasStdTurn()
    r = i.cal_factors(20171002, 20191009, N_JOBS)
    print(r)
