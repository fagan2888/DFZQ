from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
import datetime
from influxdb_data import influxdbData
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS


class Surprise(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'Surprise'

    @staticmethod
    def JOB_factors(codes, df, factor_field, save_db, save_measure):
        pd.set_option('mode.use_inf_as_na', True)
        influx = influxdbData()
        save_res = []
        pairs = []
        for i in range(1, 20):
            later_field = factor_field + '_last{0}Q'.format(str(i))
            former_field = factor_field + '_last{0}Q'.format(str(i+4))
            if former_field not in df.columns:
                break
            else:
                pairs.append([later_field, former_field])
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_and_report_period = code_df.loc[:, ['code', 'report_period']].copy()
            code_df = code_df.drop_duplicates()
            res_with_driff = []
            res_without_driff = []
            for idx, row in code_df.iterrows():
                n = 0
                driff_sum = 0
                for later_field, former_field in pairs:
                    if pd.notnull(row[later_field]) & pd.notnull(row[former_field]):
                        n += 1
                        driff_sum += row[later_field] - row[former_field]
                    # 最多用12期数据
                    if n == 6:
                        break
                # 不到8期数据无法预测
                if n < 4:
                    res_with_driff.append(np.nan)
                    res_without_driff.append(np.nan)
                    continue
                else:
                    driff = driff_sum / n
                # 随机游走模型预测
                n = 0
                diff_without_driff = []
                diff_with_driff = []
                for later_field, former_field in pairs:
                    if pd.notnull(row[later_field]) & pd.notnull(row[former_field]):
                        n += 1
                        diff_without_driff.append(row[later_field] - row[former_field])
                        diff_with_driff.append(row[later_field] + driff - row[former_field])
                std_with_driff = np.std(diff_with_driff)
                std_without_driff = np.std(diff_without_driff)
                # std等于0时通常为公司上市前只公布年报，导致前3季度计算得季度利润为0，第四季度为当年利润-上一年利润
                # 这种情况滤除
                if std_with_driff == 0:
                    res_with_driff.append(np.nan)
                    res_without_driff.append(np.nan)
                else:
                    res_with_driff.append((row[factor_field] - (row[factor_field + '_last4Q'] + driff)) / std_with_driff)
                    res_without_driff.append((row[factor_field] - row[factor_field + '_last4Q']) / std_without_driff)
            res_field_WD = 'sur_{0}_WD'.format(factor_field)
            res_field_WOD = 'sur_{0}_WOD'.format(factor_field)
            code_df[res_field_WD] = res_with_driff
            code_df[res_field_WOD] = res_without_driff
            code_df = code_df.loc[:, [res_field_WD, res_field_WOD]]
            code_df = pd.merge(code_and_report_period, code_df, right_index=True, left_index=True, how='left')
            code_df = code_df.fillna(method='ffill')
            code_df = code_df.where(pd.notnull(code_df), None)
            print('code: %s' % code)
            r = influx.saveData(code_df, save_db, save_measure)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('%s Error: %s' % (res_field_WD, r))
        return save_res

    def cal_factors(self, start, end, n_jobs):
        pd.set_option('mode.use_inf_as_na', True)
        factor_fields = ['net_profit_Q', 'net_profit_ddt_Q', 'oper_rev_Q']
        fail_list = []
        for factor_field in factor_fields:
            factor = self.influx.getDataMultiprocess('FinancialReport_Gus', factor_field, start, end)
            codes = factor['code'].unique()
            split_codes = np.array_split(codes, n_jobs)
            with parallel_backend('multiprocessing', n_jobs=n_jobs):
                res = Parallel()(delayed(Surprise.JOB_factors)
                                 (codes, factor, factor_field, self.db, self.measure) for codes in split_codes)
            print('Surprise: %s finish' % factor_field)
            print('-' * 30)
            for r in res:
                fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    pd.set_option('mode.use_inf_as_na', True)
    start_dt = datetime.datetime.now()
    sup = Surprise()
    r = sup.cal_factors(20100101, 20200315, N_JOBS)
    print(r)
    print('time token:', datetime.datetime.now()-start_dt)