from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData
import global_constant


# 财报数据更新后计算Q和ttm
class QnTTMUpdate(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'Financial_Report_Gus'
        # 目前包含字段: 净利润(net_profit)，扣非净利润(net_profit_ddt)，营收(oper_rev)，总营收(tot_oper_rev)，
        #              营业利润(oper_profit)
        self.fields = ['net_profit', 'net_profit_ddt', 'oper_rev', 'tot_oper_rev', 'oper_profit']

    @staticmethod
    def JOB_calQ(value_cur, value_last, report_period):
        if report_period[-4:] == '0331':
            Q = value_cur
        else:
            if pd.isnull(value_last):
                Q = np.nan
            else:
                Q = value_cur - value_last
        return Q


    @staticmethod
    def JOB_factors(df, field, codes):
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_df[field + '_Q'] = code_df.apply(
                lambda row: QnTTMUpdate.JOB_calQ(row[field], row[field + '_last1Q'], row['report_period']), axis=1)
            code_df[field + '_Q_last1Q'] = code_df.apply(
                lambda row: QnTTMUpdate.JOB_calQ(row[field + '_last1Q'], row[field + '_last2Q'], row['report_period']),
                axis=1)
            code_df[field + '_Q_last2Q'] = code_df.apply(
                lambda row: QnTTMUpdate.JOB_calQ(row[field + '_last2Q'], row[field + '_last3Q'], row['report_period']),
                axis=1)
            code_df[field + '_Q_last3Q'] = code_df.apply(
                lambda row: QnTTMUpdate.JOB_calQ(row[field + '_last3Q'], row[field + '_lastY'], row['report_period']),
                axis=1)
            code_df[field + '_Q_lastY'] = code_df.apply(
                lambda row: QnTTMUpdate.JOB_calQ(row[field + '_lastY'], row[field + '_last5Q'], row['report_period']),
                axis=1)

            print('.')

    def cal_factors(self, start, end, n_jobs):
        for f in self.fields:
            start = str(start)
            end = str(end)
            df = self.influx.getDataMultiprocess(self.db, f, start, end, None)
            fill_cols = [f, f+'_last1Q', f+'_last2Q', f+'_last3Q', f+'_lastY', f+'_last5Q']
            df[fill_cols] = df[fill_cols].fillna(method='bfill', axis=1)
            codes = df['code'].unique()
            split_codes = np.array_split(codes, n_jobs)
            with parallel_backend('multiprocessing', n_jobs=n_jobs):
                res = Parallel()(delayed(QnTTMUpdate.JOB_factors)
                                 (df, f, codes) for codes in split_codes)


if __name__ == '__main__':
    QU = QnTTMUpdate()
    QU.cal_factors(20180101,20190101,4)