from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData
from global_constant import N_JOBS


# 财报数据更新后计算Q和ttm
class QnTTMUpdate(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'FinancialReport_Gus'
        # 目前包含字段: 净利润(net_profit)，扣非净利润(net_profit_ddt)，营收(oper_rev)，总营收(tot_oper_rev)，
        #              营业利润(oper_profit), 现金流净额(net_CF), 经营现金流净额(net_OCF)
        self.fields = ['net_profit', 'net_profit_ddt', 'oper_rev', 'tot_oper_rev', 'oper_profit', 'net_CF', 'net_OCF',
                       'oper_income']

    @staticmethod
    def JOB_calQ(value_cur, value_last, report_period, n_last):
        rp = QnTTMUpdate.get_former_report_period(dtparser.parse(report_period), n_last).strftime('%Y%m%d')
        if rp[-4:] == '0331':
            Q = value_cur
        else:
            if pd.isnull(value_last):
                Q = np.nan
            else:
                Q = value_cur - value_last
        return Q

    @staticmethod
    def JOB_calTTM(value_cur, value_last1Q, value_last2Q, value_last3Q, report_period, n_last):
        rp = QnTTMUpdate.get_former_report_period(dtparser.parse(report_period), n_last).strftime('%Y%m%d')
        if rp[-4:] == '1231':
            TTM = value_cur
        else:
            TTM = value_cur + value_last1Q + value_last2Q + value_last3Q
        return TTM

    @staticmethod
    def JOB_factors(df, field, codes):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            # 计算Q
            code_df[field + '_Q'] = code_df.apply(lambda row: QnTTMUpdate.JOB_calQ(
                row[field], row[field + '_last1Q'], row['report_period'], 0), axis=1)
            code_df[field + '_Q_lastY'] = code_df.apply(lambda row: QnTTMUpdate.JOB_calQ(
                row[field + '_lastY'], row[field + '_last5Q'], row['report_period'], 4), axis=1)
            Q_cols = ['code', 'report_period', field + '_Q', field + '_Q_lastY']
            for n in range(1, 12):
                curr_col = field + '_last{0}Q'.format(str(n))
                prev_col = field + '_last{0}Q'.format(str(n+1))
                res_col = field + '_Q_last{0}Q'.format(str(n))
                Q_cols.append(res_col)
                code_df[res_col] = code_df.apply(lambda row: QnTTMUpdate.JOB_calQ(
                    row[curr_col], row[prev_col], row['report_period'], n), axis=1)
            # 计算TTM
            code_df[field + '_TTM'] = code_df.apply(lambda row: QnTTMUpdate.JOB_calTTM(
                row[field + '_Q'], row[field + '_Q_last1Q'], row[field + '_Q_last2Q'], row[field + '_Q_last3Q'],
                row['report_period'], 0), axis=1)
            TTM_cols = ['code', 'report_period', field + '_TTM']
            for n in range(1, 9):
                curr_col = field + '_Q_last{0}Q'.format(str(n))
                prev1_col = field + '_Q_last{0}Q'.format(str(n+1))
                prev2_col = field + '_Q_last{0}Q'.format(str(n+2))
                prev3_col = field + '_Q_last{0}Q'.format(str(n+3))
                res_col = field + '_TTM_last{0}Q'.format(str(n))
                TTM_cols.append(res_col)
                code_df[res_col] = code_df.apply(lambda row: QnTTMUpdate.JOB_calTTM(
                    row[curr_col], row[prev1_col], row[prev2_col], row[prev3_col],
                    row['report_period'], n), axis=1)
            code_df = code_df.where(pd.notnull(code_df), None)
            Q_df = code_df.loc[:, Q_cols]
            TTM_df = code_df.loc[:, TTM_cols]
            print('code: %s   field: %s' % (code, field + '_Q'))
            r = influx.saveData(Q_df, 'FinancialReport_Gus', field + '_Q')
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('QnTTM  Field: %s  Error: %s' % (field + '_Q', r))
            print('code: %s   field: %s' % (code, field + '_TTM'))
            r = influx.saveData(TTM_df, 'FinancialReport_Gus', field + '_TTM')
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('QnTTM  Field: %s  Error: %s' % (field + '_TTM', r))
        return save_res

    def cal_factors(self, start, end, n_jobs):
        fail_list = []
        for f in self.fields:
            print('field: %s begins calculating Q and TTM...' % f)
            start = str(start)
            end = str(end)
            df = self.influx.getDataMultiprocess(self.db, f, start, end, None)
            fill_cols = [f, f+'_last1Q', f+'_last2Q', f+'_last3Q', f+'_last4Q', f+'_last5Q', f+'_last6Q',
                         f+'_last7Q', f+'_last8Q', f+'_last9Q', f+'_last10Q', f+'_last11Q', f+'_last12Q']
            df[fill_cols] = df[fill_cols].fillna(method='bfill', axis=1)
            codes = df['code'].unique()
            split_codes = np.array_split(codes, n_jobs)
            with parallel_backend('multiprocessing', n_jobs=n_jobs):
                res = Parallel()(delayed(QnTTMUpdate.JOB_factors)
                                 (df, f, codes) for codes in split_codes)
            print('%s finish' % f)
            print('-' * 30)
            for r in res:
                fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    QU = QnTTMUpdate()
    r = QU.cal_factors(20100101, 20200301, N_JOBS)
    print(r)