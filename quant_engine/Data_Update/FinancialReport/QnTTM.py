from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData
from global_constant import N_JOBS


# 财报数据更新后计算Q和ttm
class QnTTMUpdate(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'FinancialReport_Gus'
        # 目前包含字段: 净利润(net_profit)，扣非净利润(net_profit_ddt)，总利润(tot_profit), 营收(oper_rev)，
        #              总营收(tot_oper_rev)，总营业成本(tot_oper_cost), 营业利润(oper_profit), 现金流净额(net_CF),
        #              经营现金流净额(net_OCF)，经营利润(oper_income)，毛利(gross_margin)
        self.fields = ['net_profit', 'net_profit_ddt', 'tot_profit', 'oper_rev', 'tot_oper_rev', 'tot_oper_cost',
                       'oper_profit', 'net_CF', 'net_OCF', 'oper_income', 'gross_margin', 'non_interest_income',
                       'interest_income']

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
    def JOB_factors(df, field, codes):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            drop_dupl_df = code_df.drop_duplicates().copy()
            # 计算Q
            drop_dupl_df[field + '_Q'] = drop_dupl_df.apply(lambda row: QnTTMUpdate.JOB_calQ(
                row[field], row[field + '_last1Q'], row['report_period'], 0), axis=1)
            drop_dupl_df[field + '_Q_lastY'] = drop_dupl_df.apply(lambda row: QnTTMUpdate.JOB_calQ(
                row[field + '_lastY'], row[field + '_last5Q'], row['report_period'], 4), axis=1)
            Q_cols = [field + '_Q', field + '_Q_lastY']
            for n in range(1, 12):
                curr_col = field + '_last{0}Q'.format(str(n))
                prev_col = field + '_last{0}Q'.format(str(n+1))
                res_col = field + '_Q_last{0}Q'.format(str(n))
                Q_cols.append(res_col)
                drop_dupl_df[res_col] = drop_dupl_df.apply(lambda row: QnTTMUpdate.JOB_calQ(
                    row[curr_col], row[prev_col], row['report_period'], n), axis=1)
            # 计算TTM
            drop_dupl_df[field + '_TTM'] = drop_dupl_df[field + '_Q'] + drop_dupl_df[field + '_Q_last1Q'] + \
                                           drop_dupl_df[field + '_Q_last2Q'] + drop_dupl_df[field + '_Q_last3Q']
            TTM_cols = [field + '_TTM']
            for n in range(1, 9):
                curr_col = field + '_Q_last{0}Q'.format(str(n))
                prev1_col = field + '_Q_last{0}Q'.format(str(n+1))
                prev2_col = field + '_Q_last{0}Q'.format(str(n+2))
                prev3_col = field + '_Q_last{0}Q'.format(str(n+3))
                res_col = field + '_TTM_last{0}Q'.format(str(n))
                TTM_cols.append(res_col)
                drop_dupl_df[res_col] = drop_dupl_df[curr_col] + drop_dupl_df[prev1_col] + \
                                        drop_dupl_df[prev2_col] + drop_dupl_df[prev3_col]
            Q_TTM_cols = Q_cols + TTM_cols
            drop_dupl_df = drop_dupl_df[Q_TTM_cols]
            code_df = pd.merge(code_df.loc[:, ['code', 'report_period']], drop_dupl_df,
                               how='left', left_index=True, right_index=True)
            code_df = code_df.fillna(method='ffill')
            code_df = code_df.where(pd.notnull(code_df), None)
            Q_df = code_df.loc[:, ['code', 'report_period'] + Q_cols]
            TTM_df = code_df.loc[:, ['code', 'report_period'] + TTM_cols]
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
    r = QU.cal_factors(20090101, 20200604, N_JOBS)
    print(r)