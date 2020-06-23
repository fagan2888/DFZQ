from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData
from global_constant import N_JOBS
import datetime


class SecMonthlyReport(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'FinancialReport_Gus'

    @staticmethod
    def JOB_factors(df, field, codes, calendar, start, save_db):
        columns = df.columns
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            insert_dates = calendar - set(code_df.index)
            content = [[np.nan] * len(columns)] * len(insert_dates)
            insert_df = pd.DataFrame(content, columns=columns, index=list(insert_dates))
            code_df = code_df.append(insert_df, ignore_index=False).sort_index()
            code_df = code_df.fillna(method='ffill')
            code_df = code_df.dropna(subset=['code'])
            code_df = code_df.loc[str(start):, ]
            # 所有report_period 为 columns, 去掉第一列(code)
            rp_keys = np.flipud(code_df.columns[1:])
            # 选择最新的report_period
            code_df['report_period'] = code_df.apply(lambda row: row.dropna().index[-1], axis=1)
            choices = []
            for rp in rp_keys:
                choices.append(code_df[rp].values)
            # 计算 当期 和 去年同期
            code_df['process_rp'] = code_df['report_period']
            conditions = []
            for rp in rp_keys:
                conditions.append(code_df['process_rp'].values == rp)
            code_df[field] = np.select(conditions, choices, default=np.nan)
            # 计算过去每一月
            res_flds = []
            for i in range(1, 21):
                res_field = field + '_last{0}M'.format(str(i))
                res_flds.append(res_field)
                code_df['process_rp'] = code_df['report_period'].apply(
                    lambda x: (pd.to_datetime(x) + datetime.timedelta(days=1) - relativedelta(months=i) -
                               datetime.timedelta(days=1)).strftime("%Y%m%d"))
                conditions = []
                for rp in rp_keys:
                    conditions.append(code_df['process_rp'].values == rp)
                code_df[res_field] = np.select(conditions, choices, default=np.nan)
            # 处理储存数据
            code_df = code_df.loc[:, ['code', 'report_period', field] + res_flds]
            code_df = code_df.where(pd.notnull(code_df), None)
            print('code: %s' % code)
            r = influx.saveData(code_df, save_db, field)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('Banks Field: %s  Error: %s' % (field, r))
        return save_res


    def cal_factors(self, start, end, n_jobs):
        query = "select ANN_DT, S_INFO_WINDCODE, REPORT_PERIOD, OPER_REV, NET_PROFIT_EXCL_MIN_INT_INC, " \
                "TOT_SHRHLDR_EQY_EXCL_MIN_INT " \
                "from wind_filesync.AShareMonthlyReportsofBrokers " \
                "where ANN_DT >= {0} and ANN_DT <= {1} " \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "order by report_period, ann_dt " \
            .format((dtparser.parse(str(start)) - relativedelta(years=4)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        report = pd.DataFrame(self.rdf.curs.fetchall(),
                              columns=['date', 'code', 'report_period', 'oper_rev_M', 'net_profit_M', 'net_equity_M'])
        report.set_index('date', inplace=True)
        report.index = pd.to_datetime(report.index)
        # 处理数据
        calendar = self.rdf.get_trading_calendar()
        calendar = \
            set(calendar.loc[(calendar >= (dtparser.parse(str(start)) - relativedelta(years=2)).strftime('%Y%m%d')) &
                             (calendar <= str(end))])
        fields = ['oper_rev_M', 'net_profit_M', 'net_equity_M']
        # 存放的db
        fail_list = []
        for f in fields:
            print('BANKS \n field: %s begins processing...' % f)
            df = pd.DataFrame(report.dropna(subset=[f]).groupby(['code', 'date', 'report_period'])[f].last()) \
                .reset_index()
            df = df.sort_values(by=['report_period', 'date'])
            df.set_index(['code', 'date', 'report_period'], inplace=True)
            df = df.unstack(level=2)
            df = df.loc[:, f]
            df = df.reset_index().set_index('date')
            codes = df['code'].unique()
            split_codes = np.array_split(codes, n_jobs)
            with parallel_backend('multiprocessing', n_jobs=n_jobs):
                res = Parallel()(delayed(SecMonthlyReport.JOB_factors)
                                 (df, f, codes, calendar, start, self.db) for codes in split_codes)
            print('%s finish' % f)
            print('-' * 30)
            for r in res:
                fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    mr = SecMonthlyReport()
    r = mr.cal_factors(20090101, 20200622, N_JOBS)