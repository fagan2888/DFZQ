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
    def JOB_factors(df, field, codes, calendar, start, save_db, if_TTM):
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
            # 计算TTM
            if if_TTM:
                TTM_cur_cols = [field]
                for i in range(1, 12):
                    TTM_cur_cols.append('{0}_last{1}M'.format(field, i))
                code_df['{0}_TTM'.format(field)] = np.where(np.any(pd.isnull(code_df[TTM_cur_cols]), axis=1), np.nan,
                                                            np.sum(code_df[TTM_cur_cols], axis=1))
                for n in range(1, 9):
                    TTM_field = '{0}_TTM_last{1}M'.format(field, n)
                    TTM_hist_cols = []
                    for i in range(n, n+12):
                        TTM_hist_cols.append('{0}_last{1}M'.format(field, i))
                    code_df[TTM_field] = np.where(np.any(pd.isnull(code_df[TTM_hist_cols]), axis=1), np.nan,
                                                  np.sum(code_df[TTM_hist_cols], axis=1))
            code_df = code_df.where(pd.notnull(code_df), None)
            print('code: %s' % code)
            r = influx.saveData(code_df, save_db, field)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('Sec Field: %s  Error: %s' % (field, r))
        return save_res


    def cal_factors(self, start, end, n_jobs):
        query = "select ANN_DT, S_INFO_WINDCODE, REPORT_PERIOD, OPER_REV, NET_PROFIT_EXCL_MIN_INT_INC, " \
                "TOT_SHRHLDR_EQY_EXCL_MIN_INT " \
                "from wind_filesync.AShareMonthlyReportsofBrokers " \
                "where ANN_DT >= {0} and ANN_DT <= {1} " \
                "order by report_period, ann_dt " \
            .format((dtparser.parse(str(start)) - relativedelta(years=4)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        report = pd.DataFrame(self.rdf.curs.fetchall(),
                              columns=['date', 'code', 'report_period', 'oper_rev_M', 'net_profit_M', 'net_equity_M'])
        report['date'] = pd.to_datetime(report['date'])
        report = report.sort_values('date')
        # 填充缺失
        report['net_equity_M'] = report.groupby('code')['net_equity_M'].fillna(method='ffill')
        query = "select S_INFO_WINDCODE, S_INFO_NAME, S_INFO_ENAME " \
                "from wind_filesync.WindCustomCode " \
                "where S_INFO_WINDCODE in " + str(tuple(report['code'].unique()))
        self.rdf.curs.execute(query)
        info = pd.DataFrame(self.rdf.curs.fetchall(), columns=['code', 'c_name', 'e_name'])
        report = pd.merge(report, info, on=['code'])
        report['e_name'] = report['e_name'].str.lower()
        c_names = report['c_name'].unique()
        e_names = report['e_name'].unique()
        c_nicknames = []
        for c_name in c_names:
            tmp = c_name
            tmp = tmp.split('（')[0]
            tmp = tmp.split('(')[0]
            tmp = tmp.split('股份有限公司')[0]
            tmp = tmp.split('证券')[0]
            tmp = tmp.split('联合')[0]
            if len(tmp) >= 4:
                if '上海' in tmp:
                    tmp = tmp.split('上海')[1]
                if '中国' in tmp:
                    tmp = tmp.split('中国')[1]
            c_nicknames.append(tmp)
        c_nickname_map = dict(zip(c_names, c_nicknames))
        e_nicknames = []
        for e_name in e_names:
            tmp = e_name
            tmp = tmp.split(' securities')[0]
            tmp = tmp.split(' security')[0]
            e_nicknames.append(tmp)
        e_nickname_map = dict(zip(e_names, e_nicknames))
        report['c_nickname'] = report['c_name'].map(c_nickname_map)
        report['e_nickname'] = report['e_name'].map(e_nickname_map)
        report['c_nickname'] = report['c_nickname'].replace('东证融汇', '东北')
        report['c_nickname'] = report['c_nickname'].replace('瑞信方正', '方正')
        report['c_nickname'] = report['c_nickname'].replace('银河金汇', '银河')
        report['c_nickname'] = report['c_nickname'].replace('浙江浙商', '浙商')
        report['c_nickname'] = report['c_nickname'].replace('中银国际', '中银')
        report['c_nickname'] = report['c_nickname'].replace('兴证', '兴业')
        report['c_nickname'] = report['c_nickname'].replace('金通', '中信')
        report['c_nickname'] = report['c_nickname'].replace('申万宏源西部', '申万宏源')
        report['c_nickname'] = report['c_nickname'].replace('摩根士丹利华鑫', '华鑫')
        # 替换 code
        for nickname in report['c_nickname'].unique():
            codes = report.loc[report['c_nickname'] == nickname, 'code'].unique()
            true_code = ''
            for code in codes:
                if code.split('.')[1] in ['SZ', 'SH']:
                    true_code = code
                    break
                else:
                    continue
            if true_code:
                report.loc[report['c_nickname'] == nickname, 'code'] = true_code
            else:
                continue
        # 个别 券商特殊维护
        report.loc[report['c_nickname'] == '华鑫', 'code'] = '600621.SH'
        report.loc[report['c_nickname'] == '兴业', 'code'] = '601377.SH'
        report.loc[report['c_nickname'] == '山西', 'code'] = '002500.SZ'
        report.loc[report['c_nickname'] == '东吴', 'code'] = '601555.SH'
        report.loc[report['c_nickname'] == '中原', 'code'] = '601375.SH'
        report.loc[report['c_nickname'] == '国信', 'code'] = '002736.SZ'
        report.loc[report['c_nickname'] == '申万宏源', 'code'] = '000166.SZ'
        report.loc[report['c_nickname'] == '财通', 'code'] = '601108.SH'
        report.loc[report['c_nickname'] == '国盛', 'code'] = '002670.SZ'
        # group
        report_sum = report.groupby(['date', 'code'])[['oper_rev_M', 'net_profit_M', 'net_equity_M']].sum()
        report_sum['report_period'] = report.groupby(['date', 'code'])['report_period'].first()
        report = report_sum.reset_index()
        # 处理数据
        calendar = self.rdf.get_trading_calendar()
        calendar = set(calendar.loc[(calendar >= (dtparser.parse(str(start)) - relativedelta(years=2)).
                                     strftime('%Y%m%d')) & (calendar <= str(end))])
        task = [('oper_rev_M', True), ('net_profit_M', True), ('net_equity_M', False)]
        # 存放的db
        fail_list = []
        for f, if_TTM in task:
            print('Sec \n field: %s begins processing...' % f)
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
                                 (df, f, codes, calendar, start, self.db, if_TTM) for codes in split_codes)
            print('%s finish' % f)
            print('-' * 30)
            for r in res:
                fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    mr = SecMonthlyReport()
    r = mr.cal_factors(20100101, 20200629, N_JOBS)