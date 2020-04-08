from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData
from global_constant import N_JOBS


class CashFlowUpdate(FactorBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_former_data(series, n_Qs):
        report_period = FactorBase.get_former_report_period(series['report_period'], n_Qs)
        if report_period not in series.index:
            return np.nan
        else:
            return series[report_period]

    @staticmethod
    def JOB_factors(df, field, codes, calendar, start):
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
            code_df['report_period'] = code_df.apply(lambda row: row.dropna().index[-1], axis=1)
            code_df[field] = \
                code_df.apply(lambda row: CashFlowUpdate.get_former_data(row, 0), axis=1)
            code_df[field + '_last1Q'] = \
                code_df.apply(lambda row: CashFlowUpdate.get_former_data(row, 1), axis=1)
            code_df[field + '_last2Q'] = \
                code_df.apply(lambda row: CashFlowUpdate.get_former_data(row, 2), axis=1)
            code_df[field + '_last3Q'] = \
                code_df.apply(lambda row: CashFlowUpdate.get_former_data(row, 3), axis=1)
            code_df[field + '_lastY'] = \
                code_df.apply(lambda row: CashFlowUpdate.get_former_data(row, 4), axis=1)
            code_df[field + '_last4Q'] = \
                code_df.apply(lambda row: CashFlowUpdate.get_former_data(row, 4), axis=1)
            code_df[field + '_last5Q'] = \
                code_df.apply(lambda row: CashFlowUpdate.get_former_data(row, 5), axis=1)
            code_df[field + '_last6Q'] = \
                code_df.apply(lambda row: CashFlowUpdate.get_former_data(row, 6), axis=1)
            code_df[field + '_last7Q'] = \
                code_df.apply(lambda row: CashFlowUpdate.get_former_data(row, 7), axis=1)
            code_df[field + '_last8Q'] = \
                code_df.apply(lambda row: CashFlowUpdate.get_former_data(row, 8), axis=1)
            code_df[field + '_last9Q'] = \
                code_df.apply(lambda row: CashFlowUpdate.get_former_data(row, 9), axis=1)
            code_df[field + '_last10Q'] = \
                code_df.apply(lambda row: CashFlowUpdate.get_former_data(row, 10), axis=1)
            code_df[field + '_last11Q'] = \
                code_df.apply(lambda row: CashFlowUpdate.get_former_data(row, 11), axis=1)
            code_df[field + '_last12Q'] = \
                code_df.apply(lambda row: CashFlowUpdate.get_former_data(row, 12), axis=1)

            code_df = \
                code_df.loc[str(start):,
                ['code', 'report_period', field, field + '_last1Q', field + '_last2Q', field + '_last3Q',
                 field + '_lastY', field + '_last4Q', field + '_last5Q', field + '_last6Q', field + '_last7Q',
                 field + '_last8Q', field + '_last9Q', field + '_last10Q', field + '_last11Q', field + '_last12Q']]
            code_df['report_period'] = code_df['report_period'].apply(lambda x: x.strftime('%Y%m%d'))
            code_df = code_df.where(pd.notnull(code_df), None)
            print('code: %s' % code)
            r = influx.saveData(code_df, 'FinancialReport_Gus', field)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('CashFlow  Field: %s  Error: %s' % (field, r))
        return save_res

    def cal_factors(self, start, end, n_jobs):
        # type要用408001000，408005000，408004000(合并报表，合并更正前，合并调整后)，同时有408001000和408005000用408005000
        # 有408004000时，根据ann_dt酌情使用
        query = "select ANN_DT, S_INFO_WINDCODE, REPORT_PERIOD, NET_CASH_FLOWS_OPER_ACT, " \
                "NET_INCR_CASH_CASH_EQU, STATEMENT_TYPE " \
                "from wind_filesync.AShareCashFlow " \
                "where ANN_DT >= {0} and ANN_DT <= {1} " \
                "and (STATEMENT_TYPE = '408001000' or STATEMENT_TYPE = '408005000' or STATEMENT_TYPE = '408004000') " \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "order by report_period, ann_dt, statement_type " \
            .format((dtparser.parse(str(start)) - relativedelta(years=4)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        cash_flow = pd.DataFrame(self.rdf.curs.fetchall(),
                                 columns=['date', 'code', 'report_period', 'net_OCF', 'net_CF', 'type'])
        # 同一code，同一date，同一report_period，同时出现type1，2，3时，取type大的
        cash_flow['type'] = \
            cash_flow['type'].apply(lambda x: '1' if x == '408001000' else ('2' if x == '408005000' else '3'))
        cash_flow = cash_flow.sort_values(by=['code', 'date', 'report_period', 'type'])
        cash_flow['date'] = pd.to_datetime(cash_flow['date'])
        cash_flow['report_period'] = pd.to_datetime((cash_flow['report_period']))
        # 处理数据
        calendar = self.rdf.get_trading_calendar()
        calendar = \
            set(calendar.loc[(calendar >= (dtparser.parse(str(start)) - relativedelta(years=2)).strftime('%Y%m%d')) &
                             (calendar <= str(end))])
        fields = cash_flow.columns.difference(['date', 'code', 'report_period', 'type'])
        fail_list = []
        for f in fields:
            print('field: %s begins processing...' %f)
            df = pd.DataFrame(cash_flow.dropna(subset=[f]).groupby(['code', 'date', 'report_period'])[f].last())\
                .reset_index()
            df = df.sort_values(by=['report_period', 'date'])
            df.set_index(['code', 'date', 'report_period'], inplace=True)
            df = df.unstack(level=2)
            df = df.loc[:, f]
            df = df.reset_index().set_index('date')
            codes = df['code'].unique()
            split_codes = np.array_split(codes, n_jobs)
            with parallel_backend('multiprocessing', n_jobs=n_jobs):
                res = Parallel()(delayed(CashFlowUpdate.JOB_factors)
                                 (df, f, codes, calendar, start) for codes in split_codes)
            print('%s finish' % f)
            print('-' * 30)
            for r in res:
                fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    cu = CashFlowUpdate()
    r = cu.cal_factors(20100101, 20200407, N_JOBS)
