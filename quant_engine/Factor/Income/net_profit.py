# 净利润数据
# 原数据

from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData
import global_constant


class net_profit(FactorBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def JOB_factors(net_profit, codes, calendar, start):
        columns = net_profit.columns
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_net_profit = net_profit.loc[net_profit['code'] == code, :].copy()
            insert_dates = calendar - set(code_net_profit.index)
            content = [[np.nan] * len(columns)] * len(insert_dates)
            insert_df = pd.DataFrame(content, columns=columns, index=list(insert_dates))
            code_net_profit = code_net_profit.append(insert_df, ignore_index=False).sort_index()
            code_net_profit = code_net_profit.fillna(method='ffill')
            code_net_profit = code_net_profit.dropna(subset=['code'])
            code_net_profit['report_period'] = code_net_profit.apply(lambda row: row.dropna().index[-1], axis=1)
            code_net_profit['net_profit'] = \
                code_net_profit.apply(lambda row: TotAssets.get_former_data(row, 0), axis=1)
            code_net_profit['net_profit_last1Q'] = \
                code_net_profit.apply(lambda row: TotAssets.get_former_data(row, 1), axis=1)
            code_net_profit['net_profit_last2Q'] = \
                code_net_profit.apply(lambda row: TotAssets.get_former_data(row, 2), axis=1)
            code_net_profit['net_profit_last3Q'] = \
                code_net_profit.apply(lambda row: TotAssets.get_former_data(row, 3), axis=1)
            code_net_profit['net_profit_lastY'] = \
                code_net_profit.apply(lambda row: TotAssets.get_former_data(row, 4), axis=1)
            code_net_profit['net_profit_last5Q'] = \
                code_net_profit.apply(lambda row: TotAssets.get_former_data(row, 5), axis=1)
            code_net_profit = \
                code_net_profit.loc[str(start):,
                ['code', 'report_period', 'net_profit', 'net_profit_last1Q', 'net_profit_last2Q', 'net_profit_last3Q',
                 'net_profit_lastY', 'net_profit_last5Q']]
            code_net_profit['report_period'] = code_net_profit['report_period'].apply(lambda x: x.strftime('%Y%m%d'))
            code_net_profit = code_net_profit.where(pd.notnull(code_net_profit), None)
            print('code: %s' % code)
            r = influx.saveData(code_net_profit, 'Financial_Report_Gus', 'net_equity')
            if r == 'No error occurred...':
                pass
            else:
                save_res.append(r)
        return save_res

    def cal_factors(self, start, end, n_jobs):
        # type要用408001000，408005000，408004000(合并报表，合并更正前，合并调整后)，同时有408001000和408005000用408005000
        # 有408004000时，根据ann_dt酌情使用
        query = "select ANN_DT, S_INFO_WINDCODE, REPORT_PERIOD, NET_PROFIT_EXCL_MIN_INT_INC, " \
                "NET_PROFIT_AFTER_DED_NR_LP, OPER_REV, TOT_OPER_REV, OPER_PROFIT, S_FA_EPS_DILUTED, STATEMENT_TYPE " \
                "from wind_filesync.AShareIncome " \
                "where ANN_DT >= {0} and ANN_DT <= {1} " \
                "and (STATEMENT_TYPE = '408001000' or STATEMENT_TYPE = '408005000' or STATEMENT_TYPE = '408004000') " \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "order by report_period, ann_dt, statement_type " \
            .format((dtparser.parse(str(start)) - relativedelta(years=2)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        income = pd.DataFrame(self.rdf.curs.fetchall(),
                              columns=['date', 'code', 'report_period', 'net_profit', 'net_profit_ddt', 'oper_rev',
                                       'tot_oper_rev', 'oper_profit', 'EPS_diluted', 'type'])
        # 同一code，同一date，同一report_period，同时出现type1，2，3时，取type大的
        income['type'] = income['type'].apply(lambda x: '1' if x == '408001000' else ('2' if x == '408005000' else '3'))
        income = income.sort_values(by=['code', 'date', 'report_period', 'type'])
        income['date'] = pd.to_datetime(income['date'])
        income['report_period'] = pd.to_datetime((income['report_period']))
        net_profit = income.dropna(subset=['net_profit']).groupby(['code', 'date', 'report_period'])[
            'net_profit'].last()
        net_profit_ddt = income.dropna(subset=['net_profit_ddt']).groupby(['code', 'date', 'report_period'])[
            'net_profit_ddt'].last()
        oper_rev = income.dropna(subset=['oper_rev']).groupby(['code', 'date', 'report_period'])[
            'oper_rev'].last()
        tot_oper_rev = income.dropna(subset=['tot_oper_rev']).groupby(['code', 'date', 'report_period'])[
            'tot_oper_rev'].last()
        # 处理数据
        calendar = self.rdf.get_trading_calendar()
        calendar = \
            set(calendar.loc[(calendar >= (dtparser.parse(str(start)) - relativedelta(years=2)).strftime('%Y%m%d')) &
                             (calendar <= str(end))])
        net_profit = net_profit.sort_values(by=['report_period', 'date'])
        net_profit['date'] = pd.to_datetime(net_profit['date'])
        net_profit['report_period'] = pd.to_datetime(net_profit['report_period'])
        net_profit.set_index(['code', 'date', 'report_period'], inplace=True)
        net_profit = net_profit.unstack(level=2)
        net_profit = net_profit.loc[:, 'net_profit']
        net_profit.reset_index(inplace=True)
        net_profit.set_index('date', inplace=True)
        
        
        codes = net_profit['code'].unique()
        split_codes = np.array_split(codes, n_jobs)

        return


if __name__ == '__main__':
    netprofit_data = net_profit()
    r = netprofit_data.cal_factors(20100101, 20190901,4)
    print('data got')
    net_profit = r[0]
    net_profit_ddt = r[1]
    h5 = pd.HDFStore(global_constant.ROOT_DIR + 'Data_Resource/Income/net_profit.h5', 'w')
    h5['data'] = net_profit
    h5.close()
    print('net profit saved')
    h5 = pd.HDFStore(global_constant.ROOT_DIR + 'Data_Resource/Income/net_profit_ddt.h5', 'w')
    h5['data'] = net_profit_ddt
    h5.close()
    print('net profit ddt saved')
