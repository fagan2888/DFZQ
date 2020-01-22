# 股东权益数据

from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData
import global_constant


class TotAssets(FactorBase):
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
    def JOB_factors(tot_assets, codes, calendar, start):
        columns = tot_assets.columns
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_tot_assets = tot_assets.loc[tot_assets['code'] == code, :].copy()
            insert_dates = calendar - set(code_tot_assets.index)
            content = [[np.nan] * len(columns)] * len(insert_dates)
            insert_df = pd.DataFrame(content, columns=columns, index=list(insert_dates))
            code_tot_assets = code_tot_assets.append(insert_df, ignore_index=False).sort_index()
            code_tot_assets = code_tot_assets.fillna(method='ffill')
            code_tot_assets = code_tot_assets.dropna(subset=['code'])
            code_tot_assets['report_period'] = code_tot_assets.apply(lambda row: row.dropna().index[-1], axis=1)
            code_tot_assets['tot_assets'] = \
                code_tot_assets.apply(lambda row: TotAssets.get_former_data(row, 0), axis=1)
            code_tot_assets['tot_assets_last1Q'] = \
                code_tot_assets.apply(lambda row: TotAssets.get_former_data(row, 1), axis=1)
            code_tot_assets['tot_assets_last2Q'] = \
                code_tot_assets.apply(lambda row: TotAssets.get_former_data(row, 2), axis=1)
            code_tot_assets['tot_assets_last3Q'] = \
                code_tot_assets.apply(lambda row: TotAssets.get_former_data(row, 3), axis=1)
            code_tot_assets['tot_assets_lastY'] = \
                code_tot_assets.apply(lambda row: TotAssets.get_former_data(row, 4), axis=1)
            code_tot_assets['tot_assets_last5Q'] = \
                code_tot_assets.apply(lambda row: TotAssets.get_former_data(row, 5), axis=1)
            code_tot_assets = \
                code_tot_assets.loc[str(start):,
                ['code', 'report_period', 'tot_assets', 'tot_assets_last1Q', 'tot_assets_last2Q', 'tot_assets_last3Q',
                 'tot_assets_lastY', 'tot_assets_last5Q']]
            code_tot_assets['report_period'] = code_tot_assets['report_period'].apply(lambda x: x.strftime('%Y%m%d'))
            code_tot_assets = code_tot_assets.where(pd.notnull(code_tot_assets), None)
            print('code: %s' % code)
            r = influx.saveData(code_tot_assets, 'Financial_Report_Gus', 'net_equity')
            if r == 'No error occurred...':
                pass
            else:
                save_res.append(r)
        return save_res

    def cal_factors(self, start, end, n_jobs):
        # type要用408001000，408005000，408004000(合并报表，合并更正前，合并调整后)，同时有408001000和408005000用408005000
        # 有408004000时，根据ann_dt酌情使用
        query = "select ANN_DT, S_INFO_WINDCODE, REPORT_PERIOD, TOT_ASSETS, STATEMENT_TYPE " \
                "from wind_filesync.AShareBalanceSheet " \
                "where ANN_DT >= {0} and ANN_DT <= {1} " \
                "and (STATEMENT_TYPE = '408001000' or STATEMENT_TYPE = '408005000' or STATEMENT_TYPE = '408004000') " \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '60%') " \
                "order by report_period, ann_dt, statement_type " \
            .format((dtparser.parse(str(start)) - relativedelta(years=2)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        balance_sheet = pd.DataFrame(self.rdf.curs.fetchall(),
                                     columns=['date', 'code', 'report_period', 'tot_assets', 'type'])
        # 同一code，同一date，同一report_period，同时出现type1，2，3时，取type大的
        balance_sheet['type'] = balance_sheet['type'].apply(
            lambda x: '1' if x == '408001000' else ('2' if x == '408005000' else '3'))
        balance_sheet = balance_sheet.sort_values(by=['code', 'date', 'report_period', 'type'])
        tot_assets = balance_sheet.dropna(subset=['tot_assets']).groupby(['code', 'date', 'report_period'])[
            'tot_assets'].last()
        tot_assets = pd.DataFrame(tot_assets).reset_index()
        # 处理数据
        calendar = self.rdf.get_trading_calendar()
        calendar = \
            set(calendar.loc[(calendar >= (dtparser.parse(str(start)) - relativedelta(years=2)).strftime('%Y%m%d')) &
                             (calendar <= str(end))])
        tot_assets = tot_assets.sort_values(by=['report_period', 'date'])
        tot_assets['date'] = pd.to_datetime(tot_assets['date'])
        tot_assets['report_period'] = pd.to_datetime(tot_assets['report_period'])
        tot_assets.set_index(['code', 'date', 'report_period'], inplace=True)
        tot_assets = tot_assets.unstack(level=2)
        tot_assets = tot_assets.loc[:, 'tot_assets']
        tot_assets.reset_index(inplace=True)
        tot_assets.set_index('date', inplace=True)
        codes = tot_assets['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(TotAssets.JOB_factors)
                             (tot_assets, codes, calendar, start) for codes in split_codes)
        print('NetEquity finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    tot_assets = TotAssets()
    r = tot_assets.cal_factors(20100101, 20200122, 4)
    print(r)