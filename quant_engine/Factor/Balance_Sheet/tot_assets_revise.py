from factor_base import FactorBase
import pandas as pd
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import global_constant
from influxdb_data import influxdbData


class tot_assets_revise(FactorBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_former_data(series, Q_backward):
        report_period = tot_assets_revise.get_former_report_period(series['report_period'], Q_backward)
        if report_period not in series.index:
            return np.nan
        else:
            return series[report_period]

    @staticmethod
    def job_factors(calendar, codes, tot_assets, start, end):
        columns = tot_assets.columns
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
                code_tot_assets.apply(lambda row: tot_assets_revise.get_former_data(row, 0), axis=1)
            code_tot_assets['tot_assets_last1Q'] = \
                code_tot_assets.apply(lambda row: tot_assets_revise.get_former_data(row, 1), axis=1)
            code_tot_assets['tot_assets_last2Q'] = \
                code_tot_assets.apply(lambda row: tot_assets_revise.get_former_data(row, 2), axis=1)
            code_tot_assets['tot_assets_last3Q'] = \
                code_tot_assets.apply(lambda row: tot_assets_revise.get_former_data(row, 3), axis=1)
            code_tot_assets['tot_assets_lastY'] = \
                code_tot_assets.apply(lambda row: tot_assets_revise.get_former_data(row, 4), axis=1)
            code_tot_assets = code_tot_assets.loc[start:end,
                              ['code', 'report_period', 'tot_assets', 'tot_assets_last1Q',
                               'tot_assets_last2Q', 'tot_assets_last3Q', 'tot_assets_lastY']]
            code_tot_assets['report_period'] = code_tot_assets['report_period'].apply(lambda x: x.strftime('%Y%m%d'))
            code_tot_assets = code_tot_assets.where(pd.notnull(code_tot_assets), None)
            print('code: %s' % code)
            save_inf = influxdbData()
            save_inf.saveData(code_tot_assets, 'Financial_Report_Gus', 'tot_assets')

    def cal_factors(self, start, end):
        self.calendar = self.rdf.get_trading_calendar()
        start = str(start)
        end = str(end)
        calendar = set(self.calendar.loc[(self.calendar >= start) & (self.calendar <= end)])
        tot_assets = pd.read_hdf(global_constant.ROOT_DIR + 'Data_Resource/Balance_Sheet/tot_assets.h5', key='data')
        tot_assets = tot_assets.sort_values(by=['report_period', 'date'])
        tot_assets['date'] = pd.to_datetime(tot_assets['date'])
        tot_assets['report_period'] = pd.to_datetime(tot_assets['report_period'])
        tot_assets.set_index(['code', 'date', 'report_period'], inplace=True)
        tot_assets = tot_assets.unstack(level=2)
        tot_assets = tot_assets.loc[:, 'tot_assets']
        tot_assets.reset_index(inplace=True)
        tot_assets.set_index('date', inplace=True)
        codes = tot_assets['code'].unique()
        split_codes = np.array_split(codes, 10)
        with parallel_backend('multiprocessing', n_jobs=4):
            Parallel()(delayed(self.job_factors)(calendar, codes, tot_assets, start, end) for codes in split_codes)


if __name__ == '__main__':
    n = tot_assets_revise()
    n.cal_factors(20100101, 20190901)
