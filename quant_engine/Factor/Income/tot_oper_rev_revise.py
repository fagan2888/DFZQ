from factor_base import FactorBase
import pandas as pd
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import global_constant
from influxdb_data import influxdbData


class tot_oper_rev_revise(FactorBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_former_data(series, Q_backward):
        report_period = tot_oper_rev_revise.get_former_report_period(series['report_period'], Q_backward)
        if report_period not in series.index:
            return np.nan
        else:
            return series[report_period]

    @staticmethod
    def job_factors(calendar, codes, tot_oper_rev, start, end):
        columns = tot_oper_rev.columns
        for code in codes:
            code_tot_oper_rev = tot_oper_rev.loc[tot_oper_rev['code'] == code, :].copy()
            insert_dates = calendar - set(code_tot_oper_rev.index)
            content = [[np.nan] * len(columns)] * len(insert_dates)
            insert_df = pd.DataFrame(content, columns=columns, index=list(insert_dates))
            code_tot_oper_rev = code_tot_oper_rev.append(insert_df, ignore_index=False).sort_index()
            code_tot_oper_rev = code_tot_oper_rev.fillna(method='ffill')
            code_tot_oper_rev = code_tot_oper_rev.dropna(subset=['code'])
            code_tot_oper_rev['report_period'] = code_tot_oper_rev.apply(lambda row: row.dropna().index[-1], axis=1)
            # 二维数据变成一维数据，取得tot_oper_rev
            code_tot_oper_rev['tot_oper_rev'] = \
                code_tot_oper_rev.apply(lambda row: tot_oper_rev_revise.get_former_data(row, 0), axis=1)
            code_tot_oper_rev['tot_oper_rev_last1Q'] = \
                code_tot_oper_rev.apply(lambda row: tot_oper_rev_revise.get_former_data(row, 1), axis=1)
            code_tot_oper_rev['tot_oper_rev_last2Q'] = \
                code_tot_oper_rev.apply(lambda row: tot_oper_rev_revise.get_former_data(row, 2), axis=1)
            code_tot_oper_rev['tot_oper_rev_last3Q'] = \
                code_tot_oper_rev.apply(lambda row: tot_oper_rev_revise.get_former_data(row, 3), axis=1)
            code_tot_oper_rev['tot_oper_rev_lastY'] = \
                code_tot_oper_rev.apply(lambda row: tot_oper_rev_revise.get_former_data(row, 4), axis=1)
            code_tot_oper_rev['tot_oper_rev_last5Q'] = \
                code_tot_oper_rev.apply(lambda row: tot_oper_rev_revise.get_former_data(row, 5), axis=1)
            code_tot_oper_rev['report_period'] = \
                code_tot_oper_rev['report_period'].apply(lambda x: x.strftime('%Y%m%d'))
            code_tot_oper_rev = code_tot_oper_rev.loc[:,
                                ['code', 'report_period', 'tot_oper_rev', 'tot_oper_rev_last1Q', 'tot_oper_rev_last2Q',
                                 'tot_oper_rev_last3Q', 'tot_oper_rev_lastY', 'tot_oper_rev_last5Q']]
            # 由tot_oper_rev求得tot_oper_rev_Q
            code_tot_oper_rev['tot_oper_rev_Q'] = \
                code_tot_oper_rev.apply(lambda row: row['tot_oper_rev'] if row['report_period'][-4:] == '0331'
                else row['tot_oper_rev'] - row['tot_oper_rev_last1Q'], axis=1)
            code_tot_oper_rev['tot_oper_rev_Q_last1Q'] = \
                code_tot_oper_rev.apply(lambda row: row['tot_oper_rev_last1Q'] if row['report_period'][-4:] == '0630'
                else row['tot_oper_rev_last1Q'] - row['tot_oper_rev_last2Q'], axis=1)
            code_tot_oper_rev['tot_oper_rev_Q_last2Q'] = \
                code_tot_oper_rev.apply(lambda row: row['tot_oper_rev_last2Q'] if row['report_period'][-4:] == '0930'
                else row['tot_oper_rev_last2Q'] - row['tot_oper_rev_last3Q'], axis=1)
            code_tot_oper_rev['tot_oper_rev_Q_last3Q'] = \
                code_tot_oper_rev.apply(lambda row: row['tot_oper_rev_last3Q'] if row['report_period'][-4:] == '1231'
                else row['tot_oper_rev_last3Q'] - row['tot_oper_rev_lastY'], axis=1)
            code_tot_oper_rev['tot_oper_rev_Q_lastY'] = \
                code_tot_oper_rev.apply(lambda row: row['tot_oper_rev_lastY'] if row['report_period'][-4:] == '0331'
                else row['tot_oper_rev_lastY'] - row['tot_oper_rev_last5Q'], axis=1)
            code_tot_oper_rev = code_tot_oper_rev.loc[start:end, :]
            code_tot_oper_rev = code_tot_oper_rev.where(pd.notnull(code_tot_oper_rev), None)
            print('code: %s' % code)
            save_inf = influxdbData()
            save_inf.saveData(code_tot_oper_rev, 'Financial_Report_Gus', 'tot_oper_rev')

    def cal_factors(self, start, end):
        calendar = self.rdf.get_trading_calendar()
        start = str(start)
        end = str(end)
        calendar = set(calendar.loc[(calendar >= start) & (calendar <= end)])
        tot_oper_rev = pd.read_hdf(global_constant.ROOT_DIR + 'Data_Resource/Income/tot_oper_rev.h5', key='data')
        tot_oper_rev = tot_oper_rev.sort_values(by=['report_period', 'date'])
        tot_oper_rev['date'] = pd.to_datetime(tot_oper_rev['date'])
        tot_oper_rev['report_period'] = pd.to_datetime(tot_oper_rev['report_period'])
        tot_oper_rev.set_index(['code', 'date', 'report_period'], inplace=True)
        tot_oper_rev = tot_oper_rev.unstack(level=2)
        tot_oper_rev = tot_oper_rev.loc[:, 'tot_oper_rev']
        tot_oper_rev.reset_index(inplace=True)
        tot_oper_rev.set_index('date', inplace=True)
        codes = tot_oper_rev['code'].unique()
        split_codes = np.array_split(codes, 10)
        with parallel_backend('multiprocessing', n_jobs=4):
            Parallel()(delayed(self.job_factors)(calendar, codes, tot_oper_rev, start, end) for codes in split_codes)


if __name__ == '__main__':
    n = tot_oper_rev_revise()
    n.cal_factors(20100101, 20190901)
