from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
from joblib import Parallel, delayed, parallel_backend


class shares_and_turnover(FactorBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def job_upload(codes, df):
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            print('code: %s' % code)
            influx = influxdbData()
            influx.saveData(code_df, 'DailyData_Gus', 'indicators')

    def run(self, start, end):
        query = "select TRADE_DT, S_INFO_WINDCODE, TOT_SHR_TODAY, FLOAT_A_SHR_TODAY, FREE_SHARES_TODAY " \
                "from wind_filesync.AShareEODDerivativeIndicator " \
                "where TRADE_DT >= {0} and TRADE_DT <= {1} " \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '60%') " \
                "order by TRADE_DT, S_INFO_WINDCODE " \
            .format(str(start), str(end))
        self.rdf.curs.execute(query)
        shares = pd.DataFrame(self.rdf.curs.fetchall(),
                              columns=['date', 'code', 'tot_shares', 'float_shares', 'free_shares'])
        shares = shares.loc[
                 pd.notnull(shares['tot_shares']) | pd.notnull(shares['float_shares']) | pd.notnull(
                     shares['free_shares']), :]
        shares.replace(0, np.nan, inplace=True)
        shares = shares.fillna(method='ffill', axis=1)
        shares['date'] = pd.to_datetime(shares['date'])
        print('shares got')
        mkt = self.influx.getDataMultiprocess('DailyData_Gus', 'marketData', start, end, ['code', 'volume'])
        mkt.index.names = ['date']
        mkt.reset_index(inplace=True)
        print('mkt got')
        merge = pd.merge(mkt, shares, on=['date', 'code'])
        merge['turnover'] = merge['volume'] / merge['tot_shares']
        merge['float_turnover'] = merge['volume'] / merge['float_shares']
        merge['free_turnover'] = merge['volume'] / merge['free_shares']
        merge = merge.drop('volume', axis=1)
        merge.set_index('date', inplace=True)
        merge = merge.where(pd.notnull(merge), None)
        codes = merge['code'].unique()
        split_codes = np.array_split(codes, 10)
        with parallel_backend('multiprocessing', n_jobs=4):
            Parallel()(delayed(shares_and_turnover.job_upload)
                       (codes, merge) for codes in split_codes)


if __name__ == '__main__':
    i = shares_and_turnover()
    i.run(20100101, 20190901)
