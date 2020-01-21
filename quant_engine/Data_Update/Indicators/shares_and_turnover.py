from rdf_data import rdf_data
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
from joblib import Parallel, delayed, parallel_backend


class shares_and_turnover:
    def __init__(self):
        self.rdf = rdf_data()
        self.influx = influxdbData()
        self.db = 'DailyData_Gus'
        self.measure = 'indicators'

    def run(self, start, end, n_jobs):
        query = "select TRADE_DT, S_INFO_WINDCODE, TOT_SHR_TODAY, FLOAT_A_SHR_TODAY, FREE_SHARES_TODAY " \
                "from wind_filesync.AShareEODDerivativeIndicator " \
                "where TRADE_DT >= {0} and TRADE_DT <= {1} " \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '6%') " \
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
        shares = shares.groupby(['date', 'code']).last()
        shares = shares.reset_index()
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
        merge = merge.groupby(['date', 'code']).last()
        merge = merge.reset_index().set_index('date')
        merge = merge.where(pd.notnull(merge), None)
        codes = merge['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            Parallel()(delayed(influxdbData.JOB_saveData)
                       (merge, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('shares and turnover finish...')
        print('-'*30)


if __name__ == '__main__':
    i = shares_and_turnover()
    i.run(20190801, 20200201, 4)
