from rdf_data import rdf_data
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS

class shares_turnover:
    def __init__(self):
        self.rdf = rdf_data()
        self.influx = influxdbData()
        self.db = 'DailyMarket_Gus'
        self.measure = 'shares_turnover'

    def process_data(self, start, end, n_jobs):
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
                 pd.notnull(shares['tot_shares']) | pd.notnull(shares['float_shares']) |
                 pd.notnull(shares['free_shares']), :]
        shares.replace(0, np.nan, inplace=True)
        shares[['tot_shares', 'float_shares', 'free_shares']] = \
            shares[['tot_shares', 'float_shares', 'free_shares']].fillna(method='ffill', axis=1)
        shares['date'] = pd.to_datetime(shares['date'])
        shares = shares.groupby(['date', 'code']).last()
        shares = shares.reset_index()
        mkt = self.influx.getDataMultiprocess('DailyMarket_Gus', 'market', start, end, ['code', 'volume'])
        mkt.index.names = ['date']
        mkt.reset_index(inplace=True)
        merge = pd.merge(mkt, shares, on=['date', 'code'])
        merge['turnover'] = merge['volume'] / merge['tot_shares']
        merge['float_turnover'] = merge['volume'] / merge['float_shares']
        merge['free_turnover'] = merge['volume'] / merge['free_shares']
        merge = merge.drop('volume', axis=1)
        merge = merge.set_index('date')
        merge = merge.where(pd.notnull(merge), None)
        codes = merge['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (merge, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('shares and turnover finish...')
        print('-'*30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    i = shares_turnover()
    i.process_data(20100101, 20200401, N_JOBS)
