from factor_base import FactorBase
import pandas as pd
import numpy as np
from influxdb_data import influxdbData
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, parallel_backend

class shares(FactorBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def job_upload(codes,df):
        for code in codes:
            code_df = df.loc[df['code']==code,:].copy()
            print('code: %s' % code)
            influx = influxdbData()
            influx.saveData(code_df, 'DailyData_Gus', 'indicators')

    def run(self, start, end):
        q_start = (dtparser.parse(str(start)) - relativedelta(years=1)).strftime('%Y%m%d')
        query = "select TRADE_DT, S_INFO_WINDCODE, TOT_SHR_TODAY, FLOAT_A_SHR_TODAY, FREE_SHARES_TODAY " \
                "from wind_filesync.AShareEODDerivativeIndicator " \
                "where TRADE_DT >= {0} and TRADE_DT <= {1} " \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '60%') " \
                "order by TRADE_DT, S_INFO_WINDCODE " \
            .format(q_start, str(end))
        self.rdf.curs.execute(query)
        indi = pd.DataFrame(self.rdf.curs.fetchall(),
                            columns=['date', 'code', 'tot_shares', 'float_shares', 'free_shares'])
        print('raw_data got')
        indi = indi.loc[
               pd.notnull(indi['tot_shares']) | pd.notnull(indi['float_shares']) | pd.notnull(indi['free_shares']), :]
        indi.replace(0, np.nan, inplace=True)
        indi = indi.fillna(method='ffill', axis=1)
        indi['date'] = pd.to_datetime(indi['date'])
        indi.set_index('date',inplace=True)
        indi = indi.where(pd.notnull(indi), None)
        print('data process finish')
        codes = indi['code'].unique()
        split_codes = np.array_split(codes, 10)
        with parallel_backend('multiprocessing', n_jobs=4):
            Parallel()(delayed(shares.job_upload)
                       (codes, indi) for codes in split_codes)


if __name__ == '__main__':
    i = shares()
    i.run(20100101, 20190901)
