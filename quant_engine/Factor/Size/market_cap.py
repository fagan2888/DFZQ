from factor_base import FactorBase
import pandas as pd
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData

class market_cap(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'Size'

    def cal_factors(self, start, end, n_jobs):
        query = "select TRADE_DT, S_INFO_WINDCODE, S_VAL_MV, S_DQ_MV, FREE_SHARES_TODAY, S_DQ_CLOSE_TODAY " \
                "from wind_filesync.AShareEODDerivativeIndicator " \
                "where TRADE_DT >= {0} and TRADE_DT <= {1} " \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "order by TRADE_DT, S_INFO_WINDCODE " \
            .format(str(start), str(end))
        self.rdf.curs.execute(query)
        mkt_cap = pd.DataFrame(self.rdf.curs.fetchall(),
                               columns=['date', 'code', 'market_cap', 'float_market_cap', 'free_shares', 'close'])
        mkt_cap['free_market_cap'] = mkt_cap['close'] * mkt_cap['free_shares']
        mkt_cap = mkt_cap.loc[pd.notnull(mkt_cap['market_cap']) | pd.notnull(mkt_cap['float_market_cap']) |
                              pd.notnull(mkt_cap['free_market_cap']), :]
        mkt_cap['ln_market_cap'] = np.log(mkt_cap['market_cap'])
        mkt_cap['ln_float_market_cap'] = np.log(mkt_cap['float_market_cap'])
        mkt_cap['ln_free_market_cap'] = np.log(mkt_cap['free_market_cap'])
        mkt_cap['date'] = pd.to_datetime(mkt_cap['date'])
        mkt_cap.set_index('date', inplace=True)
        mkt_cap = mkt_cap.loc[:, ['code', 'market_cap', 'ln_market_cap', 'float_market_cap', 'ln_float_market_cap',
                                  'free_market_cap', 'ln_free_market_cap']]
        mkt_cap = mkt_cap.where(pd.notnull(mkt_cap), None)
        codes = mkt_cap['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (mkt_cap, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('size data finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    mc = market_cap()
    mc.cal_factors(20100101, 20200204, 5)