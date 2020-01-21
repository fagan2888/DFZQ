from factor_base import FactorBase
import pandas as pd
import numpy as np


class market_cap(FactorBase):
    def __init__(self):
        super().__init__()

    def cal_factors(self,start,end):
        query = "select TRADE_DT, S_INFO_WINDCODE, S_VAL_MV, S_DQ_MV, FREE_SHARES_TODAY, S_DQ_CLOSE_TODAY " \
                "from wind_filesync.AShareEODDerivativeIndicator " \
                "where TRADE_DT >= {0} and TRADE_DT <= {1} " \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '60%') " \
                "order by TRADE_DT, S_INFO_WINDCODE " \
            .format(str(start), str(end))

        self.rdf.curs.execute(query)
        mkt_cap = pd.DataFrame(self.rdf.curs.fetchall(),columns=['date','code','market_cap','float_market_cap','free_shares','close'])
        print('raw data got!')
        mkt_cap['free_market_cap'] = mkt_cap['close'] * mkt_cap['free_shares']
        mkt_cap = mkt_cap.loc[pd.notnull(mkt_cap['market_cap'])|pd.notnull(mkt_cap['float_market_cap'])|
                              pd.notnull(mkt_cap['free_market_cap']),:]
        mkt_cap['ln_market_cap'] = np.log(mkt_cap['market_cap'])
        mkt_cap['ln_float_market_cap'] = np.log(mkt_cap['float_market_cap'])
        mkt_cap['ln_free_market_cap'] = np.log(mkt_cap['free_market_cap'])

        mkt_cap['date'] = pd.to_datetime(mkt_cap['date'])
        mkt_cap.set_index('date',inplace=True)
        mkt_cap = mkt_cap.loc[:,['code','market_cap','ln_market_cap','float_market_cap','ln_float_market_cap',
                                 'free_market_cap','ln_free_market_cap']]
        mkt_cap = mkt_cap.where(pd.notnull(mkt_cap), None)
        print('data process finish!')
        self.save_factor_to_influx(mkt_cap, 'DailyFactor_Gus', 'Size')


if __name__ == '__main__':
    mc = market_cap()
    mc.cal_factors(20100101,20200101)