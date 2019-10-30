from influxdb_data import influxdbData
from rdf_data import rdf_data
import pandas as pd
import numpy as np
import datetime
import dateutil.parser as dtparser

class UpdateSwapData:
    def __init__(self):
        self.rdf = rdf_data()
        self.influx = influxdbData()
        self.db = 'DailyData_backtest'
        self.msr = 'marketData'

    def run(self,start_input,end_input):
        calendar = self.rdf.get_trading_calendar()
        rdf_query = 'select TRANSFERER_WINDCODE, TARGETCOMP_WINDCODE, CONVERSIONRATIO, ANNCEDATE, LASTTRADEDATE, LISTDATE ' \
                'from wind_filesync.AShareStockSwap where PROGRESS = 3 and anncedate >= 20090101'
        self.rdf.curs.execute(rdf_query)
        swap = pd.DataFrame(self.rdf.curs.fetchall(),
                            columns=['transfer_code','target_code','conversion_ratio','ann_dt','last_trade_dt','list_dt'])
        swap = swap.loc[(swap['ann_dt']<=str(end_input))&(swap['list_dt']>=str(start_input))&
                        (swap['last_trade_dt']>='20100101'),:]
        swap = swap.loc[(swap['transfer_code'].str.get(0)=='0')|(swap['transfer_code'].str.get(0)=='6'),:]
        swap['ann_dt'] = pd.to_datetime(swap['ann_dt'])
        swap['last_trade_dt'] = pd.to_datetime(swap['last_trade_dt'])
        swap['list_dt'] = pd.to_datetime(swap['list_dt'])
        to_merge_list = []
        for idx,row in swap.iterrows():
            start = min(row['ann_dt'],row['last_trade_dt'])-datetime.timedelta(hours=8)
            start = int(start.timestamp() * 1000 * 1000* 1000)
            end = row['list_dt']-datetime.timedelta(hours=8)
            end = int(end.timestamp() * 1000 * 1000* 1000)
            print(row)
            influx_query = 'select * from "{0}"."autogen"."{1}" where time>={2} and time<{3} and "code"=\'{4}\''.\
                format(self.db,self.msr,start,end,row['transfer_code'])
            result = self.influx.client.query(influx_query)
            stk_swap = pd.DataFrame(result[self.msr])
            stk_swap.loc[row['ann_dt']:,'swap_ratio'] = row['conversion_ratio']
            stk_swap.loc[row['ann_dt']:,'swap_code'] = row['target_code']
            stk_swap.loc[row['ann_dt']:,'swap_date'] = calendar[calendar<row['list_dt']].iloc[-1]
            to_merge_list.append(stk_swap.loc[:,['code','swap_ratio','swap_code','swap_date']])
        merged_df = pd.concat(to_merge_list)
        merged_df = merged_df.dropna()
        merged_df['swap_date'] = merged_df['swap_date'].apply(lambda x: x.strftime('%Y%m%d'))

        self.influx.saveData(merged_df,'DailyData_Gus','marketData')


if __name__ == '__main__':
    us = UpdateSwapData()
    us.run(20100101,20190901)
