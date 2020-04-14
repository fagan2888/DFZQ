from influxdb_data import influxdbData
from rdf_data import rdf_data
import pandas as pd
import numpy as np
import datetime
import dateutil.parser as dtparser

# 用于填充吸收合并期间的空档期无数据的情况
class FillSwapData:
    def __init__(self):
        pass

    def process_data(self):
        influx = influxdbData()
        rdf = rdf_data()

        calendar = rdf.get_trading_calendar()
        query = "select TRANSFERER_WINDCODE, TARGETCOMP_WINDCODE, CONVERSIONRATIO, ANNCEDATE, LASTTRADEDATE, LISTDATE " \
                "from wind_filesync.AShareStockSwap where PROGRESS = 3 and anncedate >= 20090101 " \
                "and (TRANSFERER_WINDCODE like '0%' or TRANSFERER_WINDCODE like '3%' or TRANSFERER_WINDCODE like '6%') "
        rdf.curs.execute(query)
        swap = pd.DataFrame(rdf.curs.fetchall(),
                            columns=['transfer_code', 'target_code', 'conversion_ratio', 'ann_dt', 'last_trade_dt',
                                     'list_dt'])
        db = 'DailyData_Gus'
        measure = 'marketData'

        columns = ['amount', 'citics_lv1_code', 'citics_lv1_name', 'citics_lv2_code', 'citics_lv2_name',
                   'citics_lv3_code', 'citics_lv3_name', 'close', 'code', 'conversed_ratio', 'high', 'isST',
                   'low', 'open', 'preclose', 'status', 'sw_lv1_code', 'sw_lv1_name', 'sw_lv2_code',
                   'sw_lv2_name', 'swap_code', 'swap_date', 'swap_ratio', 'volume', 'vwap']
        tmp_list = []
        for idx, row in swap.iterrows():
            q = 'select * from "{0}"."autogen"."{1}" where "code" = \'{2}\''.format(db, measure, row['transfer_code'])
            result = influx.client.query(q)
            if not result:
                continue
            else:
                df = pd.DataFrame(result[measure])
                df = df.tz_convert(None)
                df = df.loc[min(row['last_trade_dt'], row['ann_dt']):row['list_dt'], :]
                dates_needed = calendar.loc[(calendar >= dtparser.parse(min(row['last_trade_dt'], row['ann_dt']))) &
                                            (calendar < dtparser.parse(row['list_dt']))]
                dates_needed = set(dates_needed)
                insert_dates = dates_needed - set(df.index)
                if len(insert_dates) == 0:
                    pass
                else:
                    content = [[np.nan] * len(columns)] * len(insert_dates)
                    insert_df = pd.DataFrame(content, columns=columns, index=list(insert_dates))
                    insert_df['code'] = row['transfer_code']
                    insert_df['amount'] = 0
                    insert_df['volume'] = 0
                    insert_df['vwap'] = 0
                    df = df.append(insert_df, ignore_index=False).sort_index()
                    df = df.loc['20100101':, :]
                df.fillna(method='ffill', inplace=True)
                df['isST'] = df['isST'].apply(lambda x: True if x == 1 else False)
                df.loc[row['ann_dt']:, 'swap_code'] = row['target_code']
                df.loc[row['ann_dt']:, 'swap_ratio'] = row['conversion_ratio']
                df.loc[row['ann_dt']:, 'swap_date'] = \
                    (calendar[calendar < dtparser.parse(row['list_dt'])].iloc[-1]).strftime('%Y%m%d')
                tmp_list.append(df)

        merged_df = pd.concat(tmp_list)
        merged_df = merged_df.dropna(subset=['code'])
        merged_df = merged_df.where(pd.notnull(merged_df), None)
        res = influx.saveData(merged_df, db, measure)
        print('fill swap data finish')
        print('-'*30)
        return res


if __name__ == '__main__':
    f = FillSwapData()
    f.process_data()