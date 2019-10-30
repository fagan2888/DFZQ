from influxdb_data import influxdbData
from rdf_data import rdf_data
import pandas as pd
import numpy as np
import datetime
import dateutil.parser as dtparser

def SwapDataProcess():
    influx = influxdbData()
    rdf = rdf_data()

    calendar = rdf.get_trading_calendar()
    query = 'select TRANSFERER_WINDCODE, TARGETCOMP_WINDCODE, CONVERSIONRATIO, ANNCEDATE, LASTTRADEDATE, LISTDATE ' \
                'from wind_filesync.AShareStockSwap where PROGRESS = 3 and anncedate >= 20090101'
    rdf.curs.execute(query)
    swap = pd.DataFrame(rdf.curs.fetchall(),
                        columns=['transfer_code', 'target_code', 'conversion_ratio', 'ann_dt', 'last_trade_dt', 'list_dt'])
    codes = swap['transfer_code'].unique()
    db = 'DailyData_Gus'
    measure = 'marketData'
    tmp_list = []
    for code in codes:
        q = 'select * from "{0}"."autogen"."{1}" where "code" = \'{2}\''.format(db,measure,code)
        result = influx.client.query(q)
        if not result:
            continue
        else:
            tmp_list.append(pd.DataFrame(result[measure]))

    merged_df = pd.concat(tmp_list)
    merged_df = merged_df.loc[pd.notnull(merged_df['swap_code']),:]
    merged_df['swap_date'] = pd.to_datetime(merged_df['swap_date'])
    merged_df = merged_df.tz_convert(None)
    df_columns = merged_df.columns
    codes = merged_df['code'].unique()

    result_df_list = []
    for code in codes:
        df = merged_df.loc[merged_df['code']==code,:]
        swap_dates = df['swap_date'].unique()
        for swap_date in swap_dates:
            if swap_date in df.index:
                continue
            else:
                tmp_calendar = calendar[calendar>df.index[-1]]
                insert_dates = tmp_calendar[tmp_calendar<=swap_date]
                number_of_dates = insert_dates.shape[0]
                content = [[np.nan]*len(df_columns)] * number_of_dates
                insert_df = pd.DataFrame(content,columns=df_columns,index=insert_dates)
                df = df.append(insert_df,ignore_index=False)
                df.fillna(method='ffill',inplace=True)
                df['isST'] = df['isST'].apply(lambda x: True if x==1 else False)
                result_df_list.append(df)
    result_df = pd.concat(result_df_list)
    result_df = result_df.where(pd.notnull(result_df), None)
    result_df['swap_date'] = result_df['swap_date'].apply(lambda x: x.strftime('%Y%m%d'))
    influx.saveData(result_df,db,measure)


if __name__ == '__main__':
    SwapDataProcess()