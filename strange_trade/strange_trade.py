import pandas as pd
from rdf_data import rdf_data
import datetime
import numpy as np

print(datetime.datetime.now())


def date_forward(date, days):
    raw_days_forward = trading_calendar[trading_calendar > date].iloc[days - 1]
    if raw_days_forward > datetime.datetime(2019, 6, 19):
        return np.nan
    else:
        return raw_days_forward


def fetch_close_price(date):
    if date != date:
        return np.nan
    else:
        date_close = close_df.loc[date.strftime('%Y%m%d'), :]
        if date_close.empty:
            return np.nan
        else:
            return close_df.loc[date, '收盘价']


pd.set_option('display.max_columns', None)
rdf = rdf_data()
trading_calendar = rdf.get_trading_calendar()
df = rdf.get_strange_trade(20180101,20190724)
bool_type_true = df['type'].str.contains('02').fillna(False) | df['type'].str.contains('03').fillna(False) |\
    df['type'].str.contains('05').fillna(False) | df['type'].str.contains('06').fillna(False) |\
    df['type'].str.contains('07').fillna(False)
bool_type_false = df['type'].str.contains('11').fillna(False) | df['type'].str.contains('12').fillna(False) |\
    df['type'].str.contains('14').fillna(False) | df['type'].str.contains('18').fillna(False) |\
    df['type'].str.contains('19').fillna(False) | df['type'].str.contains('20').fillna(False)
bool_huatai = df['trader_name'].str.contains('华泰').fillna(False)
df = df[bool_type_true & ~bool_type_false & bool_huatai]

#df.to_csv('strange_trade.csv',encoding='gbk')
'''
df_list = []
print(len(df['code'].unique()))
#for code in ['601313.SH']:
for code in df['code'].unique():
    print(code)
    sql_sentence = "select s_info_windcode,trade_dt,s_dq_close " \
                   "from wind_filesync.AShareEODPrices " \
                   "where s_info_windcode = :cd and trade_dt > 20160101 and trade_dt < 20190620"
    rdf.curs.execute(sql_sentence,cd=code)
    fetch = rdf.curs.fetchall()
    close_df = pd.DataFrame(fetch, columns=['股票代码', '日期', '收盘价'])
    if close_df.empty:
        continue
    close_df['日期'] = pd.to_datetime(close_df['日期'], format="%Y%m%d")
    close_df.set_index('日期', inplace=True)

    tmp_df = df.loc[df['code']==code,:].copy()
    tmp_df['end_date'] = pd.to_datetime(tmp_df['end_date'])
    tmp_df['3_days_later'] = tmp_df.apply(lambda row: date_forward(row['end_date'],3), axis=1)
    tmp_df['5_days_later'] = tmp_df.apply(lambda row: date_forward(row['end_date'],5), axis=1)
    tmp_df['10_days_later'] = tmp_df.apply(lambda row: date_forward(row['end_date'],10), axis=1)
    tmp_df['20_days_later'] = tmp_df.apply(lambda row: date_forward(row['end_date'],20), axis=1)

    tmp_df['ann_close'] = tmp_df.apply(lambda row: fetch_close_price(row['end_date']),axis=1)
    tmp_df['3_days_close'] = tmp_df.apply(lambda row: fetch_close_price(row['3_days_later']), axis=1)
    tmp_df['5_days_close'] = tmp_df.apply(lambda row: fetch_close_price(row['5_days_later']), axis=1)
    tmp_df['10_days_close'] = tmp_df.apply(lambda row: fetch_close_price(row['10_days_later']), axis=1)
    tmp_df['20_days_close'] = tmp_df.apply(lambda row: fetch_close_price(row['20_days_later']), axis=1)
    df_list.append(tmp_df)

merged_df = pd.concat(df_list)
merged_df.to_csv('strange_trade.csv',encoding='utf8')
print(merged_df)
'''
'''
store = pd.HDFStore('strange_trade.h5')
store.put('strange_trade',merged_df,format='table')
store.close()
'''