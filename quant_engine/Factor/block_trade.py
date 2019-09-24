from rdf_data import rdf_data
from Index_comp_sql import IndexCompSQL
import pandas as pd
import datetime
from joblib import Parallel,delayed,parallel_backend

def get_close(code,date):
    r = rdf_data()
    return r.get_stock_close(code,date)

def cal_mv_tr(code,date):
    r = rdf_data()
    oracle_sen = "select S_INFO_WINDCODE, TRADE_DT, S_VAL_MV, S_DQ_MV " \
                 "from wind_filesync.AShareEODDerivativeIndicator " \
                 "where S_INFO_WINDCODE = '{0}' and TRADE_DT <= {1} " \
                 "order by TRADE_DT DESC FETCH FIRST 100 ROWS ONLY".format(code, date.strftime('%Y%m%d'))
    r.curs.execute(oracle_sen)
    mv = pd.DataFrame(r.curs.fetchall(), columns=['code', 'date', 'market_value', 'free_market_value'])
    mv_max = mv['market_value'].max()
    mv_min = mv['market_value'].min()
    mv_0 = mv['market_value'].iloc[0]
    mv_100 = mv['market_value'].iloc[-1]
    free_mv_max = mv['free_market_value'].max()
    free_mv_min = mv['free_market_value'].min()
    free_mv_0 = mv['free_market_value'].iloc[0]
    free_mv_100 = mv['free_market_value'].iloc[-1]
    mv_tr = (mv_0 - mv_100) / (mv_max - mv_min)
    free_mv_tr = (free_mv_0 - free_mv_100) / (free_mv_max - free_mv_min)
    df = pd.DataFrame({'mv_tr':[mv_tr],'free_mv_tr':[free_mv_tr],'mv_0':[mv_0],'free_mv_0':[free_mv_0],
                       'code':[code]},index=[date])
    return df

def cal_20D_amt(code,date):
    r = rdf_data()
    oracle_sen = "select S_INFO_WINDCODE, TRADE_DT, S_DQ_AMOUNT " \
                 "from wind_filesync.AShareEODPrices " \
                 "where S_INFO_WINDCODE = '{0}' and TRADE_DT <= {1} " \
                 "order by TRADE_DT DESC FETCH FIRST 20 ROWS ONLY".format(code, date.strftime('%Y%m%d'))
    r.curs.execute(oracle_sen)
    amt_20d = pd.DataFrame(r.curs.fetchall(),columns=['code','date','amount'])
    amt_avg = amt_20d['amount'].sum()/20
    df = pd.DataFrame({'avg_amt_20':[amt_avg],'code':[code]},index=[date])
    return df

def cal_stk_pct_change(code,date,mode):
    # mode为before时取之前的涨跌幅(含当天)，不为before取之后(不含当天)
    if mode == 'before':
        oper_symbol = '<='
        oracle_sen = "select S_INFO_WINDCODE, TRADE_DT, PCT_CHANGE_D " \
                     "from wind_filesync.AShareYield " \
                     "where S_INFO_WINDCODE = '{}' and TRADE_DT {} {} " \
                     "order by TRADE_DT DESC FETCH FIRST 20 ROWS ONLY".format(code, oper_symbol, date.strftime('%Y%m%d'))
    else:
        oper_symbol = '>'
        oracle_sen = "select S_INFO_WINDCODE, TRADE_DT, PCT_CHANGE_D " \
                     "from wind_filesync.AShareYield " \
                     "where S_INFO_WINDCODE = '{}' and TRADE_DT {} {} " \
                     "order by TRADE_DT FETCH FIRST 20 ROWS ONLY".format(code, oper_symbol, date.strftime('%Y%m%d'))
    r = rdf_data()
    r.curs.execute(oracle_sen)
    pct_change = pd.DataFrame(r.curs.fetchall(),columns=['code','date','pct_change_d'])
    pct_change_10 = ((1+pct_change['pct_change_d'].iloc[0:10]/100).prod()-1)*100
    pct_change_20 = ((1+pct_change['pct_change_d']/100).prod()-1)*100
    col_10 = 'stk_pct_change_10_'+ mode
    col_20 = 'stk_pct_change_20_'+ mode
    df = pd.DataFrame({col_10:[pct_change_10],col_20:[pct_change_20],'code':[code]},index=[date])
    return df

def cal_idx_pct_change(code,date,mode):
    # mode为before时取之前的涨跌幅(含当天)，不为before取之后(不含当天)
    if mode == 'before':
        oper_symbol = '<='
        oracle_sen = "select S_INFO_WINDCODE, TRADE_DT, S_DQ_PCTCHANGE  " \
                     "from wind_filesync.AIndexEODPrices " \
                     "where S_INFO_WINDCODE = '{}' and TRADE_DT {} {} " \
                     "order by TRADE_DT DESC FETCH FIRST 20 ROWS ONLY".format(code, oper_symbol, date)
    else:
        oper_symbol = '>'
        oracle_sen = "select S_INFO_WINDCODE, TRADE_DT, S_DQ_PCTCHANGE " \
                     "from wind_filesync.AIndexEODPrices " \
                     "where S_INFO_WINDCODE = '{}' and TRADE_DT {} {} " \
                     "order by TRADE_DT FETCH FIRST 20 ROWS ONLY".format(code, oper_symbol, date)
    r = rdf_data()
    r.curs.execute(oracle_sen)
    pct_change = pd.DataFrame(r.curs.fetchall(), columns=['code', 'date', 'pct_change_d'])
    pct_change_10 = ((1 + pct_change['pct_change_d'].iloc[0:10] / 100).prod() - 1) * 100
    pct_change_20 = ((1 + pct_change['pct_change_d'] / 100).prod() - 1) * 100
    col_10 = 'idx_pct_change_10_' + mode
    col_20 = 'idx_pct_change_20_' + mode
    df = pd.DataFrame({col_10: [pct_change_10], col_20:[pct_change_20], 'code': [code]}, index=[date])
    df.index = pd.to_datetime(df.index)
    return df



if __name__ == '__main__':
    rdf = rdf_data()
    idx_cmp = IndexCompSQL()
    start = 20100101
    end = 20190815
    index = 300

    print(datetime.datetime.now())

    block_trade = rdf.get_block_trade(start,end)
    unique_date = block_trade['date'].unique()
    block_trade['date'] = pd.to_datetime(block_trade['date'])
    block_trade.set_index(['date','code'],inplace=True)
    idx_cmp = idx_cmp.get_IndexComp(index,start,end)
    idx_cmp.set_index([idx_cmp.index,'code'],inplace=True)

    block_trade = block_trade.loc[block_trade.index.isin(idx_cmp.index),:]
    unique_date_code = list(block_trade.index.unique())
    print('block trade data got!')

    # 获取收盘价
    with parallel_backend('multiprocessing', n_jobs=-1):
        c_list = Parallel()(delayed(get_close)(code,date)
                                 for date,code in unique_date_code)
    close = pd.concat(c_list)
    close.columns = ['date','close']
    close.set_index(['date',close.index],inplace=True)
    print('close data got!')
    
    # 计算market value的true range
    with parallel_backend('multiprocessing', n_jobs=-1):
        tr_list = Parallel()(delayed(cal_mv_tr)(code,date)
                                 for date,code in unique_date_code)
    true_range = pd.concat(tr_list)
    true_range.set_index([true_range.index,'code'],inplace=True)
    print('true range got!')
    
    # 获取20日成交额
    with parallel_backend('multiprocessing', n_jobs=-1):
        amt_list = Parallel()(delayed(cal_20D_amt)(code, date)
                             for date, code in unique_date_code)
    avg_amt = pd.concat(amt_list)
    avg_amt.set_index([avg_amt.index,'code'],inplace=True)
    print('avg amount got!')

    # 获取历史stk涨幅
    with parallel_backend('multiprocessing', n_jobs=-1):
        stk_pct_list = Parallel()(delayed(cal_stk_pct_change)(code, date,'before')
                              for date, code in unique_date_code)
    stk_pct_before = pd.concat(stk_pct_list)
    stk_pct_before.set_index([stk_pct_before.index, 'code'], inplace=True)

    # 获取未来stk涨幅
    with parallel_backend('multiprocessing', n_jobs=-1):
        stk_pct_list = Parallel()(delayed(cal_stk_pct_change)(code, date, 'after')
                                 for date, code in unique_date_code)
    stk_pct_after = pd.concat(stk_pct_list)
    stk_pct_after.set_index([stk_pct_after.index, 'code'], inplace=True)
    print('stock pct change got!')

    # 整合以上字段
    block_trade['close'] = close
    block_trade['mv_tr'] = true_range['mv_tr']
    block_trade['free_mv_tr'] = true_range['free_mv_tr']
    block_trade['mv_0'] = true_range['mv_0']
    block_trade['free_mv_0'] = true_range['free_mv_0']
    block_trade['avg_amt_20'] = avg_amt
    block_trade['stk_pct_10_before'] = stk_pct_before['stk_pct_change_10_before']
    block_trade['stk_pct_20_before'] = stk_pct_before['stk_pct_change_20_before']
    block_trade['stk_pct_10_after'] = stk_pct_after['stk_pct_change_10_after']
    block_trade['stk_pct_20_after'] = stk_pct_after['stk_pct_change_20_after']
    block_trade = block_trade.reset_index()
    block_trade.set_index('date',inplace=True)
    print(datetime.datetime.now())

    # 获取历史指数涨幅
    with parallel_backend('multiprocessing', n_jobs=-1):
        idx_pct_list = Parallel()(delayed(cal_idx_pct_change)('000300.SH', date, 'before')
                                     for date in unique_date)
    idx_pct_before = pd.concat(idx_pct_list)

    # 获取未来指数涨幅
    with parallel_backend('multiprocessing', n_jobs=-1):
        idx_pct_list = Parallel()(delayed(cal_idx_pct_change)('000300.SH', date, 'after')
                                     for date in unique_date)
    idx_pct_after = pd.concat(idx_pct_list)
    print('index pct change got!')

    # 整合以上字段
    block_trade['idx_pct_change_10_before'] = idx_pct_before['idx_pct_change_10_before']
    block_trade['idx_pct_change_20_before'] = idx_pct_before['idx_pct_change_20_before']
    block_trade['idx_pct_change_10_after'] = idx_pct_after['idx_pct_change_10_after']
    block_trade['idx_pct_change_20_after'] = idx_pct_after['idx_pct_change_20_after']

    print(datetime.datetime.now())
    block_trade.to_csv('aaa.csv',encoding='gbk')