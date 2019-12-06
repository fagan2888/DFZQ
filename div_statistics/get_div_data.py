import cx_Oracle as oracle
import pandas as pd
from configparser import ConfigParser
import datetime

def get_constituent_list(date,type):
    cfg = ConfigParser(allow_no_value=True)
    path_prefix = 'C:/Users/Administrator/Desktop/期现套利/篮子配置/'
    if type == 50:
        postfix = 'A001.ini'
    elif type == 300:
        postfix = 'A002stk.ini'
    elif type == 500:
        postfix = 'A003_x3.ini'
    else:
        print('type error!')
        return
    path = path_prefix + str(date)[-4:] + '/' + postfix
    print(path)
    cfg.read(path)
    raw_list = cfg.items('BASKET')[2:]
    raw_list.remove(('endendend', ''))
    p_list = []
    for d in raw_list:
        p_list.append(d[0].split('|')[0]+'.'+(d[0].split('|')[1]).upper())
    return p_list

def add_stock_name(df):
    if len(df)>1:
        code_str = str(tuple(df['股票代码']))
    elif len(df)== 1:
        code_str = "('" + df.loc[0,'股票代码'] + "')"
    else:
        df['股票名称'] = None
        return df
    get_name_sentense = 'select s_info_windcode,s_info_name ' \
                        'from wind_filesync.AShareDescription ' \
                        'where s_info_windcode in '
    get_name_sentense = get_name_sentense + code_str
    curs.execute(get_name_sentense)
    code_name = curs.fetchall()

    name_dict = {}
    for pair in code_name:
        name_dict[pair[0]] = pair[1]

    for i in range(len(df)):
        df.loc[i,'股票名称'] = name_dict[df.loc[i,'股票代码']]

    return df


def get_div_df(dt_input):
    date = str(dt_input)
    # 每股转送：STK_DVD_PER_SH，税前派息：CASH_DVD_PER_SH_PRE_TAX，税后派息：CASH_DVD_PER_SH_AFTER_TAX，
    # 方案进度（只选3）：S_DIV_PROGRESS，股权登记日：EQY_RECORD_DT，
    # 除权除息日：EX_DT，派息日：DVD_PAYOUT_DT，分红实施公告日：DVD_ANN_DT
    sql_sentense = 'select s_info_windcode,STK_DVD_PER_SH,CASH_DVD_PER_SH_PRE_TAX,CASH_DVD_PER_SH_AFTER_TAX,' \
                   'S_DIV_PROGRESS,EQY_RECORD_DT,EX_DT,DVD_PAYOUT_DT,DVD_ANN_DT ' \
                   'from wind_filesync.AShareDividend '\
                   'where s_div_progress <= 3 and eqy_record_dt >= :dt ' \
                   'order by eqy_record_dt'

    curs.execute(sql_sentense,dt=date)
    kk = curs.fetchall()
    div_df = pd.DataFrame(kk,columns=['股票代码','每股转送','税前派息','税后派息','方案进度','股权登记日','除权除息日',
                                   '派息日','分红实施公告日'])
    div_df = add_stock_name(div_df)
    print(div_df)
    return div_df.loc[:,['股票代码','股票名称','每股转送','税前派息','税后派息','方案进度','股权登记日','除权除息日',
                         '派息日','分红实施公告日']]


def get_right_issue_df(date_input):
    sql_sentense = 'select s_info_windcode,S_RIGHTSISSUE_PROGRESS,S_RIGHTSISSUE_PRICE,S_RIGHTSISSUE_RATIO,' \
                   'S_RIGHTSISSUE_AMOUNT,S_RIGHTSISSUE_AMOUNTACT,S_RIGHTSISSUE_NETCOLLECTION,S_RIGHTSISSUE_REGDATESHAREB,' \
                   'S_RIGHTSISSUE_EXDIVIDENDDATE,S_RIGHTSISSUE_LISTEDDATE,S_RIGHTSISSUE_PAYSTARTDATE,' \
                   'S_RIGHTSISSUE_PAYENDDATE,S_RIGHTSISSUE_ANNCEDATE,S_RIGHTSISSUE_RESULTDATE,S_RIGHTSISSUE_YEAR ' \
                   'from wind_filesync.AShareRightIssue '\
                   'where S_RIGHTSISSUE_PROGRESS <= 3 and S_RIGHTSISSUE_REGDATESHAREB >= :dt ' \
                   'order by S_RIGHTSISSUE_REGDATESHAREB'

    curs.execute(sql_sentense,dt=date_input)
    qq = curs.fetchall()
    right_issue_df = pd.DataFrame(qq,columns=['股票代码','方案进度','配股价格','配股比例','配股计划数量（万股）',
                                              '配股实际数量（万股）','募集资金（元）','股权登记日','除权日','配股上市日',
                                              '缴款起始日','缴款终止日','配股实施公告日','配股结果公告日','配股年度'])

    right_issue_df = add_stock_name(right_issue_df)
    return right_issue_df.loc[:,['股票代码','股票名称','方案进度','配股价格','配股比例','配股计划数量（万股）',
                                 '配股实际数量（万股）','募集资金（元）','股权登记日','除权日','配股上市日',
                                 '缴款起始日','缴款终止日','配股实施公告日','配股结果公告日','配股年度']]


def filtering_and_writing(div,right_issue,dt):
    filename = '转送配股分红统计_更新至' + str(dt) + '.xls'
    writer = pd.ExcelWriter(filename)

    index_300 = []
    index_500 = []
    for i in range(len(div)):
        if div.loc[i,'股票代码'] in list_300:
            index_300.append(i)
        if div.loc[i, '股票代码'] in list_500:
            index_500.append(i)

    div.loc[index_300,:].reset_index(drop=True).to_excel(writer,sheet_name="300成分分红")
    div.loc[index_500,:].reset_index(drop=True).to_excel(writer,sheet_name="500成分分红")


    index_300 = []
    index_500 = []
    for i in range(len(right_issue)):
        if right_issue.loc[i, '股票代码'] in list_300:
            index_300.append(i)
        if right_issue.loc[i, '股票代码'] in list_500:
            index_500.append(i)

    right_issue.loc[index_300,:].reset_index(drop=True).to_excel(writer,sheet_name='300成分配股')
    right_issue.loc[index_500,:].reset_index(drop=True).to_excel(writer,sheet_name='500成分配股')
    writer.save()


if __name__ == "__main__":
    pd.set_option("display.max_columns",None)

    date_input = 20190412
    dt_input = datetime.datetime.strptime(str(date_input), "%Y%m%d")
    another_date = 20190304

    ip_address = '172.17.21.3'
    port = '1521'
    username = 'yspread'
    pwd = 'Y*iaciej123456'
    conn = oracle.connect(username + "/" + pwd + "@" + ip_address + ":" + port + "/" + "WDZX")
    curs = conn.cursor()

    list_50 = get_constituent_list(date_input, 50)
    list_300 = get_constituent_list(date_input, 300)
    list_500 = get_constituent_list(date_input, 500)

    div_df = get_div_df(another_date)
    right_issue_df = get_right_issue_df(another_date)

    filtering_and_writing(div_df,right_issue_df,date_input)