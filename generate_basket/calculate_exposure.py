import pandas as pd
from rdf_data import  rdf_data
import datetime

def cal_exposure(date,ftrs_per_bsk,ftrs_vol):
    local_dir = 'D:/alpha/张黎/' + str(date) + '/'
    positions = pd.read_excel(local_dir+'综合信息查询_组合证券_ZL_'+str(date)+'.xls')
    positions = positions.loc[:, ['证券代码', '证券名称', '持仓']].dropna(subset=['证券代码'])
    positions['证券代码'] = positions['证券代码'].astype('int')
    positions['证券代码'] = positions['证券代码'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
    positions['code'] = positions['证券代码'].apply(lambda x: x + '.SH' if x[0] == '6' else x + '.SZ')
    positions['持仓'] = positions['持仓'].astype('int')

    rdf = rdf_data()
    codes = positions['code'].tolist()
    query = "select S_INFO_WINDCODE,TRADE_DT,S_DQ_CLOSE " \
            "from wind_filesync.AShareEODPrices " \
            "where trade_dt = {0} and s_info_windcode in ".format(date) + str(tuple(codes))
    rdf.curs.execute(query)
    pos_close = pd.DataFrame(rdf.curs.fetchall(),columns=['code','date','close'])
    positions = positions.merge(pos_close, right_on='code', left_on='code', how='outer')
    print(positions.loc[pd.isnull(positions['close']),:])
    pos_total_amount = (positions['持仓']*positions['close']).sum()

    f = open(local_dir+'ZL_bsk.ini', 'r')
    line_list = []
    for line in f.readlines():
        if line[0] == '0' or line[0] == '3' or line[0] == '6':
            line = line.strip()
            splitted_line = line.split('|')
            line_list.append([splitted_line[0], splitted_line[2], splitted_line[3]])
        else:
            continue
    basket = pd.DataFrame(line_list)
    print('.')


if __name__ == '__main__':
    cal_exposure(20191021,3,42)