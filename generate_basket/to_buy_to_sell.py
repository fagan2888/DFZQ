import pandas as pd
import configparser
import datetime

def cal_tobuy_tosell(positions_file,basket_file,ftrs_per_bsk,ftrs_vol):
    yyyymmdd = datetime.datetime.now().strftime('%Y%m%d')
    positions = pd.read_excel(positions_file)
    positions = positions.loc[:,['证券代码','证券名称','持仓']].dropna(subset=['证券代码'])
    positions['证券代码'] = positions['证券代码'].astype('int')
    positions['证券代码'] = positions['证券代码'].apply(lambda x: '0'*(6-len(str(x)))+str(x))

    f = open(basket_file,'r')
    line_list = []
    for line in f.readlines():
        if line[0] == '0' or line[0] == '3' or line[0] == '6':
            line = line.strip()
            splitted_line = line.split('|')
            line_list.append([splitted_line[0],splitted_line[2],splitted_line[3]])
        else:
            continue
    basket = pd.DataFrame(line_list)
    basket.columns = ['证券代码','证券名称','篮子配置数']
    basket['篮子配置数'] = basket['篮子配置数'].astype('int')* ftrs_vol/ ftrs_per_bsk

    positions.set_index('证券代码',inplace=True)
    basket.set_index('证券代码',inplace=True)
    merged_df = positions.join(basket['篮子配置数'],how='outer')
    merged_df.loc[pd.isnull(merged_df['证券名称']),'证券名称'] = basket['证券名称']
    merged_df = merged_df.fillna(0)
    merged_df['补买补卖'] = merged_df['篮子配置数']-merged_df['持仓']
    merged_df.to_csv('to_buy_to_sell_1021.csv',encoding='gbk')

    with open('tobuytosell_'+yyyymmdd+'.ini',mode='w',encoding='gbk') as f:
        f.write('[BASKET]\n')
        f.write('Fundid1=' + 'tobuytosell_'+yyyymmdd+'.ini' + '\n')
        f.write('TAGTAG\n')
        for idx,row in merged_df.iterrows():
            exchange = 'SH' if idx[0]=='6' else 'SZ'
            f.write(idx+'|'+exchange+'|'+row['证券名称']+'|'+str(int(round(row['补买补卖'],-2)))+'\n')
        f.write('ENDENDEND')

if __name__ == '__main__':
    cal_tobuy_tosell("综合信息查询_组合证券_ZL_1021.xls","ZL_bsk.ini",3,42)