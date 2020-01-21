import pandas as pd
import datetime
import dateutil.parser as dtparser
import os

class tot_change:
    def __init__(self,dt_input = datetime.datetime.now()):
        if isinstance(dt_input,datetime.datetime):
            self.dt = dt_input
        else:
            self.dt = dtparser.parse(str(dt_input))
        self.dir_pre = 'D:/alpha/许帆/'

    def df_to_ini(self, df, fund_id):
        folder_name = self.dir_pre + self.dt.strftime('%Y%m%d')
        if os.path.exists(folder_name):
            pass
        else:
            os.makedirs(folder_name)
        with open(folder_name + '/' + fund_id + '.ini', mode='w', encoding='gbk') as f:
            f.write('[BASKET]\n')
            f.write('Fundid1=' + fund_id + '\n')
            f.write('TAGTAG\n')
            for idx, row in df.iterrows():
                content = row['code']+'|'+row['exchange']+'|'+row['name']+'|'+str(int(row['to_trade_vol']))+'\n'
                f.write(content)
            f.write('ENDENDEND')
        print('%s 生成完毕' % fund_id)

    def run(self,file_path,sheet_name):
        target_vol = pd.read_excel(file_path,sheet_name)
        target_vol['code'] = target_vol['股票代码'].apply(lambda x: x.split('.')[0])
        positions_date = self.dt
        folder_name = positions_date.strftime('%Y%m%d')
        file_name = '综合信息查询_组合证券_S4_' + folder_name + '.xls'
        while True:
            folder_path = self.dir_pre + folder_name
            if os.path.exists(folder_path) and file_name in os.listdir(folder_path):
                break
            else:
                positions_date = positions_date - datetime.timedelta(days=1)
                folder_name = positions_date.strftime('%Y%m%d')
                file_name = '综合信息查询_组合证券_S4_' + folder_name + '.xls'
        print('XF持仓日期: %s' % folder_name)
        positions_path = self.dir_pre + '/' + folder_name + '/' + file_name
        positions = pd.read_excel(positions_path)
        positions = positions.loc[:, ['证券代码', '证券名称', '持仓']].dropna(subset=['证券代码'])
        positions['证券代码'] = positions['证券代码'].astype('int')
        positions['code'] = positions['证券代码'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
        merge = pd.merge(positions, target_vol, how='outer', on='code')
        merge['exchange'] = merge['code'].apply(lambda x: 'SH' if x[0] == '6' else 'SZ')
        merge['name'] = merge.apply(lambda x: x['股票简称'] if pd.notnull(x['股票简称']) else x['证券名称'], axis=1)
        merge['持仓'] = merge['持仓'].fillna(0)
        merge['购买股数'] = merge['购买股数'].fillna(0)
        merge['vol_diff'] = merge['购买股数'] - merge['持仓']
        merge['to_trade_vol'] = merge['vol_diff'].apply(lambda x: round(x,-2))
        change_buy = merge.loc[merge['to_trade_vol'] > 0, ['code', 'name', 'to_trade_vol', 'exchange']]
        change_sell = merge.loc[merge['to_trade_vol'] < 0, ['code', 'name', 'to_trade_vol', 'exchange']]
        change_sell['to_trade_vol'] = change_sell['to_trade_vol'] * -1
        self.df_to_ini(change_buy,'XF_tot_buy')
        self.df_to_ini(change_sell,'XF_tot_sell')


if __name__ == '__main__':
    tot = tot_change()
    tot.run('D:/alpha/许帆/20200120/许帆20200120.xlsx','change')
