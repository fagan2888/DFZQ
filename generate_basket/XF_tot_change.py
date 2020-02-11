import pandas as pd
import numpy as np
import datetime
import dateutil.parser as dtparser
import math
import os


class tot_change:
    def __init__(self, dt_input=datetime.datetime.now()):
        if isinstance(dt_input, datetime.datetime):
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
                content = row['code'] + '|' + row['exchange'] + '|' + row['name'] + '|' + \
                          str(int(row['to_trade_vol'])) + '\n'
                f.write(content)
            f.write('ENDENDEND')
        print('%s 生成完毕' % fund_id)

    def split_basket(self, tot_df, split):
        tmp_df = tot_df.copy()
        tmp_df['each_vol'] = np.floor(tmp_df['to_trade_vol'] / split / 100) * 100
        tmp_df['remain_vol'] = tmp_df['to_trade_vol'] - tmp_df['each_vol'] * split
        split_df = tmp_df.loc[:, ['code', 'name', 'each_vol', 'exchange']].copy()
        split_df.rename(columns={'each_vol': 'to_trade_vol'}, inplace=True)
        remain_df = tmp_df.loc[tmp_df['remain_vol'] > 0, ['code', 'name', 'remain_vol', 'exchange']].copy()
        remain_df.rename(columns={'remain_vol': 'to_trade_vol'}, inplace=True)
        return [split_df, remain_df]


    def run(self, file_path, sheet_name, split):
        target_vol = pd.read_excel(file_path, sheet_name)
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
        merge['to_trade_vol'] = merge['vol_diff'].apply(lambda x: round(x, -2))
        change_buy = merge.loc[merge['to_trade_vol'] > 0, ['code', 'name', 'to_trade_vol', 'exchange']]
        change_sell = merge.loc[merge['to_trade_vol'] < 0, ['code', 'name', 'to_trade_vol', 'exchange']]
        change_sell['to_trade_vol'] = change_sell['to_trade_vol'] * -1

        self.df_to_ini(change_buy, 'XF_tot_buy')
        self.df_to_ini(change_sell, 'XF_tot_sell')
        if split == 1:
            pass
        else:
            split_buy, remain_buy = self.split_basket(change_buy, split)
            split_sell, remain_sell = self.split_basket(change_sell, split)
            self.df_to_ini(split_buy, 'XF_split_buy')
            self.df_to_ini(remain_buy, 'XF_remain_buy')
            self.df_to_ini(split_sell, 'XF_split_sell')
            self.df_to_ini(remain_sell, 'XF_remain_sell')


if __name__ == '__main__':
    tot = tot_change()
    tot.run('D:/alpha/许帆/20200211/许帆20200211.xlsx', 'change', 3)
