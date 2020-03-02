from FTP_service import FTP_service
import datetime
import os.path
import dateutil.parser as dtparser
import pandas as pd
import numpy as np


class basket_trade:
    def __init__(self, dt_input=datetime.datetime.now()):
        self.ftp = FTP_service(host='192.168.38.213', username='index', password='dfzq1234')
        self.remote_dir = '/产品管理/每日跟踪/'
        if isinstance(dt_input, datetime.datetime):
            self.dt = dt_input
        else:
            self.dt = dtparser.parse(str(dt_input))
        self.yyyymmdd = self.dt.strftime('%Y%m%d')
        self.root_dir = 'D:/alpha/'
        self.ZL_path = '{0}{1}/{2}/'.format(self.root_dir, '张黎', self.yyyymmdd)
        if os.path.exists(self.ZL_path):
            pass
        else:
            os.makedirs(self.ZL_path.rstrip('/'))
        self.XF_path = '{0}{1}/{2}/'.format(self.root_dir, '许帆', self.yyyymmdd)
        if os.path.exists(self.XF_path):
            pass
        else:
            os.makedirs(self.XF_path.rstrip('/'))

    # ZL 专用
    def get_open_close_bsk(self):
        # 下载上一日DayCc作为平仓篮子
        self.ftp.ftp.cwd(self.remote_dir)
        filelist = self.ftp.ftp.nlst()
        last_trade_day = self.dt - datetime.timedelta(days=1)
        filename = last_trade_day.strftime('%Y%m%d') + '_DayCc.txt'
        while not filename in filelist:
            last_trade_day = last_trade_day - datetime.timedelta(days=1)
            filename = last_trade_day.strftime('%Y%m%d') + '_DayCc.txt'
        remote_path = self.remote_dir + filename
        download_path = self.ZL_path + filename
        self.ftp.download_file(remote_path=remote_path, local_path=download_path)
        # 生成平仓篮子文件
        self.txt_to_ini(filename, 'ZL_bsk_pc')
        # 下载当日DayCc作为开仓篮子
        filename = self.yyyymmdd + '_DayCc.txt'
        remote_path = self.remote_dir + filename
        download_path = self.ZL_path + filename
        self.ftp.download_file(remote_path=remote_path, local_path=download_path)
        # 生成开仓篮子文件
        self.txt_to_ini(filename, 'ZL_bsk_kc')

    def get_change_bsk(self, split):
        self.ZL_per_bsk = int(input('请输入ZL每个篮子对应的期货手数:'))
        self.ZL_ftrs = int(input('请输入ZL开仓的期货手数:'))
        self.XF_per_bsk = int(input('请输入XF每个篮子对应的期货手数:'))
        self.XF_ftrs = int(input('请输入XF开仓的期货手数:'))
        # 读取持仓ZL
        ZL_positions = self.read_positions('张黎')
        # 读取持仓XF
        XF_positions = self.read_positions('许帆')
        # 读取最新篮子ZL
        file_path = self.ZL_path + 'ZL_bsk_kc.ini'
        ZL_bsk = self.ini_to_df(file_path)
        ZL_bsk['target_vol'] = ZL_bsk['vol_per_bsk'] * self.ZL_ftrs / self.ZL_per_bsk
        # 读取最新篮子XF
        latest_bsk_dt = self.dt
        file_name = 'X1.ini'
        while True:
            folder_path = self.root_dir + '许帆/{0}/'.format(latest_bsk_dt.strftime('%Y%m%d'))
            if os.path.exists(folder_path) and file_name in os.listdir(folder_path):
                break
            else:
                latest_bsk_dt = latest_bsk_dt - datetime.timedelta(days=1)
        print('XF篮子更新日期: %s' % latest_bsk_dt.strftime('%Y%m%d'))
        file_path = folder_path + '/' + file_name
        XF_bsk = self.ini_to_df(file_path)
        XF_bsk['target_vol'] = XF_bsk['vol_per_bsk'] * self.XF_ftrs / self.XF_per_bsk
        # 自行计算调仓
        ZL_change_buy, ZL_change_sell = self.process_change_bsk(ZL_positions, ZL_bsk)
        XF_change_buy, XF_change_sell = self.process_change_bsk(XF_positions, XF_bsk)
        self.df_to_ini(ZL_change_buy, '张黎', 'ZL_change_buy')
        self.df_to_ini(ZL_change_sell, '张黎', 'ZL_change_sell')
        self.df_to_ini(XF_change_buy, '许帆', 'XF_change_buy')
        self.df_to_ini(XF_change_sell, '许帆', 'XF_change_sell')
        if split == 1:
            pass
        else:
            ZL_split_change_buy, ZL_remain_change_buy = self.split_basket(ZL_change_buy, split)
            ZL_split_change_sell, ZL_remain_change_sell = self.split_basket(ZL_change_sell, split)
            XF_split_change_buy, XF_remain_change_buy = self.split_basket(XF_change_buy, split)
            XF_split_change_sell, XF_remain_change_sell = self.split_basket(XF_change_sell, split)
            self.df_to_ini(ZL_split_change_buy, '张黎', 'ZL_split_change_buy')
            self.df_to_ini(ZL_remain_change_buy, '张黎', 'ZL_remain_change_buy')
            self.df_to_ini(ZL_split_change_sell, '张黎', 'ZL_split_change_sell')
            self.df_to_ini(ZL_remain_change_sell, '张黎', 'ZL_remain_change_sell')
            self.df_to_ini(XF_split_change_buy, '许帆', 'XF_split_change_buy')
            self.df_to_ini(XF_remain_change_buy, '许帆', 'XF_remain_change_buy')
            self.df_to_ini(XF_split_change_sell, '许帆', 'XF_split_change_sell')
            self.df_to_ini(XF_remain_change_sell, '许帆', 'XF_remain_change_sell')

    # ZL 专用
    def get_ZL_buy_sell_bsk(self, split):
        # 下载 txt 文件
        txts = ['_B_Tc.txt', '_S_Tc.txt']
        for txt in txts:
            filename = self.yyyymmdd + txt
            remote_path = self.remote_dir + filename
            download_path = self.ZL_path + filename
            self.ftp.download_file(remote_path=remote_path, local_path=download_path)
        # 生成每个篮子调仓的ini
        buy = self.txt_to_df(self.yyyymmdd + '_B_Tc.txt')
        sell = self.txt_to_df(self.yyyymmdd + '_S_Tc.txt')
        buy['to_trade_vol'] = buy['vol'] * self.ZL_ftrs / self.ZL_per_bsk
        sell['to_trade_vol'] = sell['vol'] * self.ZL_ftrs / self.ZL_per_bsk
        self.df_to_ini(buy, '张黎', 'ZL_algo_buy')
        self.df_to_ini(sell, '张黎', 'ZL_algo_sell')
        if split == 1:
            pass
        else:
            ZL_split_algo_buy, ZL_remain_algo_buy = self.split_basket(buy, split)
            ZL_split_algo_sell, ZL_remain_algo_sell = self.split_basket(sell, split)
            self.df_to_ini(ZL_split_algo_buy, '张黎', 'ZL_split_algo_buy')
            self.df_to_ini(ZL_remain_algo_buy, '张黎', 'ZL_remain_algo_buy')
            self.df_to_ini(ZL_split_algo_sell, '张黎', 'ZL_split_algo_sell')
            self.df_to_ini(ZL_remain_algo_sell, '张黎', 'ZL_remain_algo_sell')

    def get_reserve(self):
        ZL_reserve = int(input('请输入ZL预留的篮子数:'))
        XF_reserve = int(input('请输入XF预留的篮子数:'))
        # ZL预留
        ZL_path = self.ZL_path + 'ZL_bsk_pc.ini'
        ZL_res_df = self.ini_to_df(ZL_path)
        ZL_res_df['to_trade_vol'] = ZL_res_df['vol_per_bsk'] * ZL_reserve
        self.df_to_ini(ZL_res_df, '张黎', 'ZL_reserve')
        # XF预留
        latest_bsk_dt = self.dt
        file_name = 'X1.ini'
        while True:
            folder_path = self.root_dir + '许帆/{0}/'.format(latest_bsk_dt.strftime('%Y%m%d'))
            if os.path.exists(folder_path) and file_name in os.listdir(folder_path):
                break
            else:
                latest_bsk_dt = latest_bsk_dt - datetime.timedelta(days=1)
        print('XF篮子更新日期: %s' % latest_bsk_dt.strftime('%Y%m%d'))
        file_path = folder_path + '/' + file_name
        XF_res_df = self.ini_to_df(file_path)
        XF_res_df['to_trade_vol'] = XF_res_df['vol_per_bsk'] * XF_reserve
        self.df_to_ini(XF_res_df, '许帆', 'XF_reserve')

    # ----------------------------------------------------------
    # 工具函数
    def read_positions(self, user):
        user_dict = {'张黎': 'S2', '许帆': 'S4'}
        positions_dt = self.dt
        positions_date = positions_dt.strftime('%Y%m%d')
        file_name = '综合信息查询_组合证券_{0}_{1}.xls'.format(user_dict[user], positions_date)
        while True:
            folder_path = self.root_dir + '{0}/{1}'.format(user, positions_date)
            if os.path.exists(folder_path) and file_name in os.listdir(folder_path):
                break
            else:
                positions_dt = positions_dt - datetime.timedelta(days=1)
                positions_date = positions_dt.strftime('%Y%m%d')
                file_name = '综合信息查询_组合证券_{0}_{1}.xls'.format(user_dict[user], positions_date)
        print('%s持仓日期: %s' % (user, positions_date))
        positions = pd.read_excel(folder_path + '/' + file_name)
        positions = positions.loc[:, ['日期', '证券代码', '证券名称', '持仓', '最新价']].dropna(subset=['证券代码'])
        positions['证券代码'] = positions['证券代码'].astype('int')
        positions['证券代码'] = positions['证券代码'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
        positions.rename(columns={'日期': 'date', '证券代码': 'code', '证券名称': 'name', '持仓': 'vol_hold',
                                  '最新价': 'close'}, inplace=True)
        return positions

    def df_to_ini(self, df, user, fund_id):
        if user == '张黎':
            folder_path = self.ZL_path
        elif user == '许帆':
            folder_path = self.XF_path
        else:
            raise NameError
        with open(folder_path + fund_id + '.ini', mode='w', encoding='gbk') as f:
            f.write('[BASKET]\n')
            f.write('Fundid1=' + fund_id + '\n')
            f.write('TAGTAG\n')
            for idx, row in df.iterrows():
                content = row['code'] + '|' + row['exchange'] + '|' + row['name'] + '|' + \
                          str(int(row['to_trade_vol'])) + '\n'
                f.write(content)
            f.write('ENDENDEND')
        print('%s 生成完毕' % fund_id)

    # ZL专用
    # 生成开平仓篮子使用
    def txt_to_ini(self, filename, fund_id):
        with open(self.ZL_path + '{0}.ini'.format(fund_id), mode='w', encoding='gbk') as f:
            f.write('[BASKET]\n')
            f.write('Fundid1={0}\n'.format(fund_id))
            f.write('TAGTAG\n')
            read = open(self.ZL_path + filename, 'r', encoding='utf-8')
            for line in read.readlines():
                if line[0] == '0' or line[0] == '3' or line[0] == '6':
                    splitted_line = line.strip().split('\t')
                    exchange = 'SH' if (splitted_line[0][0] == '6') | (splitted_line[0][0] == '5') else 'SZ'
                    content = splitted_line[0] + '|' + exchange + '|' + splitted_line[1] + '|' + splitted_line[3] + '\n'
                    f.write(content)
                else:
                    pass
            f.write('ENDENDEND')
        print('篮子: %s 生成完毕' % fund_id)

    # 计算完整调仓时使用
    def ini_to_df(self, file_path):
        f = open(file_path, 'r')
        line_list = []
        for line in f.readlines():
            if line[0] == '0' or line[0] == '3' or line[0] == '6' or line[0] == '5':
                line = line.strip()
                splitted_line = line.split('|')
                line_list.append([splitted_line[0], splitted_line[1], splitted_line[2], splitted_line[3]])
            else:
                continue
        basket = pd.DataFrame(line_list)
        basket.columns = ['code', 'exchange', 'name', 'vol_per_bsk']
        basket['vol_per_bsk'] = basket['vol_per_bsk'].astype('int')
        return basket

    # 生成调仓时使用
    def txt_to_df(self, filename):
        read = open(self.ZL_path + filename, 'r', encoding='utf-8')
        line_list = []
        for line in read.readlines():
            if line[0] == '0' or line[0] == '3' or line[0] == '6':
                splitted_line = line.strip().split('|')
                line_list.append([splitted_line[0], splitted_line[1], splitted_line[2],
                                  splitted_line[3]])
            else:
                continue
        df = pd.DataFrame(line_list, columns= ['code', 'exchange', 'name', 'vol'])
        df['vol'] = df['vol'].astype('int')
        return df

    def split_basket(self, tot_df, split):
        tmp_df = tot_df.copy()
        tmp_df['each_vol'] = np.floor(tmp_df['to_trade_vol'] / split / 100) * 100
        tmp_df['remain_vol'] = tmp_df['to_trade_vol'] - tmp_df['each_vol'] * split
        split_df = tmp_df.loc[tmp_df['each_vol'] > 0, ['code', 'name', 'each_vol', 'exchange']].copy()
        split_df.rename(columns={'each_vol': 'to_trade_vol'}, inplace=True)
        remain_df = tmp_df.loc[tmp_df['remain_vol'] > 0, ['code', 'name', 'remain_vol', 'exchange']].copy()
        remain_df.rename(columns={'remain_vol': 'to_trade_vol'}, inplace=True)
        return [split_df, remain_df]

    def process_change_bsk(self, positions, basket):
        # 滤除转债
        positions = positions.loc[(positions['code'].str[0] == '0') | (positions['code'].str[0] == '3') |
                                  (positions['code'].str[0] == '6'), :]
        positions = positions.merge(basket, on='code', how='outer')
        positions['vol_hold'] = positions['vol_hold'].fillna(0)
        positions['name'] = positions.apply(lambda row: row['name_x'] if pd.notnull(row['name_x']) else row['name_y'],
                                            axis=1)
        positions['exchange'] = positions['code'].apply(lambda x: 'SH' if (x[0] == '6') | (x[0] == '5') else 'SZ')
        positions['target_vol'] = positions['target_vol'].fillna(0)
        positions['vol_diff'] = positions['target_vol'] - positions['vol_hold']
        positions['to_trade_vol'] = positions['vol_diff'].apply(lambda x: round(x, -2))
        change_buy = positions.loc[positions['to_trade_vol'] > 0, ['code', 'name', 'to_trade_vol', 'exchange']]
        change_sell = positions.loc[positions['to_trade_vol'] < 0, ['code', 'name', 'to_trade_vol', 'exchange']]
        change_sell['to_trade_vol'] = change_sell['to_trade_vol'] * -1
        return (change_buy, change_sell)


if __name__ == '__main__':
    bsk = basket_trade()
    bsk.get_open_close_bsk()
    bsk.get_change_bsk(3)
    bsk.get_ZL_buy_sell_bsk(3)
    bsk.get_reserve()
