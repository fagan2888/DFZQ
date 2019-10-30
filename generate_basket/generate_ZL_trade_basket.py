from FTP_service import FTP_service
import datetime
import os.path
import dateutil.parser as dtparser
import pandas as pd


class basket_trade:
    def __init__(self,dt_input=datetime.datetime.now()):
        self.ftp = FTP_service(host='192.168.38.213',username='index',password='dfzq1234')
        if isinstance(dt_input,datetime.datetime):
            self.dt = dt_input
        else:
            self.dt = dtparser.parse(str(dt_input))
        self.yyyymmdd = self.dt.strftime('%Y%m%d')
        self.local_dir = 'D:/alpha/张黎/' + self.yyyymmdd + '/'
        if os.path.exists(self.local_dir):
            pass
        else:
            os.makedirs(self.local_dir.rstrip('/'))
        self.remote_dir = '/产品管理/每日跟踪/'

    def download_generate(self):
        # 处理张黎篮子
        # 下载上一日文件作为平仓依据
        self.ftp.ftp.cwd(self.remote_dir)
        filelist = self.ftp.ftp.nlst()
        last_trade_day = self.dt - datetime.timedelta(days=1)
        filename = last_trade_day.strftime('%Y%m%d')+'_DayCc.txt'
        while not filename in filelist:
            last_trade_day = last_trade_day - datetime.timedelta(days=1)
            filename = last_trade_day.strftime('%Y%m%d') + '_DayCc.txt'
        remote_path = self.remote_dir + filename
        local_path = self.local_dir + filename
        self.ftp.download_file(remote_path=remote_path, local_path=local_path)

        # 生成平仓篮子文件
        with open(self.local_dir + 'ZL_bsk_pc.ini',mode='w',encoding='gbk') as f:
            f.write('[BASKET]\n')
            f.write('Fundid1=' + 'ZL_bsk_pc' + '\n')
            f.write('TAGTAG\n')
            read = open(self.local_dir + filename, 'r', encoding='utf-8')
            for line in read.readlines():
                if line[0] == '0' or line[0] == '3' or line[0] == '6':
                    splitted_line = line.strip().split('\t')
                    exchange = 'SH' if splitted_line[0][0] == '6' else 'SZ'
                    content = splitted_line[0]+'|'+exchange +'|'+splitted_line[1]+'|'+splitted_line[3]+'\n'
                    f.write(content)
                else:
                    pass
            f.write('ENDENDEND')
        print('平仓篮子生成完毕')

        # 下载当日文件
        to_do_list = ['_B_Tc.txt','_S_Tc.txt','_DayCc.txt']
        for postfix in to_do_list:
            remote_path = self.remote_dir + self.yyyymmdd + postfix
            local_path = self.local_dir + self.yyyymmdd + postfix
            self.ftp.download_file(remote_path=remote_path, local_path=local_path)

        # 生成调仓的买卖文件
        with open(self.local_dir + 'ZL_buy.ini',mode='w',encoding='gbk') as f:
            f.write('[BASKET]\n')
            f.write('Fundid1=' + 'ZL_buy' + '\n')
            f.write('TAGTAG\n')
            read = open(self.local_dir + self.yyyymmdd + '_B_Tc.txt', 'r',encoding='utf-8')
            for line in read.readlines():
                if line[0] == '0' or line[0] == '3' or line[0] == '6':
                    splitted_line = line.strip().split('|')
                    content = splitted_line[0]+'|'+splitted_line[1]+'|'+splitted_line[2]+'|'+splitted_line[3]+'\n'
                    f.write(content)
                else:
                    pass
            f.write('ENDENDEND')
        print('买篮子生成完毕')
        with open(self.local_dir + 'ZL_sell.ini',mode='w',encoding='gbk') as f:
            f.write('[BASKET]\n')
            f.write('Fundid1=' + 'ZL_sell' + '\n')
            f.write('TAGTAG\n')
            read = open(self.local_dir + self.yyyymmdd + '_S_Tc.txt', 'r',encoding='utf-8')
            for line in read.readlines():
                if line[0] == '0' or line[0] == '3' or line[0] == '6':
                    splitted_line = line.strip().split('|')
                    content = splitted_line[0]+'|'+splitted_line[1]+'|'+splitted_line[2]+'|'+splitted_line[3]+'\n'
                    f.write(content)
                else:
                    pass
            f.write('ENDENDEND')
        print('卖篮子生成完毕')
        with open(self.local_dir + 'ZL_bsk_kc.ini',mode='w',encoding='gbk') as f:
            f.write('[BASKET]\n')
            f.write('Fundid1=' + 'ZL_bsk_kc' + '\n')
            f.write('TAGTAG\n')
            read = open(self.local_dir + self.yyyymmdd + '_DayCc.txt', 'r',encoding='utf-8')
            for line in read.readlines():
                if line[0] == '0' or line[0] == '3' or line[0] == '6':
                    splitted_line = line.strip().split('\t')
                    exchange = 'SH' if splitted_line[0][0] == '6' else 'SZ'
                    content = splitted_line[0]+'|'+exchange +'|'+splitted_line[1]+'|'+splitted_line[3]+'\n'
                    f.write(content)
                else:
                    pass
            f.write('ENDENDEND')
        print('开仓篮子生成完毕')


    def process_change_bsk(self,positions,basket):
        positions = positions.merge(basket, left_on='证券代码', right_on='证券代码', how='outer')
        positions['持仓'] = positions['持仓'].fillna(0)
        positions['证券名称'] = positions.apply(lambda row: row['name'] if pd.isnull(row['证券名称']) else row['证券名称'],
                                            axis=1)
        positions['target_vol'] = positions['target_vol'].fillna(0)
        positions['vol_diff'] = round(positions['target_vol'] - positions['持仓'], -2)
        positions['exchange'] = positions['证券代码'].apply(lambda x: 'SH' if x[0] == '6' else 'SZ')
        change_buy = positions.loc[positions['vol_diff'] > 0, ['证券代码', '证券名称', 'vol_diff', 'exchange']]
        change_sell = positions.loc[positions['vol_diff'] < 0, ['证券代码', '证券名称', 'vol_diff', 'exchange']]
        change_sell['vol_diff'] = change_sell['vol_diff'] * -1
        return (change_buy,change_sell)


    def df_to_ini(self,df,user,fund_id):
        user_dict = {'ZL':'张黎','XF':'许帆'}
        folder_name = 'D:/alpha/'+ user_dict[user] +'/' + self.dt.strftime('%Y%m%d')
        if os.path.exists(folder_name):
            pass
        else:
            os.makedirs(folder_name)
        with open(folder_name + '/' + fund_id + '.ini', mode='w', encoding='gbk') as f:
            f.write('[BASKET]\n')
            f.write('Fundid1=' + fund_id + '\n')
            f.write('TAGTAG\n')
            for idx, row in df.iterrows():
                content = row['证券代码']+'|'+row['exchange']+'|'+row['证券名称']+'|'+str(int(row['vol_diff']))+'\n'
                f.write(content)
            f.write('ENDENDEND')
        print('%s 生成完毕' %fund_id)



    def get_change_bsk(self):
        ZL_per_bsk = int(input('请输入ZL每个篮子对应的期货手数:'))
        ZL_ftrs = int(input('请输入ZL开仓的期货手数:'))
        XF_per_bsk = int(input('请输入XF每个篮子对应的期货手数:'))
        XF_ftrs = int(input('请输入XF开仓的期货手数:'))

        root_dir = 'D:/alpha/'
        # 读取持仓ZL
        positions_date = self.dt
        folder_name = positions_date.strftime('%Y%m%d')
        file_name = '综合信息查询_组合证券_ZL_' + folder_name + '.xls'
        while True :
            folder_path = root_dir + '张黎/' + folder_name
            if os.path.exists(folder_path) and file_name in os.listdir(folder_path):
                break
            else:
                positions_date = positions_date - datetime.timedelta(days=1)
                folder_name = positions_date.strftime('%Y%m%d')
                file_name = '综合信息查询_组合证券_ZL_' + folder_name + '.xls'

        print('ZL持仓日期: %s' % folder_name)
        positions_ZL = pd.read_excel(root_dir + '张黎/'+ folder_name+'/'+'综合信息查询_组合证券_ZL_'+folder_name+'.xls')
        positions_ZL = positions_ZL.loc[:, ['证券代码', '证券名称', '持仓']].dropna(subset=['证券代码'])
        positions_ZL['证券代码'] = positions_ZL['证券代码'].astype('int')
        positions_ZL['证券代码'] = positions_ZL['证券代码'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
        #positions_ZL['code'] = positions_ZL['证券代码'].apply(lambda x: x + '.SH' if x[0] == '6' else x + '.SZ')

        # 读取持仓XF
        positions_date = self.dt
        folder_name = positions_date.strftime('%Y%m%d')
        file_name = '综合信息查询_组合证券_XF_' + folder_name + '.xls'
        while True:
            folder_path = root_dir + '许帆/' + folder_name
            if os.path.exists(folder_path) and file_name in os.listdir(folder_path):
                break
            else:
                positions_date = positions_date - datetime.timedelta(days=1)
                folder_name = positions_date.strftime('%Y%m%d')
                file_name = '综合信息查询_组合证券_XF_' + folder_name + '.xls'

        print('XF持仓日期: %s' % folder_name)
        positions_XF = pd.read_excel(root_dir + '许帆/' + folder_name + '/' + '综合信息查询_组合证券_XF_' + folder_name + '.xls')
        positions_XF = positions_XF.loc[:, ['证券代码', '证券名称', '持仓']].dropna(subset=['证券代码'])
        positions_XF['证券代码'] = positions_XF['证券代码'].astype('int')
        positions_XF['证券代码'] = positions_XF['证券代码'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
        #positions_XF['code'] = positions_XF['证券代码'].apply(lambda x: x + '.SH' if x[0] == '6' else x + '.SZ')

        # 读取最新篮子ZL
        folder_name = self.dt.strftime('%Y%m%d')
        file_name = 'ZL_bsk_kc.ini'
        f = open(root_dir+'张黎/'+folder_name+'/'+file_name, 'r')
        line_list = []
        for line in f.readlines():
            if line[0] == '0' or line[0] == '3' or line[0] == '6':
                line = line.strip()
                splitted_line = line.split('|')
                line_list.append([splitted_line[0], splitted_line[2], splitted_line[3]])
            else:
                continue
        basket_ZL = pd.DataFrame(line_list)
        basket_ZL.columns = ['证券代码', 'name', '篮子配置数']
        basket_ZL['篮子配置数'] = basket_ZL['篮子配置数'].astype('int')
        basket_ZL['target_vol'] =  basket_ZL['篮子配置数'] * ZL_ftrs / ZL_per_bsk
        basket_ZL = basket_ZL.loc[:,['证券代码','name','target_vol']]

        # 读取最新篮子XF
        latest_bsk_date = self.dt
        file_name = 'X1.ini'
        folder_path = root_dir + '许帆/' + latest_bsk_date.strftime('%Y%m%d')
        while True:
            if os.path.exists(folder_path) and file_name in os.listdir(folder_path):
                break
            else:
                latest_bsk_date = latest_bsk_date - datetime.timedelta(days=1)
                folder_path = root_dir + '许帆/' + latest_bsk_date.strftime('%Y%m%d')
        folder_name = latest_bsk_date.strftime('%Y%m%d')
        print('XF篮子更新日期: %s' % folder_name)

        f = open(root_dir+'许帆/'+folder_name+'/'+file_name, 'r')
        line_list = []
        for line in f.readlines():
            if line[0] == '0' or line[0] == '3' or line[0] == '6':
                line = line.strip()
                splitted_line = line.split('|')
                line_list.append([splitted_line[0], splitted_line[2], splitted_line[3]])
            else:
                continue
        basket_XF = pd.DataFrame(line_list)
        basket_XF.columns = ['证券代码', 'name', '篮子配置数']
        basket_XF['篮子配置数'] = basket_XF['篮子配置数'].astype('int')
        basket_XF['target_vol'] = basket_XF['篮子配置数'] * XF_ftrs / XF_per_bsk
        basket_XF = basket_XF.loc[:,['证券代码','name','target_vol']]


        # 自行计算调仓
        ZL_change_buy = self.process_change_bsk(positions_ZL,basket_ZL)[0]
        ZL_change_sell = self.process_change_bsk(positions_ZL,basket_ZL)[1]
        XF_change_buy = self.process_change_bsk(positions_XF,basket_XF)[0]
        XF_change_sell = self.process_change_bsk(positions_XF,basket_XF)[1]

        self.df_to_ini(ZL_change_buy, 'ZL', 'ZL_change_buy')
        self.df_to_ini(ZL_change_sell, 'ZL', 'ZL_change_sell')
        self.df_to_ini(XF_change_buy, 'XF', 'XF_change_buy')
        self.df_to_ini(XF_change_sell, 'XF', 'XF_change_sell')


if __name__ == '__main__':
    bsk = basket_trade()
    bsk.download_generate()
    bsk.get_change_bsk()