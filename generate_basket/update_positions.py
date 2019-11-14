from FTP_service import FTP_service
import os.path
import datetime
import pandas as pd
from rdf_data import rdf_data

class after_trade_task:
    def __init__(self):
        self.ftp = FTP_service(host='192.168.38.213', username='index', password='dfzq1234')
        self.rdf = rdf_data()
        self.yyyymmdd = datetime.datetime.now().strftime('%Y%m%d')
        self.local_dir = 'D:/alpha/张黎/' + self.yyyymmdd + '/'
        if os.path.exists(self.local_dir):
            pass
        else:
            os.makedirs(self.local_dir.rstrip('/'))
        self.remote_dir = '/产品管理/'

    def run(self):
        ftrs_per_bsk = int(input('请输入ZL每个篮子对应的期货手数:'))
        ftrs_vol = int(input('请输入ZL开仓的期货手数:'))
        filename = '综合信息查询_组合证券_S2_' + self.yyyymmdd + '.xls'
        positions = pd.read_excel(self.local_dir+filename)
        positions = positions.loc[:,['证券代码','证券名称','持仓','最新价']].dropna(subset=['证券代码'])
        positions['证券代码'] = positions['证券代码'].astype('int')
        positions['证券代码'] = positions['证券代码'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
        positions['code'] = positions['证券代码'].apply(lambda x: x+'.SH' if x[0]=='6' else x+'.SZ')
        positions['持仓'] = positions['持仓'].astype('int')
        positions['持仓'] = round(positions['持仓']*ftrs_per_bsk/ftrs_vol,-2)
        positions = positions.loc[positions['持仓']>0,:]
        positions['持仓'] = positions['持仓'].astype('int').astype('str')

        # 获取行业信息
        codes = positions['code'].tolist()
        query = "select S_INFO_WINDCODE,s_con_windcode,s_con_indate,s_con_outdate " \
                "from wind_filesync.AIndexMembersCITICS where s_con_windcode in " + str(tuple(codes))
        self.rdf.curs.execute(query)
        industries = pd.DataFrame(self.rdf.curs.fetchall(),columns=['industry','code','entry_date','exit_date'])
        industries = industries.loc[pd.isnull(industries['exit_date']),:]
        positions = positions.merge(industries,right_on='code',left_on='code',how='outer')
        positions['industry'] = positions['industry'].apply(lambda x:x.split('.')[0])

        # 记入txt
        filename = 'alphacc.txt'
        with open(self.local_dir+filename ,mode='w',encoding='utf-8') as f:
            f.write('stkcd:System.String'+'|'+'indu:System.String'+'|'+
                    'ccnum:System.Decimal'+'|'+'price:System.Decimal'+'\n')
            for idx,row in positions.iterrows():
                f.write(row['证券代码'] +'|'+row['industry']+'|'+row['持仓']+'|'+ str(row['最新价'])+'\n')

        self.ftp.upload_file(self.remote_dir+filename,self.local_dir+filename)


if __name__ == '__main__':
    update = after_trade_task()
    update.run()