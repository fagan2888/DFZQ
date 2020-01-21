import pandas as pd
import datetime

class S11_basket:
    def __init__(self):
        pass

    @staticmethod
    def run(date_input=None):
        if not date_input:
            str_date = datetime.datetime.now().strftime('%Y%m%d')
        else:
            str_date = str(date_input)
        root_dir = 'D:/alpha/S11/' + str_date[-4:] + '/'
        file_name = '综合信息查询_成交回报_S11_' + str_date + '.xls'
        file_dir = root_dir + file_name
        transactions = pd.read_excel(file_dir)
        transactions.dropna(subset=['证券代码'],inplace=True)
        transactions['证券代码'] = transactions['证券代码'].astype('int')
        transactions['成交数量'] = transactions['成交数量'].astype('int')
        transactions['code'] = transactions['证券代码'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
        transactions['exchange'] = transactions['code'].apply(lambda x: 'SH' if x[0] == '6' else 'SZ')
        with open('D:/alpha/S11/' + str_date[-4:] + '/' + 'S11_sell' + '.ini', mode='w', encoding='gbk') as f:
            f.write('[BASKET]\n')
            f.write('Fundid1=' + 'S11_sell' + '\n')
            f.write('TAGTAG\n')
            for idx, row in transactions.iterrows():
                if row['委托方向'] == '买入':
                    content = row['code']+'|'+row['exchange']+'|'+row['证券名称']+'|'+str(int(row['成交数量']))+'\n'
                    f.write(content)
                else:
                    pass
            f.write('ENDENDEND')
        print('basket got!')

if __name__ == '__main__':
    S11_basket.run()