from csindex_ftp_down import *
import pandas as pd
import datetime

class ftp_data:
    def __init__(self):
        self.ftp_client = FtpClient('192.168.38.213', 'index', 'dfzq1234')

    def get_index_constituent_df(self,date_input,index_type):
        if isinstance(date_input,str):
            str_date = date_input
        elif isinstance(date_input,int):
            str_date = str(date_input)
        elif isinstance(date_input,datetime.datetime):
            str_date = date_input.strftime("%Y%m%d")

        dir_prefix = "d:/basket/"
        IH_prefix = "ftp000016weightnextday"
        IF_prefix = "ftp000300weightnextday"
        IC_prefix = "ftp000905weightnextday"
        postfix = ".xls"
        DEFAULT_LOCAL_DIR = "d:/basket/ftp"

        WeightnextdayManager.down_and_up(str_date, DEFAULT_LOCAL_DIR, 0)

        IH_path = dir_prefix + IH_prefix + str_date + postfix
        IF_path = dir_prefix + IF_prefix + str_date + postfix
        IC_path = dir_prefix + IC_prefix + str_date + postfix

        path_dict = {'IH':IH_path,'IF':IF_path,'IC':IC_path}
        path = path_dict[index_type]

        if os.path.exists(path):
            index_constituent_df = pd.read_excel(path).iloc[:, [0, 1, 2, 3, 4, 5, 7, 11, 12, 13, 16]]
            index_constituent_df.columns = ['Effective Date', 'Index Code', 'Index Name', 'Index Name(Eng)',
                             'Constituent Code', 'Constituent Name', 'Exchange', 'Cap Factor',
                             'Close', 'Reference Open Price for Next Trading Day', 'Weight']
            index_constituent_df['Index Code'] = \
                index_constituent_df.apply(lambda row: str(row['Index Code']).zfill(6), axis=1)
            index_constituent_df['Constituent Code'] = \
                index_constituent_df.apply(lambda row: str(row['Constituent Code']).zfill(6) + '.SH' \
                if str(row['Constituent Code']).zfill(6)[0] == '6' else \
                str(row['Constituent Code']).zfill(6) + '.SZ', axis=1)
            index_constituent_df = index_constituent_df.iloc[:,[0,1,4,5,8,10]]
            index_constituent_df.columns = ['date','index_code','constituent_code','name','close','weight']
            print('from FTP: %s constituent got!' %index_type)
            return index_constituent_df
        else:
            return False

'''
a = ftp_data()
b = a.get_index_constituent_df(20190603,'IH')
print(b)
'''