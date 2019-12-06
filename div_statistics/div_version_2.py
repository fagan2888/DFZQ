from csindex_ftp_down import *
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import os
import cx_Oracle as oracle

import warnings


class div_pred_statistic:
    def __init__(self):
        self.dir_prefix = "d:/basket/"
        self.IH_prefix = "ftp000016weightnextday"
        self.IF_prefix = "ftp000300weightnextday"
        self.IC_prefix = "ftp000905weightnextday"
        self.postfix = ".xls"
        self.DEFAULT_LOCAL_DIR = "d:/basket/ftp"

        self.ftp_client = FtpClient('192.168.38.213', 'index', 'dfzq1234')

        today = datetime.datetime.today().date()
        self.dt = datetime.datetime(year=today.year, month=today.month, day=today.day)

        self.this_year = datetime.datetime.now().date().year
        self.this_year_start_date = datetime.datetime(year=self.this_year, month=1, day=1)
        self.last_year_start_date = datetime.datetime(year=self.this_year - 1, month=1, day=1)
        self.last_year_end_date = datetime.datetime(year=self.this_year - 1, month=12, day=31)

        self.ip_address = '172.17.21.3'
        self.port = '1521'
        self.username = 'yspread'
        self.pwd = 'Y*iaciej123456'
        self.conn = oracle.connect(
            self.username + "/" + self.pwd + "@" + self.ip_address + ":" + self.port + "/" + "WDZX")
        self.curs = self.conn.cursor()

    def get_constituent_df_ftp(self, dt_input):
        self.constituent_df_dict = {}
        self.constituent_list_dict = {}
        str_dt = dt_input.strftime("%Y%m%d")
        self.IH_path = self.dir_prefix + self.IH_prefix + str_dt + self.postfix
        self.IF_path = self.dir_prefix + self.IF_prefix + str_dt + self.postfix
        self.IC_path = self.dir_prefix + self.IC_prefix + str_dt + self.postfix

        WeightnextdayManager.down_and_up(str_dt, self.DEFAULT_LOCAL_DIR, 0)

        if os.path.exists(self.IH_path):
            IH_df = pd.read_excel(self.IH_path).iloc[:, [0, 1, 2, 3, 4, 5, 7, 11, 12, 13, 16]]
            IF_df = pd.read_excel(self.IF_path).iloc[:, [0, 1, 2, 3, 4, 5, 7, 11, 12, 13, 16]]
            IC_df = pd.read_excel(self.IC_path).iloc[:, [0, 1, 2, 3, 4, 5, 7, 11, 12, 13, 16]]
            IH_df.columns = ['Effective Date', 'Index Code', 'Index Name', 'Index Name(Eng)',
                             'Constituent Code', 'Constituent Name', 'Exchange', 'Cap Factor',
                             'Close', 'Reference Open Price for Next Trading Day', 'Weight']
            IF_df.columns = ['Effective Date', 'Index Code', 'Index Name', 'Index Name(Eng)',
                             'Constituent Code', 'Constituent Name', 'Exchange', 'Cap Factor',
                             'Close', 'Reference Open Price for Next Trading Day', 'Weight']
            IC_df.columns = ['Effective Date', 'Index Code', 'Index Name', 'Index Name(Eng)',
                             'Constituent Code', 'Constituent Name', 'Exchange', 'Cap Factor',
                             'Close', 'Reference Open Price for Next Trading Day', 'Weight']

            IH_df['Index Code'] = IH_df.apply(lambda row: str(row['Index Code']).zfill(6), axis=1)
            IH_df['Constituent Code'] = IH_df.apply(lambda row: str(row['Constituent Code']).zfill(6) + '.SH' \
                if str(row['Constituent Code']).zfill(6)[0] == '6' else \
                str(row['Constituent Code']).zfill(6) + '.SZ', axis=1)
            IF_df['Index Code'] = IF_df.apply(lambda row: str(row['Index Code']).zfill(6), axis=1)
            IF_df['Constituent Code'] = IF_df.apply(lambda row: str(row['Constituent Code']).zfill(6) + '.SH' \
                if str(row['Constituent Code']).zfill(6)[0] == '6' else \
                str(row['Constituent Code']).zfill(6) + '.SZ', axis=1)
            IC_df['Index Code'] = IC_df.apply(lambda row: str(row['Index Code']).zfill(6), axis=1)
            IC_df['Constituent Code'] = IC_df.apply(lambda row: str(row['Constituent Code']).zfill(6) + '.SH' \
                if str(row['Constituent Code']).zfill(6)[0] == '6' else \
                str(row['Constituent Code']).zfill(6) + '.SZ', axis=1)

            self.constituent_list_dict['IH'] = list(IH_df.loc[:, 'Constituent Code'])
            self.constituent_list_dict['IF'] = list(IF_df.loc[:, 'Constituent Code'])
            self.constituent_list_dict['IC'] = list(IC_df.loc[:, 'Constituent Code'])
            self.constituent_list_dict['all'] = list(IF_df.loc[:, 'Constituent Code']) + list(
                IC_df.loc[:, 'Constituent Code'])

            self.constituent_df_dict['IH'] = IH_df
            self.constituent_df_dict['IF'] = IF_df
            self.constituent_df_dict['IC'] = IC_df
            self.constituent_df_dict['all'] = pd.concat([IF_df, IC_df], ignore_index=True, sort=True)

            print('constituent got!')
            return True
        else:
            return False


    def get_constituent_from_csv(self):
        self.constituent_df_dict = {}
        self.constituent_list_dict = {}
        IH_df = pd.read_csv('IH_weight.csv',encoding='gbk',index_col=0).loc[:,['index_code','code','close','weight']]
        IF_df = pd.read_csv('IF_weight.csv',encoding='gbk',index_col=0).loc[:,['index_code','code','close','weight']]
        IC_df = pd.read_csv('IC_weight.csv',encoding='gbk',index_col=0).loc[:,['index_code','code','close','weight']]
        IH_df.columns = ['Index Code','Constituent Code','Close','Weight']
        IF_df.columns = ['Index Code','Constituent Code','Close','Weight']
        IC_df.columns = ['Index Code','Constituent Code','Close','Weight']

        self.constituent_list_dict['IH'] = list(IH_df.loc[:, 'Constituent Code'])
        self.constituent_list_dict['IF'] = list(IF_df.loc[:, 'Constituent Code'])
        self.constituent_list_dict['IC'] = list(IC_df.loc[:, 'Constituent Code'])
        self.constituent_list_dict['all'] = list(IF_df.loc[:, 'Constituent Code']) + \
                                            list(IC_df.loc[:, 'Constituent Code'])

        self.constituent_df_dict['IH'] = IH_df
        self.constituent_df_dict['IF'] = IF_df
        self.constituent_df_dict['IC'] = IC_df
        self.constituent_df_dict['all'] = pd.concat([IF_df, IC_df], ignore_index=True, sort=True)

        print('constituent got!')


    def get_constituent_df(self,dt_input):
        self.constituent_df_dict = {}
        self.constituent_list_dict = {}
        str_dt = dt_input.strftime("%Y%m%d")
        IH_sentense = "select TRADE_DT,S_INFO_WINDCODE,S_CON_WINDCODE,closevalue,weight " \
                      "from wind_filesync.AIndexSSE50Weight " \
                      "where trade_dt =: dt"
        self.curs.execute(IH_sentense,dt=str_dt)
        IH_df = pd.DataFrame(self.curs.fetchall(),columns=['Effective Date', 'Index Code','Constituent Code','Close','Weight'])
        IF_sentense = "select TRADE_DT,S_INFO_WINDCODE,S_CON_WINDCODE,i_weight_15,i_weight " \
                      "from wind_filesync.AIndexHS300Weight " \
                      "where trade_dt =: dt"
        self.curs.execute(IF_sentense,dt=str_dt)
        IF_df = pd.DataFrame(self.curs.fetchall(),columns=['Effective Date', 'Index Code','Constituent Code','Close','Weight'])
        IC_sentense = "select TRADE_DT,S_INFO_WINDCODE,S_CON_WINDCODE,closevalue,weight " \
                      "from wind_filesync.AIndexCSI500Weight " \
                      "where trade_dt =: dt"
        self.curs.execute(IC_sentense,dt=str_dt)
        IC_df = pd.DataFrame(self.curs.fetchall(),columns=['Effective Date', 'Index Code','Constituent Code','Close','Weight'])
        self.constituent_list_dict['IH'] = list(IH_df.loc[:, 'Constituent Code'])
        self.constituent_list_dict['IF'] = list(IF_df.loc[:, 'Constituent Code'])
        self.constituent_list_dict['IC'] = list(IC_df.loc[:, 'Constituent Code'])
        self.constituent_list_dict['all'] = list(IF_df.loc[:, 'Constituent Code']) + list(IC_df.loc[:, 'Constituent Code'])

        self.constituent_df_dict['IH'] = IH_df
        self.constituent_df_dict['IF'] = IF_df
        self.constituent_df_dict['IC'] = IC_df
        self.constituent_df_dict['all'] = pd.concat([IF_df, IC_df], ignore_index=True, sort=True)

        print('constituent got!')


    def get_trading_calendar(self):
        get_calendar_sentense = "select TRADE_DAYS from wind_filesync.AShareCalendar " \
                                "where trade_days >= 20160101"
        self.curs.execute(get_calendar_sentense)
        fetch_data = self.curs.fetchall()
        trading_days_list = []
        for data in fetch_data:
            trading_days_list.append(data[0])
        self.trading_calendar = pd.to_datetime(pd.Series(trading_days_list)).drop_duplicates()\
                                .sort_values().reset_index(drop=True)

        temp_series = self.trading_calendar[self.trading_calendar < self.dt].reset_index(drop=True)
        self.last_trading_date = temp_series[len(temp_series) - 1]

    def get_stock_name(self, code):
        get_name_sentense = 'select s_info_windcode,s_info_name ' \
                            'from wind_filesync.AShareDescription ' \
                            'where s_info_windcode = :cd '
        self.curs.execute(get_name_sentense,cd=code)
        code_name = self.curs.fetchall()[0][1]
        return code_name

    def add_stock_name(self, df):
        if len(df) > 1:
            code_str = str(tuple(df['股票代码']))
        elif len(df) == 1:
            code_str = "('" + df.loc[0, '股票代码'] + "')"
        else:
            df['股票名称'] = None
            return df
        get_name_sentense = 'select s_info_windcode,s_info_name ' \
                            'from wind_filesync.AShareDescription ' \
                            'where s_info_windcode in '
        get_name_sentense = get_name_sentense + code_str
        self.curs.execute(get_name_sentense)
        code_name = self.curs.fetchall()

        name_dict = {}
        for pair in code_name:
            name_dict[pair[0]] = pair[1]
        for i in range(len(df)):
            df.loc[i, '股票名称'] = name_dict[df.loc[i, '股票代码']]
        return df

    def get_div_df(self, prediction_type, dt_start_input, dt_end_input=None):
        if not self.constituent_list_dict:
            print('please run get constituent first')
            return
        constituent_list = self.constituent_list_dict[prediction_type]
        str_dt_start = dt_start_input.strftime("%Y%m%d")
        if not dt_end_input:
            sql_sentense = 'select s_info_windcode,STK_DVD_PER_SH,CASH_DVD_PER_SH_PRE_TAX,' \
                           'S_DIV_PROGRESS,EQY_RECORD_DT,EX_DT,DVD_PAYOUT_DT,DVD_ANN_DT,S_DIV_PRELANDATE,report_period ' \
                           'from wind_filesync.AShareDividend ' \
                           'where s_div_progress <= 3 and S_DIV_PRELANDATE >= :start_date ' \
                           'and s_info_windcode in ' + str(tuple(constituent_list)) \
                           + 'order by eqy_record_dt'
            self.curs.execute(sql_sentense, start_date=str_dt_start)
        else:
            str_dt_end = dt_end_input.strftime("%Y%m%d")
            sql_sentense = 'select s_info_windcode, STK_DVD_PER_SH, CASH_DVD_PER_SH_PRE_TAX,' \
                           'S_DIV_PROGRESS,EQY_RECORD_DT,EX_DT,DVD_PAYOUT_DT,DVD_ANN_DT,S_DIV_PRELANDATE,report_period ' \
                           'from wind_filesync.AShareDividend ' \
                           'where s_div_progress <= 3 and S_DIV_PRELANDATE >= :start_date and S_DIV_PRELANDATE <= :end_date ' \
                           'and s_info_windcode in ' + str(tuple(constituent_list)) \
                           + 'order by eqy_record_dt'
            self.curs.execute(sql_sentense, start_date=str_dt_start, end_date=str_dt_end)
        fetch_data = self.curs.fetchall()
        self.div_df = pd.DataFrame(fetch_data, columns=['股票代码', '每股转送', '税前派息', '方案进度', '股权登记日',
                                                        '除权除息日', '派息日', '分红实施公告日', '预案公告日', '分红年度'])
        self.div_df = self.add_stock_name(self.div_df).loc[:, ['股票代码', '股票名称', '每股转送', '税前派息', '方案进度',
                                        '股权登记日', '除权除息日', '派息日', '分红实施公告日', '预案公告日', '分红年度']]
        return self.div_df

    def get_all_historical_data(self,code_list):
        sql_sentense = 'select s_info_windcode,STK_DVD_PER_SH,CASH_DVD_PER_SH_PRE_TAX,S_DIV_PROGRESS,EQY_RECORD_DT,' \
                       'EX_DT,DVD_PAYOUT_DT,DVD_ANN_DT,S_DIV_PRELANDATE,s_div_smtgdate,report_period ' \
                       'from wind_filesync.AShareDividend ' \
                       'where s_div_progress <=3 and s_info_windcode in ' + str(tuple(code_list)) + \
                       'order by S_info_windcode'
        self.curs.execute(sql_sentense)
        fetch_data = self.curs.fetchall()
        all_historical_data = pd.DataFrame(fetch_data,columns=['股票代码', '每股转送', '税前派息', '方案进度', '股权登记日',
            '除权除息日', '派息日', '分红实施公告日', '预案公告日', '股东大会公告日','分红年度'])
        return all_historical_data


    def get_right_issue_df(self, dt_input):
        sql_sentence = 'select s_info_windcode,S_RIGHTSISSUE_PROGRESS,S_RIGHTSISSUE_PRICE,S_RIGHTSISSUE_RATIO,' \
                       'S_RIGHTSISSUE_AMOUNT,S_RIGHTSISSUE_AMOUNTACT,S_RIGHTSISSUE_NETCOLLECTION,' \
                       'S_RIGHTSISSUE_REGDATESHAREB,S_RIGHTSISSUE_EXDIVIDENDDATE,S_RIGHTSISSUE_LISTEDDATE,' \
                       'S_RIGHTSISSUE_PAYSTARTDATE,S_RIGHTSISSUE_PAYENDDATE,S_RIGHTSISSUE_ANNCEDATE,' \
                       'S_RIGHTSISSUE_RESULTDATE,S_RIGHTSISSUE_YEAR ' \
                       'from wind_filesync.AShareRightIssue ' \
                       'where S_RIGHTSISSUE_PROGRESS <= 3 and S_RIGHTSISSUE_REGDATESHAREB >= :dt ' \
                       'order by S_RIGHTSISSUE_REGDATESHAREB'
        str_dt = dt_input.strftime("%Y%m%d")
        self.curs.execute(sql_sentence, dt=str_dt)
        fetch_data = self.curs.fetchall()
        self.right_issue_df = pd.DataFrame(fetch_data, columns=['股票代码', '方案进度', '配股价格', '配股比例',
                                '配股计划数量（万股）', '配股实际数量（万股）', '募集资金（元）', '股权登记日', '除权日',
                                '配股上市日', '缴款起始日', '缴款终止日', '配股实施公告日', '配股结果公告日','配股年度'])
        self.right_issue_df = self.add_stock_name(self.right_issue_df).loc[:, ['股票代码', '股票名称', '方案进度',
                                '配股价格', '配股比例', '配股计划数量（万股）','配股实际数量（万股）', '募集资金（元）',
                                '股权登记日', '除权日', '配股上市日', '缴款起始日','缴款终止日', '配股实施公告日',
                                '配股结果公告日','配股年度']]
        return

    def get_index_close(self, dt_input):
        sql_sentence = "select s_info_windcode,trade_dt,s_dq_close " \
                       "from wind_filesync.AIndexEODPrices " \
                       "where s_info_windcode in ('000016.SH','000300.SH','000905.SH') and trade_dt = :dt"

        str_dt = self.last_trading_date.strftime("%Y%m%d")
        self.curs.execute(sql_sentence, dt=str_dt)
        fetch_data = self.curs.fetchall()
        self.index_close_df = pd.DataFrame(fetch_data, columns=['股票代码', '日期', '收盘价'])
        self.index_close_df['日期'] = pd.to_datetime(self.index_close_df['日期'], format="%Y%m%d")
        self.index_close_df.set_index('股票代码', inplace=True)
        return

    def get_eps_df(self, code_list):
        print('fetching eps data...')
        sql_sentence = "select s_info_windcode,trade_dt,NET_PROFIT_PARENT_COMP_TTM,TOT_SHR_TODAY " \
                       "from wind_filesync.AShareEODDerivativeIndicator " \
                       "where trade_dt >20160101 and s_info_windcode in " \
                       + str(tuple(code_list))
        self.curs.execute(sql_sentence)
        fetch_data = self.curs.fetchall()
        self.eps_df = pd.DataFrame(fetch_data, columns=['code', 'date', 'net_profit', 'total_shares'])
        self.eps_df['date'] = pd.to_datetime(self.eps_df['date'], format="%Y%m%d")
        self.eps_df.sort_values(by='date', ascending=True, inplace=True)
        self.eps_df.set_index('date', inplace=True)
        self.eps_df['eps'] = self.eps_df['net_profit'] / self.eps_df['total_shares']
        print('eps data got!')
        return

    def cal_div_point(self, div_pre_tax, prediction_type):
        constituent_df = self.constituent_df_dict[prediction_type]
        index_code_dict = {'IH': '000016.SH', 'IF': '000300.SH', 'IC': '000905.SH'}
        div_point_value = div_pre_tax / \
                          constituent_df.loc[constituent_df['Constituent Code'] == self.stock_code_processing, 'Close'].iloc[0] * \
                          constituent_df.loc[constituent_df['Constituent Code'] == self.stock_code_processing, 'Weight'].iloc[0] * \
                          self.index_close_df.loc[index_code_dict[prediction_type], '收盘价'] / 100
        return div_point_value

    def process_prediction(self, prediction_type):
        result_df = pd.DataFrame(columns=['股票代码', '股票名称', '状态', '每股转送', '股权登记日', '派息日', '预案公告日',
                                          '税前派息', '分红点数', '是否预测'])
        unconfirmed_div_df = self.stock_div_df.loc[self.stock_div_df['派息日'].isna(), :].copy()
        confirmed_div_df = self.stock_div_df.loc[self.stock_div_df['派息日'].notna(), :].copy()
        confirmed_div_df.loc[:, '派息日'] = pd.to_datetime(confirmed_div_df.loc[:, '派息日'], format="%Y%m%d")
        confirmed_div_df.loc[:, '股权登记日'] = pd.to_datetime(confirmed_div_df.loc[:, '股权登记日'], format="%Y%m%d")
        div_this_year_df = confirmed_div_df.loc[confirmed_div_df.loc[:, '派息日'] >= self.this_year_start_date, :]
        div_last_year_df = confirmed_div_df.loc[(confirmed_div_df.loc[:, '派息日'] >= self.last_year_start_date) &
                                                (confirmed_div_df.loc[:, '派息日'] <= self.last_year_end_date), :]
        this_year_div_times = len(unconfirmed_div_df) + len(div_this_year_df)

        # 去年没分，今年不论有没有，都不预测
        if div_last_year_df.empty:
            if this_year_div_times == 0:
                temp_dict = {}
                temp_dict['股票代码'] = self.stock_code_processing
                temp_dict['股票名称'] = self.stock_name_processing
                temp_dict['状态'] = '去年无分红，今年无预案，预测不分红'
                temp_dict['是否预测'] = True
                temp_dict['每股转送'] = None
                temp_dict['股权登记日'] = None
                temp_dict['派息日'] = None
                temp_dict['预案公告日'] = None
                temp_dict['税前派息'] = 0
                temp_dict['分红点数'] = 0
            else:
                if len(unconfirmed_div_df)>0:
                    estimated_div_dt = self.trading_calendar[self.trading_calendar >=
                        unconfirmed_div_df.iloc[0]['预案公告日'] + relativedelta(months=2)].reset_index(drop=True)[0]
                    temp_series = self.trading_calendar[self.trading_calendar <= estimated_div_dt].reset_index(drop=True)
                    estimated_reg_dt = temp_series.iloc[-1]
                    if estimated_reg_dt <= self.dt + datetime.timedelta(days=7):
                        estimated_reg_dt = np.nan
                        estimated_div_dt = np.nan
                    temp_dict = {}
                    temp_dict['股票代码'] = self.stock_code_processing
                    temp_dict['股票名称'] = self.stock_name_processing
                    temp_dict['状态'] = '去年无分红，今年已有预案，预测派息日和股权登记日'
                    temp_dict['是否预测'] = True
                    temp_dict['每股转送'] = unconfirmed_div_df.iloc[0]['每股转送']
                    temp_dict['股权登记日'] = estimated_reg_dt
                    temp_dict['派息日'] = estimated_div_dt
                    temp_dict['预案公告日'] = unconfirmed_div_df.iloc[0]['预案公告日']
                    temp_dict['税前派息'] = unconfirmed_div_df.iloc[0]['税前派息']
                    temp_dict['分红点数'] = self.cal_div_point(unconfirmed_div_df.iloc[0]['税前派息'], prediction_type)
                    temp_df = pd.DataFrame(temp_dict, index=[0])
                    result_df = result_df.append(temp_df, ignore_index=True)

                if len(div_this_year_df)>0:
                    for i in range(len(div_this_year_df)):
                        temp_dict = {}
                        temp_dict['股票代码'] = self.stock_code_processing
                        temp_dict['股票名称'] = self.stock_name_processing
                        temp_dict['状态'] = '去年无分红，今年已有公告，实际派息日和股权登记日'
                        temp_dict['是否预测'] = False
                        temp_dict['每股转送'] = div_this_year_df.iloc[i]['每股转送']
                        temp_dict['股权登记日'] = div_this_year_df.iloc[i]['股权登记日']
                        temp_dict['派息日'] = div_this_year_df.iloc[i]['派息日']
                        temp_dict['预案公告日'] = div_this_year_df.iloc[i]['预案公告日']
                        temp_dict['税前派息'] = div_this_year_df.iloc[i]['税前派息']
                        temp_dict['分红点数'] = self.cal_div_point(div_this_year_df.iloc[i]['税前派息'], prediction_type)
                        temp_df = pd.DataFrame(temp_dict, index=[0])
                        result_df = result_df.append(temp_df, ignore_index=True)
            return result_df

        # 去年分红
        else:
            stock_eps_df = self.eps_df.loc[self.eps_df.loc[:, 'code'] == self.stock_code_processing, :]
            eps_df_before_dt = stock_eps_df.loc[:self.dt, :]
            if eps_df_before_dt.empty:
                eps_on_dt = 1
                eps_last_year = 1
            else:
                eps_on_dt = eps_df_before_dt.iloc[-1]['eps']
                eps_date = eps_df_before_dt.index[-1]
                eps_date_last_year = eps_date - relativedelta(years=1)
                if eps_df_before_dt.loc[:eps_date_last_year, :].empty:
                    eps_on_dt = 1
                    eps_last_year = 1
                else:
                    eps_last_year = eps_df_before_dt.loc[:eps_date_last_year, :].iloc[-1]['eps']

            # 今年没有分红预案或公告
            if this_year_div_times == 0:
                for i in range(len(div_last_year_df)):
                    temp_dict = {}
                    temp_dict['股票代码'] = self.stock_code_processing
                    temp_dict['股票名称'] = self.stock_name_processing
                    temp_dict['状态'] = '去年有分红，今年无预案无公告，预测分红情况'
                    temp_dict['是否预测'] = True
                    div_on_hist_date = div_last_year_df.iloc[i]['税前派息']

                    if eps_last_year < 0 and eps_on_dt >= 0:
                        div_amount_prediction = div_on_hist_date
                    elif eps_last_year > 0 and eps_on_dt > 0:
                        div_amount_prediction = div_on_hist_date / eps_last_year * eps_on_dt
                    else:
                        div_amount_prediction = 0
                    if div_amount_prediction == 0:
                        temp_dict['每股转送'] = None
                        temp_dict['税前派息'] = 0
                        temp_dict['分红点数'] = 0
                        temp_dict['派息日'] = None
                        temp_dict['预案公告日'] = None
                        temp_dict['股权登记日'] = None
                    else:
                        temp_dict['每股转送'] = None
                        temp_dict['税前派息'] = div_amount_prediction
                        temp_dict['分红点数'] = self.cal_div_point(div_amount_prediction,prediction_type)
                        hist_div_date = div_last_year_df.iloc[i]['派息日']
                        estimated_div_dt = hist_div_date + relativedelta(years=1)
                        estimated_div_dt = self.trading_calendar[self.trading_calendar >=
                            (estimated_div_dt-datetime.timedelta(days=1))].reset_index(drop=True)[0]
                        temp_series = self.trading_calendar[self.trading_calendar <= estimated_div_dt].reset_index(drop=True)
                        estimated_reg_dt = temp_series.iloc[-1]
                        if estimated_reg_dt <= self.dt + datetime.timedelta(days=7):
                            estimated_reg_dt = np.nan
                            estimated_div_dt = np.nan
                        temp_dict['股权登记日'] = estimated_reg_dt
                        temp_dict['派息日'] = estimated_div_dt
                        temp_dict['预案公告日'] = np.nan
                    temp_df = pd.DataFrame(temp_dict,index=[0])
                    result_df = result_df.append(temp_df,ignore_index=True)
                return result_df

            # 今年有公告
            if len(div_this_year_df)>0:
                for i in range(len(div_this_year_df)):
                    temp_dict = {}
                    temp_dict['股票代码'] = self.stock_code_processing
                    temp_dict['股票名称'] = self.stock_name_processing
                    temp_dict['状态'] = '去年有分红，今年已有公告，实际派息日和股权登记日'
                    temp_dict['是否预测'] = False
                    temp_dict['每股转送'] = div_this_year_df.iloc[i]['每股转送']
                    temp_dict['股权登记日'] = div_this_year_df.iloc[i]['股权登记日']
                    temp_dict['派息日'] = div_this_year_df.iloc[i]['派息日']
                    temp_dict['预案公告日'] = div_this_year_df.iloc[i]['预案公告日']
                    temp_dict['税前派息'] = div_this_year_df.iloc[i]['税前派息']
                    temp_dict['分红点数'] = self.cal_div_point(div_this_year_df.iloc[i]['税前派息'], prediction_type)
                    temp_df = pd.DataFrame(temp_dict, index=[0])
                    result_df = result_df.append(temp_df, ignore_index=True)
            # 今年有预案
            if len(unconfirmed_div_df)>0:
                temp_dict = {}
                temp_dict['股票代码'] = self.stock_code_processing
                temp_dict['股票名称'] = self.stock_name_processing
                temp_dict['每股转送'] = unconfirmed_div_df.iloc[0]['每股转送']
                temp_dict['预案公告日'] = unconfirmed_div_df.iloc[0]['预案公告日']
                temp_dict['税前派息'] = unconfirmed_div_df.iloc[0]['税前派息']
                temp_dict['分红点数'] = self.cal_div_point(unconfirmed_div_df.iloc[0]['税前派息'], prediction_type)
                temp_dict['状态'] = '去年有分红，今年已有预案，预测派息日和股权登记日'
                temp_dict['是否预测'] = True

                estimated_df = div_last_year_df.copy().reset_index()
                for i in range(len(estimated_df)):
                    estimated_df.loc[i,'estimated_div_date'] = \
                        estimated_df.loc[i,'派息日'] + relativedelta(years=1) - datetime.timedelta(days=1)
                # 今年没有分红，分红日期参照去年+1年，且>=今天
                if div_this_year_df.empty:
                    estimated_df = estimated_df.loc[estimated_df['estimated_div_date']>self.dt,:]
                    # 若没符合的，预案日期+2个月
                    if estimated_df.empty:
                        estimated_div_dt = self.trading_calendar[self.trading_calendar >=
                            unconfirmed_div_df.iloc[0]['预案公告日'] + relativedelta(months=2)].reset_index(drop=True)[0]
                        temp_series = self.trading_calendar[self.trading_calendar <= estimated_div_dt].reset_index(drop=True)
                        estimated_reg_dt = temp_series.iloc[-1]
                        if estimated_reg_dt <= self.dt + datetime.timedelta(days=7):
                            estimated_reg_dt = np.nan
                            estimated_div_dt = np.nan
                        temp_dict['股权登记日'] = estimated_reg_dt
                        temp_dict['派息日'] = estimated_div_dt
                    else:
                        estimated_div_dt = self.trading_calendar[self.trading_calendar >=
                                            estimated_df.iloc[0]['estimated_div_date']].reset_index(drop=True)[0]
                        temp_series = self.trading_calendar[self.trading_calendar < estimated_div_dt].reset_index(drop=True)
                        estimated_reg_dt = temp_series.iloc[-1]
                        if estimated_reg_dt <= self.dt + datetime.timedelta(days=7):
                            estimated_reg_dt = np.nan
                            estimated_div_dt = np.nan
                        temp_dict['股权登记日'] = estimated_reg_dt
                        temp_dict['派息日'] = estimated_div_dt

                # 今年已有分红，分红日期参照去年+1年，且下次分红日期 >= 最新已确认日期 + 1个月
                else:
                    estimated_df = estimated_df.loc[estimated_df['estimated_div_date'] >
                        div_this_year_df.iloc[-1]['派息日'] + relativedelta(months=1),:]
                    # 若没符合的，预案日期+2个月
                    if estimated_df.empty:
                        estimated_div_dt = self.trading_calendar[self.trading_calendar >=
                            (unconfirmed_div_df.iloc[0]['预案公告日'] + relativedelta(months=2))].reset_index(drop=True)[0]
                        temp_series = self.trading_calendar[self.trading_calendar <= estimated_div_dt].reset_index(drop=True)
                        estimated_reg_dt = temp_series.iloc[-1]
                        if estimated_reg_dt <= self.dt + datetime.timedelta(days=7):
                            estimated_reg_dt = np.nan
                            estimated_div_dt = np.nan
                        temp_dict['股权登记日'] = estimated_reg_dt
                        temp_dict['派息日'] = estimated_div_dt
                    else:
                        estimated_div_dt = self.trading_calendar[self.trading_calendar >=
                            estimated_df.iloc[0]['estimated_div_date']].reset_index(drop=True)[0]
                        temp_series = self.trading_calendar[self.trading_calendar < estimated_div_dt].reset_index(drop=True)
                        estimated_reg_dt = temp_series.iloc[-1]
                        if estimated_reg_dt <= self.dt + datetime.timedelta(days=7):
                            estimated_reg_dt = np.nan
                            estimated_div_dt = np.nan
                        temp_dict['股权登记日'] = estimated_reg_dt
                        temp_dict['派息日'] = estimated_div_dt

                temp_df = pd.DataFrame(temp_dict,index=[0])
                result_df = result_df.append(temp_df, ignore_index=True)

            # 同股息率下分红额未满，预测下一次
            div_sum_last_year = 0
            for i in range(len(div_last_year_df)):
                div_sum_last_year += div_last_year_df.iloc[i]['税前派息']
            div_sum_this_year = 0
            if len(unconfirmed_div_df)>0:
                div_sum_this_year += unconfirmed_div_df.iloc[0]['税前派息']
            if len(div_this_year_df)>0:
                div_sum_this_year += div_this_year_df['税前派息'].sum()

            to_div_times = len(div_last_year_df) - this_year_div_times
            div_amount_prediction = div_sum_last_year/eps_last_year*eps_on_dt - div_sum_this_year
            if div_amount_prediction <= 0 or to_div_times <= 0:
                pass
            else:
                temp_dict = {}
                temp_dict['股票代码'] = self.stock_code_processing
                temp_dict['股票名称'] = self.stock_name_processing
                temp_dict['状态'] = '去年有分红，今年金额未满，预测仍将分红'
                temp_dict['是否预测'] = True
                temp_dict['每股转送'] = None
                temp_dict['预案公告日'] = None
                temp_dict['税前派息'] = div_amount_prediction
                temp_dict['分红点数'] = self.cal_div_point(div_amount_prediction,prediction_type)
                if div_last_year_df.iloc[-1]['派息日']+relativedelta(years=1) <= self.dt:
                    temp_dict['派息日'] = np.nan
                    temp_dict['股权登记日'] = np.nan
                    temp_dict['预案公告日'] = np.nan
                else:
                    estimated_div_dt = self.trading_calendar[self.trading_calendar >=div_last_year_df.iloc[-1]['派息日']
                                            +relativedelta(years=1)-datetime.timedelta(days=1)].reset_index(drop=True)[0]
                    temp_series = self.trading_calendar[self.trading_calendar < estimated_div_dt].reset_index(drop=True)
                    estimated_reg_dt = temp_series.iloc[-1]
                    if estimated_reg_dt <= self.dt + datetime.timedelta(days=7):
                        estimated_reg_dt = np.nan
                        estimated_div_dt = np.nan
                    temp_dict['股权登记日'] = estimated_reg_dt
                    temp_dict['派息日'] = estimated_div_dt
                    temp_dict['预案公告日'] = None
                temp_df = pd.DataFrame(temp_dict,index=[0])
                result_df = result_df.append(temp_df,ignore_index=True)
            return result_df



    def div_prediction(self, dt_input, prediction_type, years_back=2):
        dt_input = dt_input + datetime.timedelta(days=2)
        div_df_list = []
        prediction_df_list = []

        c_list = self.constituent_list_dict[prediction_type]
        self.get_eps_df(c_list)
        #self.eps_df.to_csv('eps.csv',encoding='gbk')

        while years_back > 1:
            dt_start = dt_input - relativedelta(years=years_back) - datetime.timedelta(days=years_back + 1)
            dt_end = dt_start + relativedelta(years=1)
            div_df_list.append(self.get_div_df(prediction_type, dt_start, dt_end))
            years_back = years_back - 1

        div_df_list.append(self.get_div_df(prediction_type, dt_end + datetime.timedelta(1)))

        self.history_div_df = pd.concat(div_df_list, ignore_index=True, sort=True)
        self.history_div_df = self.history_div_df.loc[self.history_div_df.loc[:, '税前派息'] != 0, :].copy()
        self.history_div_df.loc[:, '预案公告日'] = pd.to_datetime(self.history_div_df.loc[:, '预案公告日'], format="%Y%m%d")

        for code in self.constituent_list_dict[prediction_type]:
            self.stock_div_df = self.history_div_df.loc[self.history_div_df.loc[:, '股票代码'] == code, :]
            self.stock_code_processing = code
            self.stock_name_processing = self.get_stock_name(self.stock_code_processing)
            prediction_df_list.append(self.process_prediction(prediction_type))
        return prediction_df_list

    def get_prompt_date(self,dt_input):
        start_date_this_month = datetime.datetime(year=dt_input.date().year,month=dt_input.date().month,day=1)
        if start_date_this_month.weekday() <= 4:
            prompt_date = start_date_this_month + datetime.timedelta(days=(14+(4-start_date_this_month.weekday())))
        else:
            prompt_date = start_date_this_month + datetime.timedelta(days=(21-(start_date_this_month.weekday()-4)))
        return prompt_date

    def get_date_needed(self,dt_input):
        date_needed_list = []
        prompt_date_this_month = self.get_prompt_date(dt_input)
        dt_next_month = dt_input + relativedelta(months=1)
        prompt_date_next_month = self.get_prompt_date(dt_next_month)

        if dt_input <= prompt_date_this_month:
            near_prompt = prompt_date_this_month
            if dt_input <= (prompt_date_this_month - datetime.timedelta(days=7)):
                friday_before_near_prompt = (prompt_date_this_month - datetime.timedelta(days=7))
                date_needed_list.append(friday_before_near_prompt)
            next_prompt = prompt_date_next_month
            friday_before_next_prompt = prompt_date_next_month - datetime.timedelta(days=7)
        else:
            friday_before_near_prompt = prompt_date_next_month - datetime.timedelta(days=7)
            date_needed_list.append(friday_before_near_prompt)
            near_prompt = prompt_date_next_month
            next_prompt = self.get_prompt_date(near_prompt+relativedelta(months=1))
            friday_before_next_prompt = next_prompt - datetime.timedelta(days=7)
        date_needed_list.extend([near_prompt,friday_before_next_prompt,next_prompt])

        month_exceed = near_prompt.date().month % 3
        if month_exceed == 2:
            date_needed_list.extend(
                [self.get_prompt_date(near_prompt+relativedelta(months=4))-datetime.timedelta(days=7),
                 self.get_prompt_date(near_prompt+relativedelta(months=4)),
                 self.get_prompt_date(near_prompt+relativedelta(months=7))-datetime.timedelta(days=7),
                 self.get_prompt_date(near_prompt+relativedelta(months=7))])
        else:
            date_needed_list.extend(
                [self.get_prompt_date(near_prompt+relativedelta(months=3-month_exceed))-datetime.timedelta(days=7),
                 self.get_prompt_date(near_prompt+relativedelta(months=3-month_exceed)),
                 self.get_prompt_date(near_prompt+relativedelta(months=6-month_exceed))-datetime.timedelta(days=7),
                 self.get_prompt_date(near_prompt+relativedelta(months=6-month_exceed))])

        return date_needed_list

    def modify(self,raw_df,revise_filepath):
        revising_df = pd.read_csv(revise_filepath,encoding='utf-8',index_col=0)
        revising_df['预案公告日'] = pd.to_datetime(revising_df['预案公告日'],format='%Y-%m-%d %H:%M:%S')
        revising_df['股权登记日'] = pd.to_datetime(revising_df['股权登记日'],format='%Y-%m-%d %H:%M:%S')
        revising_df['派息日'] = pd.to_datetime(revising_df['派息日'],format='%Y-%m-%d %H:%M:%S')
        print(revising_df)
        for idx,row in revising_df.iterrows():
            if raw_df.loc[idx,'是否预测']== False:
                pass
            else:
                if row['税前派息']==row['税前派息']:
                    raw_df.loc[idx,'税前派息'] = row['税前派息']
                if row['分红点数']==row['分红点数']:
                    raw_df.loc[idx,'分红点数'] = row['分红点数']
                if row['每股转送']==row['每股转送']:
                    raw_df.loc[idx,'每股转送'] = row['每股转送']
                if row['预案公告日']==row['预案公告日']:
                    raw_df.loc[idx,'预案公告日'] = row['预案公告日']
                if row['股权登记日']==row['股权登记日']:
                    raw_df.loc[idx,'股权登记日'] = row['股权登记日']
                if row['派息日']==row['派息日']:
                    raw_df.loc[idx,'派息日'] = row['派息日']
        return raw_df

    def run_daily(self):
        self.get_trading_calendar()

        local_prefix = 'D:/div_statistics/'
        remote_utf8_prefix = 'indexBonus/'
        remote_gbk_prefix = 'indexBonus_gbk/'
        remote_modify_prefix = 'indexBonus_modify/'

        print(self.dt)
        self.get_constituent_df(self.dt)
        self.get_index_close(self.dt)
        date_needed = self.get_date_needed(self.dt)

        to_do_list = ["IH","IF","IC"]
        #to_do_list = ['IF']
        code_dict = {"IH":"000016","IF":"000300","IC":"000905"}
        summary_dict = {}

        for index_type in to_do_list:
            print(index_type,'is processing...')

            filename = code_dict[index_type]+'bonus' + self.last_trading_date.strftime('%Y%m%d')+'.csv'
            df = pd.concat(self.div_prediction(self.dt, index_type), ignore_index=True, sort=True)
            df.sort_values('派息日', ascending=True, inplace=True)
            df = df.loc[:, ['股票代码', '股票名称', '状态', '税前派息', '分红点数',
                '每股转送', '预案公告日', '股权登记日', '派息日','是否预测']]
            df.to_csv(local_prefix + 'raw_utf8/' + filename,encoding='utf-8')
            df.to_csv(local_prefix + 'raw_gbk/' + filename, encoding='gbk')

            # 编制需修订清单
            revise_filename = index_type + '_to_revise' + '.csv'
            df_to_revise = df.loc[df['是否预测']==True,['股票代码', '股票名称', '状态', '税前派息', '分红点数',
                '每股转送', '预案公告日', '股权登记日', '派息日']]
            df_to_revise.to_csv(local_prefix + 'to_revise/' + revise_filename,encoding='utf-8')
            self.ftp_client.upload_file(remote_modify_prefix + revise_filename, local_prefix + 'to_revise/' + revise_filename)

            # 需修订清单的历史数据
            history_df = self.get_all_historical_data(list(df_to_revise['股票代码'].unique()))
            history_filename = index_type + '_history' + '.csv'
            history_df.to_csv(local_prefix + 'history/' + history_filename,encoding='gbk')
            self.ftp_client.upload_file(remote_modify_prefix + history_filename, local_prefix + 'history/' + history_filename)

            # 修订
            path_prefix = 'D:/div_statistics/revising_file/'
            revise_filename = path_prefix + index_type + '_modify' + '.csv'
            modified_df = self.modify(df,revise_filename)
            #modified_df.to_csv('111.csv',encoding='gbk')
            modified_df.to_csv(local_prefix + 'modified_utf8/' + filename,encoding='utf-8')
            self.ftp_client.upload_file(remote_utf8_prefix + filename, local_prefix + 'modified_utf8/' + filename)
            modified_df.to_csv(local_prefix + 'modified_gbk/' + filename,encoding='gbk')
            self.ftp_client.upload_file(remote_gbk_prefix + filename, local_prefix + 'modified_gbk/' + filename)

            # 编制summary
            summary_dict[index_type]={}
            modified_df = modified_df.dropna(subset=['派息日']).sort_values(by='派息日')
            modified_df.set_index("派息日",inplace=True)
            for date in date_needed:
                summary_dict[index_type][date] = \
                    modified_df.loc[self.dt+datetime.timedelta(days=1):date,'分红点数'].sum()
        summary_df = pd.DataFrame(summary_dict)
        summary_filename = 'bonusSummary'+self.last_trading_date.strftime('%Y%m%d')+'.csv'
        summary_df.to_csv(local_prefix + 'summary/' + summary_filename,encoding='gbk')

        print('dividend is done!')

        right_issue_dt = self.dt - relativedelta(months=2)
        self.get_right_issue_df(right_issue_dt)

        filename = 'right_issue' + self.dt.strftime('%Y%m%d') + '.xls'
        writer = pd.ExcelWriter(local_prefix + 'right_issue/' + filename)
        for index_type in to_do_list:
            sheet_name = index_type + '配股'
            self.right_issue_df.loc[self.right_issue_df['股票代码'].isin(self.constituent_list_dict[index_type]), :] \
                .reset_index(drop=True).to_excel(writer, sheet_name=sheet_name)
        print('right issue is done!')
        writer.save()
        self.ftp_client.upload_file(remote_gbk_prefix + filename, local_prefix + 'right_issue/' + filename)


    def run_prediction(self):
        print('start new constituent processing')

        self.get_trading_calendar()
        local_prefix = 'D:/div_statistics/'
        remote_utf8_prefix = 'indexBonus_R/'
        remote_gbk_prefix = 'indexBonus_R_gbk/'
        remote_modify_prefix = 'indexBonus_R_modify/'

        self.get_constituent_from_csv()
        self.get_index_close(self.dt)
        date_needed = self.get_date_needed(self.dt)

        to_do_list = ["IH","IF","IC"]
        #to_do_list = ['IF']
        code_dict = {"IH":"000016","IF":"000300","IC":"000905"}
        summary_dict = {}

        for index_type in to_do_list:
            print(index_type,'is processing...')

            filename = code_dict[index_type]+'bonus_R' + self.last_trading_date.strftime('%Y%m%d')+'.csv'
            df = pd.concat(self.div_prediction(self.dt, index_type), ignore_index=True, sort=True)
            df.sort_values('派息日', ascending=True, inplace=True)
            df = df.loc[:, ['股票代码', '股票名称', '状态', '税前派息', '分红点数',
                '每股转送', '预案公告日', '股权登记日', '派息日','是否预测']]
            df.to_csv(local_prefix + 'R_raw_utf8/' + filename,encoding='utf-8')
            df.to_csv(local_prefix + 'R_raw_gbk/' + filename, encoding='gbk')

            # 编制需修订清单
            revise_filename = index_type + '_to_revise' + '.csv'
            df_to_revise = df.loc[df['是否预测']==True,['股票代码', '股票名称', '状态', '税前派息', '分红点数',
                '每股转送', '预案公告日', '股权登记日', '派息日']]
            df_to_revise.to_csv(local_prefix + 'R_to_revise/' + revise_filename,encoding='utf-8')
            self.ftp_client.upload_file(remote_modify_prefix + revise_filename, local_prefix + 'R_to_revise/' + revise_filename)

            # 需修订清单的历史数据
            history_df = self.get_all_historical_data(list(df_to_revise['股票代码'].unique()))
            history_filename = index_type + '_history' + '.csv'
            history_df.to_csv(local_prefix + 'R_history/' + history_filename,encoding='gbk')
            self.ftp_client.upload_file(remote_modify_prefix + history_filename, local_prefix + 'R_history/' + history_filename)

            # 修订
            path_prefix = 'D:/div_statistics/R_revising_file/'
            revise_filename = path_prefix + index_type + '_modify' + '.csv'
            modified_df = self.modify(df,revise_filename)
            #modified_df.to_csv('111.csv',encoding='gbk')
            modified_df.to_csv(local_prefix + 'R_modified_utf8/' + filename,encoding='utf-8')
            self.ftp_client.upload_file(remote_utf8_prefix + filename, local_prefix + 'R_modified_utf8/' + filename)
            modified_df.to_csv(local_prefix + 'R_modified_gbk/' + filename,encoding='gbk')
            self.ftp_client.upload_file(remote_gbk_prefix + filename, local_prefix + 'R_modified_gbk/' + filename)
            
            # 编制summary
            summary_dict[index_type]={}
            modified_df = modified_df.dropna(subset=['派息日']).sort_values(by='派息日')
            modified_df.set_index("派息日",inplace=True)
            for date in date_needed:
                summary_dict[index_type][date] = \
                    modified_df.loc[self.dt+datetime.timedelta(days=1):date,'分红点数'].sum()
        
        summary_df = pd.DataFrame(summary_dict)
        summary_filename = 'R_bonusSummary'+self.last_trading_date.strftime('%Y%m%d')+'.csv'
        summary_df.to_csv(local_prefix + 'summary/' + summary_filename,encoding='gbk')

        print('R_dividend is done!')



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pd.set_option('display.max_columns', None)

    div = div_pred_statistic()
    div.run_daily()
    div.run_prediction()