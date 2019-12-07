from csindex_ftp_down import *
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import os
import cx_Oracle as oracle

class div_pred_statistic:
    def __init__(self):
        self.dir_prefix = "d:/basket/"
        self.IH_prefix = "ftp000016weightnextday"
        self.IF_prefix = "ftp000300weightnextday"
        self.IC_prefix = "ftp000905weightnextday"
        self.postfix = ".xls"
        self.DEFAULT_LOCAL_DIR = "d:/basket/ftp"

        today = datetime.datetime.today().date()
        self.dt = datetime.datetime(year=today.year, month=today.month, day=today.day) - datetime.timedelta(days=1)

        self.this_year = datetime.datetime.now().date().year
        self.this_year_start_date = datetime.datetime(year=self.this_year, month=1, day=1)
        self.last_year_start_date = datetime.datetime(year=self.this_year-1, month=1, day=1)
        self.last_year_end_date = datetime.datetime(year=self.this_year-1, month=12, day=31)

        self.ip_address = '172.17.21.3'
        self.port = '1521'
        self.username = 'yspread'
        self.pwd = 'Y*iaciej123456'
        self.conn = oracle.connect(
            self.username + "/" + self.pwd + "@" + self.ip_address + ":" + self.port + "/" + "WDZX")
        self.curs = self.conn.cursor()


    def get_constituent_df(self,dt_input):
        self.constituent_df_dict = {}
        self.constituent_list_dict = {}
        str_dt = dt_input.strftime("%Y%m%d")
        self.IH_path = self.dir_prefix + self.IH_prefix + str_dt + self.postfix
        self.IF_path = self.dir_prefix + self.IF_prefix + str_dt + self.postfix
        self.IC_path = self.dir_prefix + self.IC_prefix + str_dt + self.postfix

        WeightnextdayManager.down_and_up(str_dt, self.DEFAULT_LOCAL_DIR, 0)

        if os.path.exists(self.IH_path):
            IH_df = pd.read_excel(self.IH_path).iloc[:,[0,1,2,3,4,5,7,11,12,13,16]]
            IF_df = pd.read_excel(self.IF_path).iloc[:,[0,1,2,3,4,5,7,11,12,13,16]]
            IC_df = pd.read_excel(self.IC_path).iloc[:,[0,1,2,3,4,5,7,11,12,13,16]]
            IH_df.columns = ['Effective Date','Index Code','Index Name','Index Name(Eng)',
                             'Constituent Code','Constituent Name','Exchange','Cap Factor',
                             'Close','Reference Open Price for Next Trading Day','Weight']
            IF_df.columns = ['Effective Date', 'Index Code', 'Index Name', 'Index Name(Eng)',
                             'Constituent Code', 'Constituent Name', 'Exchange', 'Cap Factor',
                             'Close', 'Reference Open Price for Next Trading Day', 'Weight']
            IC_df.columns = ['Effective Date', 'Index Code', 'Index Name', 'Index Name(Eng)',
                             'Constituent Code', 'Constituent Name', 'Exchange', 'Cap Factor',
                             'Close', 'Reference Open Price for Next Trading Day', 'Weight']

            IH_df['Index Code'] = IH_df.apply(lambda row: str(row['Index Code']).zfill(6),axis=1)
            IH_df['Constituent Code'] = IH_df.apply(lambda row: str(row['Constituent Code']).zfill(6) + '.SH' \
                                        if str(row['Constituent Code']).zfill(6)[0]=='6' else \
                                        str(row['Constituent Code']).zfill(6)+'.SZ',axis=1)
            IF_df['Index Code'] = IF_df.apply(lambda row: str(row['Index Code']).zfill(6), axis=1)
            IF_df['Constituent Code'] = IF_df.apply(lambda row: str(row['Constituent Code']).zfill(6) + '.SH' \
                                        if str(row['Constituent Code']).zfill(6)[0] == '6' else \
                                        str(row['Constituent Code']).zfill(6) + '.SZ', axis=1)
            IC_df['Index Code'] = IC_df.apply(lambda row: str(row['Index Code']).zfill(6), axis=1)
            IC_df['Constituent Code'] = IC_df.apply(lambda row: str(row['Constituent Code']).zfill(6) + '.SH' \
                                        if str(row['Constituent Code']).zfill(6)[0] == '6' else \
                                        str(row['Constituent Code']).zfill(6) + '.SZ', axis=1)

            self.constituent_list_dict['IH'] = list(IH_df.loc[:,'Constituent Code'])
            self.constituent_list_dict['IF'] = list(IF_df.loc[:,'Constituent Code'])
            self.constituent_list_dict['IC'] = list(IC_df.loc[:,'Constituent Code'])
            self.constituent_list_dict['all'] = list(IF_df.loc[:,'Constituent Code'])+list(IC_df.loc[:,'Constituent Code'])

            self.constituent_df_dict['IH'] = IH_df
            self.constituent_df_dict['IF'] = IF_df
            self.constituent_df_dict['IC'] = IC_df
            self.constituent_df_dict['all'] = pd.concat([IF_df,IC_df],ignore_index=True)

            print('constituent got!')
            return True
        else:
            return False


    def get_trading_calendar(self):
        get_calendar_sentense = "select TRADE_DAYS from wind_filesync.AShareCalendar " \
                                "where trade_days >= 20160101"
        self.curs.execute(get_calendar_sentense)
        fetch_data = self.curs.fetchall()
        trading_days_list = []
        for data in fetch_data:
            trading_days_list.append(data[0])
        self.trading_calendar = pd.to_datetime(pd.Series(trading_days_list))


    def get_stock_name(self,code):
        df = self.constituent_df_dict['all']
        return list(df.loc[df.loc[:,'Constituent Code']==code,'Constituent Name'])[0]


    def add_stock_name(self,df):
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


    def get_div_df(self,prediction_type,dt_start_input,dt_end_input=None):
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
        self.div_df = pd.DataFrame(fetch_data, columns=['股票代码', '每股转送', '税前派息', '方案进度','股权登记日',
                                                        '除权除息日','派息日', '分红实施公告日','预案公告日', '分红年度'])
        self.div_df = self.add_stock_name(self.div_df).loc[:,['股票代码','股票名称','每股转送','税前派息','方案进度',
                                                              '股权登记日','除权除息日','派息日','分红实施公告日','预案公告日','分红年度']]
        return self.div_df


    def get_right_issue_df(self,dt_input):
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
                            '配股计划数量（万股）','配股实际数量（万股）', '募集资金（元）', '股权登记日', '除权日',
                            '配股上市日','缴款起始日', '缴款终止日', '配股实施公告日', '配股结果公告日', '配股年度'])
        self.right_issue_df = self.add_stock_name(self.right_issue_df).loc[:,['股票代码','股票名称','方案进度',
                            '配股价格','配股比例','配股计划数量（万股）','配股实际数量（万股）','募集资金（元）',
                            '股权登记日','除权日','配股上市日','缴款起始日','缴款终止日','配股实施公告日','配股结果公告日',
                            '配股年度']]
        return


    def get_index_close(self,dt_input):
        sql_sentence = "select s_info_windcode,trade_dt,s_dq_close " \
                       "from wind_filesync.AIndexEODPrices " \
                       "where s_info_windcode in ('000016.SH','000300.SH','000905.SH') and trade_dt = :dt"
        str_dt = dt_input.strftime("%Y%m%d")
        self.curs.execute(sql_sentence, dt=str_dt)
        fetch_data = self.curs.fetchall()
        self.index_close_df = pd.DataFrame(fetch_data,columns=['股票代码','日期','收盘价'])
        self.index_close_df['日期'] = pd.to_datetime(self.index_close_df['日期'],format="%Y%m%d")
        self.index_close_df.set_index('股票代码',inplace=True)
        return

    def get_eps_df(self,code_list):
        print('fetching eps data...')
        sql_sentence = "select s_info_windcode,trade_dt,NET_PROFIT_PARENT_COMP_TTM,TOT_SHR_TODAY " \
                       "from wind_filesync.AShareEODDerivativeIndicator " \
                       "where trade_dt >20160101 and s_info_windcode in " \
                       + str(tuple(code_list))
        self.curs.execute(sql_sentence)
        fetch_data = self.curs.fetchall()
        self.eps_df = pd.DataFrame(fetch_data, columns=['code','date','net_profit','total_shares'])
        self.eps_df['date'] = pd.to_datetime(self.eps_df['date'],format="%Y%m%d")
        self.eps_df.sort_values(by='date',ascending=True,inplace=True)
        self.eps_df.set_index('date',inplace=True)
        self.eps_df['eps'] = self.eps_df['net_profit']/self.eps_df['total_shares']
        print('eps data got!')
        return

    def cal_div_point(self,div_pre_tax,prediction_type):
        constituent_df = self.constituent_df_dict[prediction_type]
        index_code_dict = {'IH': '000016.SH', 'IF': '000300.SH', 'IC': '000905.SH'}
        div_point_value = div_pre_tax / \
                          list(constituent_df.loc[constituent_df.loc[:,'Constituent Code']==self.stock_code_processing,'Close'])[0] * \
                          list(constituent_df.loc[constituent_df.loc[:,'Constituent Code']==self.stock_code_processing,'Weight'])[0] * \
                          self.index_close_df.loc[index_code_dict[prediction_type],'收盘价'] /100
        return div_point_value


    def process_prediction(self,prediction_type):
        result_df = pd.DataFrame(columns=['股票代码','股票名称','状态','每股转送','股权登记日','派息日','预案公告日',
                                          '税前派息','分红点数'])
        stock_eps_df = self.eps_df.loc[self.eps_df.loc[:, 'code'] == self.stock_code_processing, :]
        unconfirmed_div_df = self.stock_div_df.loc[self.stock_div_df['派息日'].isna(),:].copy()
        confirmed_div_df = self.stock_div_df.loc[self.stock_div_df['派息日'].notna(), :].copy()
        confirmed_div_df.loc[:, '派息日'] = pd.to_datetime(confirmed_div_df.loc[:, '派息日'], format="%Y%m%d")
        confirmed_div_df.loc[:, '股权登记日'] = pd.to_datetime(confirmed_div_df.loc[:, '股权登记日'], format="%Y%m%d")
        div_this_year_df = confirmed_div_df.loc[confirmed_div_df.loc[:, '派息日'] >= self.this_year_start_date, :]
        div_last_year_df = confirmed_div_df.loc[(confirmed_div_df.loc[:, '派息日'] >= self.last_year_start_date) &
                                                (confirmed_div_df.loc[:, '派息日'] <= self.last_year_end_date), :]
        this_year_div_times = len(unconfirmed_div_df) + len(div_this_year_df)

        temp_dict = {}
        temp_dict['股票代码'] = self.stock_code_processing
        temp_dict['股票名称'] = self.stock_name_processing

        # 去年没分，今年不论有没有，都不预测
        if div_last_year_df.empty:
            if this_year_div_times == 0:
                temp_dict['状态'] = '去年无分红，今年无预案，预测不分红'
                temp_dict['每股转送'] = None
                temp_dict['股权登记日'] = None
                temp_dict['派息日'] = None
                temp_dict['预案公告日'] = None
                temp_dict['税前派息'] = 0
                temp_dict['分红点数'] = 0
            else:
                if not unconfirmed_div_df.empty:
                    estimated_reg_dt = self.trading_calendar[self.trading_calendar >=
                        unconfirmed_div_df.iloc[0]['预案公告日'] + relativedelta(months=2) - datetime.timedelta(days=1)]. \
                        reset_index(drop=True)[0]
                    estimated_div_dt = self.trading_calendar[self.trading_calendar >=
                        unconfirmed_div_df.iloc[0]['预案公告日'] + relativedelta(months=2)]. \
                        reset_index(drop=True)[0]
                    temp_dict['状态'] = '去年无分红，今年已有预案，预测派息日和股权登记日'
                    temp_dict['每股转送'] = unconfirmed_div_df.iloc[0]['每股转送']
                    temp_dict['股权登记日'] = estimated_reg_dt
                    temp_dict['派息日'] = estimated_div_dt
                    temp_dict['预案公告日'] = unconfirmed_div_df.iloc[0]['预案公告日']
                    temp_dict['税前派息'] = unconfirmed_div_df.iloc[0]['税前派息']
                    temp_dict['分红点数'] = self.cal_div_point(unconfirmed_div_df.iloc[0]['税前派息'],prediction_type)
                    temp_df = pd.DataFrame(temp_dict,index=[0])
                    result_df = result_df.append(temp_df, ignore_index=True)
                if not div_this_year_df.empty:
                    for i in range(len(div_this_year_df)):
                        temp_dict['状态'] = '去年无分红，今年已有公告，实际派息日和股权登记日'
                        temp_dict['每股转送'] = div_this_year_df.iloc[i]['每股转送']
                        temp_dict['股权登记日'] = div_this_year_df.iloc[i]['股权登记日']
                        temp_dict['派息日'] = div_this_year_df.iloc[i]['派息日']
                        temp_dict['预案公告日'] = div_this_year_df.iloc[i]['预案公告日']
                        temp_dict['税前派息'] = div_this_year_df.iloc[i]['税前派息']
                        temp_dict['分红点数'] = self.cal_div_point(div_this_year_df.iloc[i]['税前派息'], prediction_type)
                        temp_df = pd.DataFrame(temp_dict, index=[0])
                        result_df = result_df.append(temp_df, ignore_index=True)








        if unconfirmed_div_df.empty:
            pass
        else:
            if div_last_year_df.empty:
                estimated_reg_dt = self.trading_calendar[self.trading_calendar >=
                                    unconfirmed_div_df.iloc[0]['预案公告日']+relativedelta(months=2)-datetime.timedelta(days=1)].\
                                    reset_index(drop=True)[0]
                estimated_div_dt = self.trading_calendar[self.trading_calendar >=
                                    unconfirmed_div_df.iloc[0]['预案公告日']+relativedelta(months=2)].\
                                    reset_index(drop=True)[0]
            else:
                estimated_reg_dt = self
                estimated_div_dt =
            if 1 <=(unconfirmed_div_df.iloc[0]['预案公告日']+relativedelta(months=2)).weekday() <=4 :
                # 周二~周五
                unconfirmed_div_df['派息日'] = (unconfirmed_div_df.iloc[0]['预案公告日']
                                            + relativedelta(months=2)).strftime('%Y%m%d')
                unconfirmed_div_df['股权登记日'] = (unconfirmed_div_df.iloc[0]['预案公告日']+relativedelta(months=2)
                                             - datetime.timedelta(days=1)).strftime('%Y%m%d')

            elif (unconfirmed_div_df.iloc[0]['预案公告日']+relativedelta(months=2)).weekday() == 0:
                # 周一
                unconfirmed_div_df['派息日'] = (unconfirmed_div_df.iloc[0]['预案公告日']
                                             + relativedelta(months=2)).strftime('%Y%m%d')
                unconfirmed_div_df['股权登记日'] = (unconfirmed_div_df.iloc[0]['预案公告日']+relativedelta(months=2)
                                             - datetime.timedelta(days=3)).strftime('%Y%m%d')

            elif (unconfirmed_div_df.iloc[0]['预案公告日']+relativedelta(months=2)).weekday() == 5:
                # 周六
                unconfirmed_div_df['派息日'] = (unconfirmed_div_df.iloc[0]['预案公告日']
                                            + datetime.timedelta(days=2)+ relativedelta(months=2)).strftime('%Y%m%d')
                unconfirmed_div_df['股权登记日'] = (unconfirmed_div_df.iloc[0]['预案公告日']+relativedelta(months=2)
                                             - datetime.timedelta(days=1)).strftime('%Y%m%d')
            else:
                # 周日
                unconfirmed_div_df['派息日'] = (unconfirmed_div_df.iloc[0]['预案公告日']
                                            + datetime.timedelta(days=1) + relativedelta(months=2)).strftime('%Y%m%d')
                unconfirmed_div_df['股权登记日'] = (unconfirmed_div_df.iloc[0]['预案公告日']+relativedelta(months=2)
                                             - datetime.timedelta(days=2)).strftime('%Y%m%d')

            unconfirmed_div_df['预案公告日'] = unconfirmed_div_df.iloc[0]['预案公告日'].strftime('%Y%m%d')
            unconfirmed_div_df['状态'] = '明确(派息日未定)'
            unconfirmed_div_df['分红点数'] = self.cal_div_point(unconfirmed_div_df.iloc[0]['税前派息'],prediction_type)

            result_df = result_df.append(unconfirmed_div_df.loc[:,['股票代码','股票名称','状态','每股转送','股权登记日',
                                        '派息日','税前派息','预案公告日','分红点数']],ignore_index=True,sort=True)


        if div_last_year_df.empty :
            # 去年没分，今年不论有没有，都不预测
            if div_this_year_df.empty:
                return result_df
            else:
                for i in range(len(div_this_year_df)):
                    temp_df = div_this_year_df.iloc[i][['股票代码','股票名称','每股转送','税前派息']].copy()
                    temp_df['预案公告日'] = div_this_year_df.iloc[i]['预案公告日'].strftime('%Y%m%d')
                    temp_df['派息日'] = div_this_year_df.iloc[i]['派息日'].strftime('%Y%m%d')
                    temp_df['股权登记日'] = div_this_year_df.iloc[i]['股权登记日'].strftime('%Y%m%d')
                    temp_df['状态'] = '明确'
                    temp_df['分红点数'] = self.cal_div_point(temp_df['税前派息'],prediction_type)
                    result_df = result_df.append(temp_df,ignore_index=True)
                return result_df

        elif len(div_last_year_df) == 1:
            if this_year_div_times >= 1:
            # 去年一次，今年已有>=一次，不预测
                if len(div_this_year_df) >=1:
                    for i in range(len(div_this_year_df)):
                        temp_df = div_this_year_df.iloc[i][['股票代码','股票名称','每股转送','税前派息']].copy()
                        temp_df['状态'] = '明确'
                        temp_df['预案公告日'] = div_this_year_df.iloc[i]['预案公告日'].strftime('%Y%m%d')
                        temp_df['股权登记日'] = div_this_year_df.iloc[i]['股权登记日'].strftime('%Y%m%d')
                        temp_df['派息日'] = div_this_year_df.iloc[i]['派息日'].strftime('%Y%m%d')
                        temp_df['分红点数'] = self.cal_div_point(temp_df['税前派息'],prediction_type)
                        result_df = result_df.append(temp_df,ignore_index=True)
                    return result_df
                else:
                    return result_df

            else:
            # 去年一次，今年没有，预测
                temp_dict = {}
                eps_df_before_dt = stock_eps_df.loc[:self.dt,:]
                if eps_df_before_dt.empty:
                    eps_on_dt = 1
                    eps_last_year = 1
                else:
                    eps_on_dt = eps_df_before_dt.iloc[-1]['eps']
                    eps_date = eps_df_before_dt.index[-1]
                    eps_date_last_year = eps_date - relativedelta(years=1)
                    if eps_df_before_dt.loc[:eps_date_last_year,:].empty:
                        eps_on_dt =1
                        eps_last_year = 1
                    else:
                        eps_last_year = eps_df_before_dt.loc[:eps_date_last_year,:].iloc[-1]['eps']

                div_on_hist_date = div_last_year_df.iloc[0]['税前派息']
                if eps_last_year <= 0 and eps_on_dt >= 0:
                    div_amount_prediction = div_on_hist_date
                elif eps_last_year > 0 and eps_on_dt > 0:
                    div_amount_prediction = div_on_hist_date/eps_last_year * eps_on_dt
                else:
                    div_amount_prediction = 0

                hist_div_date = div_last_year_df.iloc[0]['派息日']
                if hist_div_date + relativedelta(years=1) <= datetime.datetime.now():
                    temp_dict['派息日'] = '对应日期已过，待定'
                    temp_dict['股权登记日'] = '对应日期已过，待定'
                else:
                    if 1<=(hist_div_date + relativedelta(years=1)).weekday() <= 4:
                        temp_dict['派息日'] = (hist_div_date + relativedelta(years=1)).strftime('%Y%m%d')
                        temp_dict['股权登记日'] = (hist_div_date-datetime.timedelta(days=1)
                                              +relativedelta(years=1)).strftime('%Y%m%d')

                    elif (hist_div_date + relativedelta(years=1)).weekday() == 0:
                        temp_dict['派息日'] = (hist_div_date + relativedelta(years=1)).strftime('%Y%m%d')
                        temp_dict['股权登记日'] = (hist_div_date - datetime.timedelta(days=3)
                                              + relativedelta(years=1)).strftime('%Y%m%d')

                    elif (hist_div_date + relativedelta(years=1)).weekday() == 5:
                        temp_dict['派息日'] = (hist_div_date + datetime.timedelta(days=2)
                                                        +relativedelta(years=1)).strftime('%Y%m%d')
                        temp_dict['股权登记日'] = (hist_div_date - datetime.timedelta(days=1)
                                            + relativedelta(years=1)).strftime('%Y%m%d')
                    else:
                        temp_dict['派息日'] = (hist_div_date + datetime.timedelta(days=1)
                                                        + relativedelta(years=1)).strftime('%Y%m%d')
                        temp_dict['股权登记日'] = (hist_div_date - datetime.timedelta(days=2)
                                              + relativedelta(years=1)).strftime('%Y%m%d')
                temp_dict['股票代码'] = self.stock_code_processing
                temp_dict['股票名称'] = self.stock_name_processing
                temp_dict['状态'] = '预测'
                temp_dict['每股转送'] = '待定'
                temp_dict['税前派息'] = div_amount_prediction

                if div_amount_prediction == 0:
                    temp_dict['派息日'] = None
                    temp_dict['预案公告日'] = None
                    temp_dict['股权登记日'] = None
                    temp_dict['分红点数'] = 0
                else:
                    temp_dict['预案公告日'] = '待定'
                    temp_dict['分红点数'] = self.cal_div_point(div_amount_prediction,prediction_type)

                temp_df = pd.DataFrame(temp_dict,index=[0])
                result_df = result_df.append(temp_df,ignore_index=True)
                return result_df

        else:
        # 去年>1次
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
            if this_year_div_times == 0:
            # 去年>=1次，今年还没有，预测今年次数与去年相同
                for i in range(len(div_last_year_df)):
                    temp_dict = {}
                    div_amount_prediction = 0
                    div_on_hist_date = div_last_year_df.iloc[i]['税前派息']
                    if eps_last_year < 0 and eps_on_dt > 0:
                        div_amount_prediction = div_amount_prediction
                    elif eps_last_year > 0 and eps_on_dt > 0:
                        div_amount_prediction = div_on_hist_date / eps_last_year * eps_on_dt
                    else:
                        div_amount_prediction = 0
                    if div_amount_prediction == 0:
                        temp_dict['股票代码'] = self.stock_code_processing
                        temp_dict['股票名称'] = self.stock_name_processing
                        temp_dict['状态'] = '预测'
                        temp_dict['每股转送'] = '待定'
                        temp_dict['税前派息'] = 0
                        temp_dict['派息日'] = None
                        temp_dict['预案公告日'] = None
                        temp_dict['股权登记日'] = None
                        temp_dict['分红点数'] = 0
                    else:
                        temp_dict['股票代码'] = self.stock_code_processing
                        temp_dict['股票名称'] = self.stock_name_processing
                        temp_dict['状态'] = '预测'
                        temp_dict['每股转送'] = '待定'
                        temp_dict['税前派息'] = div_amount_prediction
                        hist_div_date = div_last_year_df.iloc[i]['派息日']
                        if hist_div_date + relativedelta(years=1) <= datetime.datetime.now():
                            temp_dict['派息日'] = '对应日期已过，待定'
                        else:
                            if 1 <= (hist_div_date + relativedelta(years=1)).weekday() <= 4:
                                temp_dict['派息日'] = (hist_div_date + relativedelta(years=1)).strftime('%Y%m%d')
                                temp_dict['股权登记日'] = (hist_div_date + relativedelta(years=1) -
                                                      datetime.timedelta(days=1)).strftime('%Y%m%d')

                            elif (hist_div_date + relativedelta(years=1)).weekday() == 0:
                                temp_dict['派息日'] = (hist_div_date + relativedelta(years=1)).strftime('%Y%m%d')
                                temp_dict['股权登记日'] = (hist_div_date + relativedelta(years=1) -
                                                      datetime.timedelta(days=3)).strftime('%Y%m%d')

                            elif (hist_div_date + relativedelta(years=1)).weekday() == 5:
                                temp_dict['派息日'] = (hist_div_date + datetime.timedelta(days=2)
                                                                + relativedelta(years=1)).strftime('%Y%m%d')
                                temp_dict['股权登记日'] = (hist_div_date + relativedelta(years=1) -
                                                      datetime.timedelta(days=1)).strftime('%Y%m%d')
                            else:
                                temp_dict['派息日'] = (hist_div_date + datetime.timedelta(days=1)
                                                                + relativedelta(years=1)).strftime('%Y%m%d')
                                temp_dict['股权登记日'] = (hist_div_date + relativedelta(years=1) -
                                                      datetime.timedelta(days=2)).strftime('%Y%m%d')

                        temp_dict['预案公告日'] = '待定'
                        temp_dict['分红点数'] = self.cal_div_point(div_amount_prediction,prediction_type)

                    temp_df = pd.DataFrame(temp_dict,index=[0])
                    result_df = result_df.append(temp_df,ignore_index=True,sort=True)
                return result_df

            else:
            # 去年一次以上，今年已有一次及以上，比较eps
                div_sum = 0
                hist_div_eps_sum = 0
                for i in range(len(div_last_year_df)):
                    div_on_hist_date = div_last_year_df.iloc[i]['税前派息']
                    if eps_last_year <=0 and eps_on_dt >= 0:
                        hist_div_eps_sum += div_on_hist_date
                    if eps_last_year > 0 and eps_on_dt > 0:
                        hist_div_eps_sum += div_on_hist_date/eps_last_year * eps_on_dt
                if not unconfirmed_div_df.empty:
                    div_on_dt = unconfirmed_div_df.iloc[0]['税前派息']
                    div_sum += div_on_dt
                for i in range(len(div_this_year_df)):
                    div_on_dt = div_this_year_df.iloc[i]['税前派息']
                    temp_df = div_this_year_df.iloc[i][['股票代码', '股票名称', '每股转送', '税前派息']]

                    temp_df['股权登记日'] = div_this_year_df.iloc[i]['股权登记日'].strftime('%Y%m%d')
                    temp_df['预案公告日'] = div_this_year_df.iloc[i]['预案公告日'].strftime('%Y%m%d')
                    temp_df['派息日'] = div_this_year_df.iloc[i]['派息日'].strftime('%Y%m%d')
                    temp_df['状态'] = '明确'
                    temp_df['分红点数'] = self.cal_div_point(temp_df['税前派息'], prediction_type)
                    result_df = result_df.append(temp_df, ignore_index=True,sort=True)
                    div_sum += div_on_dt
                if div_sum >= hist_div_eps_sum :
                    pass
                else:
                    div_amount_prediction = hist_div_eps_sum - div_sum
                    temp_dict = {}
                    temp_dict['股票代码'] = self.stock_code_processing
                    temp_dict['股票名称'] = self.stock_name_processing
                    temp_dict['状态'] = '预测'
                    temp_dict['每股转送'] = '待定'
                    temp_dict['税前派息'] = div_amount_prediction
                    hist_div_date = div_last_year_df.iloc[-1]['派息日']
                    if hist_div_date + relativedelta(years=1) <= datetime.datetime.now():
                        temp_dict['派息日'] = '对应日期已过，待定'
                    else:
                        if 1 <= (hist_div_date + relativedelta(years=1)).weekday() <= 4:
                            temp_dict['派息日'] = (hist_div_date + relativedelta(years=1)).strftime('%Y%m%d')
                            temp_dict['股权登记日'] = (hist_div_date + relativedelta(years=1)
                                                  - datetime.timedelta(days=1)).strftime('%Y%m%d')

                        elif (hist_div_date + relativedelta(years=1)).weekday() == 0:
                            temp_dict['派息日'] = (hist_div_date + relativedelta(years=1)).strftime('%Y%m%d')
                            temp_dict['股权登记日'] = (hist_div_date + relativedelta(years=1)
                                                  - datetime.timedelta(days=3)).strftime('%Y%m%d')

                        elif (hist_div_date + relativedelta(years=1)).weekday() == 5:
                            temp_dict['派息日'] = (hist_div_date + datetime.timedelta(days=2)
                                                            + relativedelta(years=1)).strftime('%Y%m%d')
                            temp_dict['股权登记日'] = (hist_div_date + relativedelta(years=1)
                                                  - datetime.timedelta(days=1)).strftime('%Y%m%d')
                        else:
                            temp_dict['派息日'] = (hist_div_date + datetime.timedelta(days=1)
                                                            + relativedelta(years=1)).strftime('%Y%m%d')
                            temp_dict['股权登记日'] = (hist_div_date + relativedelta(years=1)
                                                  - datetime.timedelta(days=2)).strftime('%Y%m%d')
                    temp_dict['预案公告日'] = '待定'
                    temp_dict['分红点数'] = self.cal_div_point(div_amount_prediction, prediction_type)

                    temp_df = pd.DataFrame(temp_dict,index=[0])
                    result_df = result_df.append(temp_df,ignore_index=True,sort=True)
                return result_df


    def div_prediction(self,dt_input,prediction_type,years_back=2):
        dt_input = dt_input + datetime.timedelta(days=2)
        div_df_list = []
        prediction_df_list = []

        c_list = self.constituent_list_dict[prediction_type]
        self.get_eps_df(c_list)
        #self.eps_df.to_csv('eps.csv',encoding='gbk')

        while years_back > 1:
            dt_start = dt_input - relativedelta(years=years_back) - datetime.timedelta(days=years_back+1)
            dt_end = dt_start + relativedelta(years=1)
            div_df_list.append(self.get_div_df(prediction_type,dt_start,dt_end))
            years_back = years_back-1

        div_df_list.append(self.get_div_df(prediction_type,dt_end+datetime.timedelta(1)))

        self.history_div_df = pd.concat(div_df_list,ignore_index=True)
        self.history_div_df = self.history_div_df.loc[self.history_div_df.loc[:,'税前派息']!=0,:].copy()
        self.history_div_df.loc[:,'预案公告日'] = pd.to_datetime(self.history_div_df.loc[:,'预案公告日'],format="%Y%m%d")

        #for code in ['600019.SH']:
        for code in self.constituent_list_dict[prediction_type]:
            #print('code:',code)
            self.stock_div_df = self.history_div_df.loc[self.history_div_df.loc[:,'股票代码']==code,:]
            self.stock_code_processing = code
            self.stock_name_processing = self.get_stock_name(self.stock_code_processing)
            prediction_df_list.append(self.process_prediction(prediction_type))

        return prediction_df_list

    def run_daily(self):
        filename = '配股统计及分红预测_更新至' + str(datetime.datetime.now().date()) + '.xls'
        writer = pd.ExcelWriter(filename)

        while not self.get_constituent_df(self.dt):
            self.dt = self.dt - datetime.timedelta(days=1)
        self.get_index_close(self.dt)
        to_do_list = ["IH","IF","IC"]

        for index_type in to_do_list:
            sheet_name = index_type+'分红预测'
            df = pd.concat(div.div_prediction(self.dt,index_type), ignore_index=True, sort=True)
            #print(df)
            df['派息日'] = df['派息日'].astype('str')
            df.sort_values('派息日',ascending=True,inplace=True)
            df.loc[:,['股票代码','股票名称','状态','税前派息','分红点数',
                      '每股转送','预案公告日','股权登记日','派息日']].to_excel(writer,sheet_name=sheet_name)

        right_issue_dt = self.dt - relativedelta(months=2)
        self.get_right_issue_df(right_issue_dt)
        print('dividend is done!')

        for index_type in to_do_list:
            sheet_name = index_type+'配股'
            self.right_issue_df.loc[self.right_issue_df['股票代码'].isin(self.constituent_list_dict[index_type]),:]\
                .reset_index(drop=True).to_excel(writer,sheet_name=sheet_name)
        print('right issue is done!')
        writer.save()



if __name__ == "__main__":
    pd.set_option('display.max_columns',None)

    div = div_pred_statistic()
    div.run_daily()

