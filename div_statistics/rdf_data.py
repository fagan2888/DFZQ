import cx_Oracle as oracle
import pandas as pd
import datetime

class rdf_data:
    def __init__(self):
        self.ip_address = '172.17.21.3'
        self.port = '1521'
        self.username = 'yspread'
        self.pwd = 'Y*iaciej123456'
        self.conn = oracle.connect(
            self.username + "/" + self.pwd + "@" + self.ip_address + ":" + self.port + "/" + "WDZX")
        self.curs = self.conn.cursor()

    def get_index_constituent_df(self,index_type,date_input):
        if isinstance(date_input,str):
            str_date = date_input
        elif isinstance(date_input,int):
            str_date = str(date_input)
        elif isinstance(date_input,datetime.datetime):
            str_date = date_input.strftime("%Y%m%d")

        IH_sentence = "select TRADE_DT,S_INFO_WINDCODE,S_CON_WINDCODE,closevalue,weight " \
                      "from wind_filesync.AIndexSSE50Weight " \
                      "where trade_dt =: dt"
        IF_sentence = "select TRADE_DT,S_INFO_WINDCODE,S_CON_WINDCODE,i_weight_15,i_weight " \
                      "from wind_filesync.AIndexHS300Weight " \
                      "where trade_dt =: dt"
        IC_sentence = "select TRADE_DT,S_INFO_WINDCODE,S_CON_WINDCODE,closevalue,weight " \
                      "from wind_filesync.AIndexCSI500Weight " \
                      "where trade_dt =: dt"
        sentence_dict = {'IH':IH_sentence,'IF':IF_sentence,'IC':IC_sentence}

        self.curs.execute(sentence_dict[index_type], dt=str_date)
        index_constituent_df = pd.DataFrame(self.curs.fetchall(),
                             columns=['date', 'index_code', 'constituent_code', 'close', 'weight'])
        #print('from RDF: %s constituent got!' %index_type)
        return index_constituent_df

    def get_trading_calendar(self):
        get_calendar_sentence = "select TRADE_DAYS from wind_filesync.AShareCalendar " \
                                "where trade_days >= 20160101"
        self.curs.execute(get_calendar_sentence)
        fetch_data = self.curs.fetchall()
        trading_days_list = []
        for data in fetch_data:
            trading_days_list.append(data[0])
        trading_calendar = pd.to_datetime(pd.Series(trading_days_list)).drop_duplicates() \
            .sort_values().reset_index(drop=True)
        return trading_calendar

    def get_stock_name(self,*code_input):
        code_list = []
        code_name_dict = {}
        for code in code_input:
            code_list.append(code)
        if len(code_list) > 1:
            get_name_sentence = 'select s_info_windcode,s_info_name ' \
                                'from wind_filesync.AShareDescription ' \
                                'where s_info_windcode in ' + str(tuple(code_list))
            self.curs.execute(get_name_sentence)
        else:
            get_name_sentence = 'select s_info_windcode,s_info_name ' \
                                'from wind_filesync.AShareDescription ' \
                                'where s_info_windcode = :cd'
            self.curs.execute(get_name_sentence,cd=code_list[0])
        fetch_data = self.curs.fetchall()
        for fetch_pair in fetch_data:
            code_name_dict[fetch_pair[0]] = fetch_pair[1]
        return code_name_dict

    def get_constituent_div_df(self, constituent_list, date_start_input, date_end_input=None):
        if isinstance(date_start_input,str):
            str_date_start = date_start_input
        elif isinstance(date_start_input,int):
            str_date_start = str(date_start_input)
        elif isinstance(date_start_input,datetime.datetime):
            str_date_start = date_start_input.strftime("%Y%m%d")

        if not date_end_input:
            sql_sentence = 'select s_info_windcode,STK_DVD_PER_SH,CASH_DVD_PER_SH_PRE_TAX,' \
                           'S_DIV_PROGRESS,EQY_RECORD_DT,EX_DT,DVD_PAYOUT_DT,DVD_ANN_DT,S_DIV_PRELANDATE,report_period ' \
                           'from wind_filesync.AShareDividend ' \
                           'where s_div_progress <= 3 and S_DIV_PRELANDATE >= :start_date ' \
                           'and s_info_windcode in ' + str(tuple(constituent_list)) \
                           + 'order by eqy_record_dt'
            self.curs.execute(sql_sentence, start_date=str_date_start)
        else:
            if isinstance(date_end_input, str):
                str_date_end = date_end_input
            elif isinstance(date_end_input, int):
                str_date_end = str(date_end_input)
            elif isinstance(date_end_input, datetime.datetime):
                str_date_end = date_end_input.strftime("%Y%m%d")
            sql_sentence = 'select s_info_windcode, STK_DVD_PER_SH, CASH_DVD_PER_SH_PRE_TAX,' \
                           'S_DIV_PROGRESS,EQY_RECORD_DT,EX_DT,DVD_PAYOUT_DT,DVD_ANN_DT,S_DIV_PRELANDATE,report_period ' \
                           'from wind_filesync.AShareDividend ' \
                           'where s_div_progress <= 3 and S_DIV_PRELANDATE >= :start_date and S_DIV_PRELANDATE <= :end_date ' \
                           'and s_info_windcode in ' + str(tuple(constituent_list)) \
                           + 'order by eqy_record_dt'
            self.curs.execute(sql_sentence, start_date=str_date_start, end_date=str_date_end)
        fetch_data = self.curs.fetchall()
        constituent_div_df = pd.DataFrame(fetch_data, columns=['股票代码', '每股转送', '税前派息', '方案进度',
            '股权登记日','除权除息日', '派息日', '分红实施公告日', '预案公告日', '分红年度'])
        stock_name_dict = self.get_stock_name(*constituent_list)
        constituent_div_df['股票名称'] = None
        for idx,row in constituent_div_df.iterrows():
            constituent_div_df.loc[idx,'股票名称'] = stock_name_dict[row['股票代码']]
        constituent_div_df = constituent_div_df.loc[:, ['股票代码', '股票名称', '每股转送', '税前派息', '方案进度',
            '股权登记日', '除权除息日', '派息日', '分红实施公告日', '预案公告日', '分红年度']]
        return constituent_div_df

    def get_stock_historical_div_data(self,code_list):
        sql_sentense = 'select s_info_windcode,STK_DVD_PER_SH,CASH_DVD_PER_SH_PRE_TAX,S_DIV_PROGRESS,EQY_RECORD_DT,' \
                       'EX_DT,DVD_PAYOUT_DT,DVD_ANN_DT,S_DIV_PRELANDATE,s_div_smtgdate,report_period ' \
                       'from wind_filesync.AShareDividend ' \
                       'where s_div_progress <=3 and s_info_windcode in ' + str(tuple(code_list)) + \
                       'order by S_info_windcode'
        self.curs.execute(sql_sentense)
        fetch_data = self.curs.fetchall()
        all_historical_data = pd.DataFrame(fetch_data, columns=['股票代码', '每股转送', '税前派息', '方案进度',
            '股权登记日', '除权除息日', '派息日', '分红实施公告日', '预案公告日', '股东大会公告日', '分红年度'])
        stock_name_dict = self.get_stock_name(*code_list)
        all_historical_data['股票名称'] = None
        for idx,row in all_historical_data.iterrows():
            all_historical_data.loc[idx,'股票名称'] = stock_name_dict[row['股票代码']]
        return all_historical_data

    def get_right_issue_df(self, date_input):
        if isinstance(date_input,str):
            str_date = date_input
        elif isinstance(date_input,int):
            str_date = str(date_input)
        elif isinstance(date_input,datetime.datetime):
            str_date = date_input.strftime("%Y%m%d")

        sql_sentence = 'select s_info_windcode,S_RIGHTSISSUE_PROGRESS,S_RIGHTSISSUE_PRICE,S_RIGHTSISSUE_RATIO,' \
                       'S_RIGHTSISSUE_AMOUNT,S_RIGHTSISSUE_AMOUNTACT,S_RIGHTSISSUE_NETCOLLECTION,' \
                       'S_RIGHTSISSUE_REGDATESHAREB,S_RIGHTSISSUE_EXDIVIDENDDATE,S_RIGHTSISSUE_LISTEDDATE,' \
                       'S_RIGHTSISSUE_PAYSTARTDATE,S_RIGHTSISSUE_PAYENDDATE,S_RIGHTSISSUE_ANNCEDATE,' \
                       'S_RIGHTSISSUE_RESULTDATE,S_RIGHTSISSUE_YEAR ' \
                       'from wind_filesync.AShareRightIssue ' \
                       'where S_RIGHTSISSUE_PROGRESS <= 3 and S_RIGHTSISSUE_REGDATESHAREB >= :dt ' \
                       'order by S_RIGHTSISSUE_REGDATESHAREB'
        self.curs.execute(sql_sentence, dt=str_date)
        fetch_data = self.curs.fetchall()
        right_issue_df = pd.DataFrame(fetch_data, columns=['股票代码', '方案进度', '配股价格', '配股比例',
            '配股计划数量（万股）', '配股实际数量（万股）', '募集资金（元）', '股权登记日', '除权日','配股上市日',
            '缴款起始日', '缴款终止日', '配股实施公告日', '配股结果公告日','配股年度'])
        code_list = right_issue_df['股票代码'].tolist()
        if right_issue_df.empty:
            return right_issue_df
        stock_name_dict = self.get_stock_name(*code_list)
        right_issue_df['股票名称'] = None
        for idx,row in right_issue_df.iterrows():
            right_issue_df.loc[idx,'股票名称'] = stock_name_dict[row['股票代码']]
        right_issue_df = right_issue_df.loc[:, ['股票代码', '股票名称', '方案进度',
            '配股价格', '配股比例', '配股计划数量（万股）','配股实际数量（万股）', '募集资金（元）','股权登记日',
            '除权日', '配股上市日', '缴款起始日','缴款终止日', '配股实施公告日','配股结果公告日', '配股年度']]
        return right_issue_df

    def get_stock_close(self,code_input,date_input):
        if isinstance(date_input,str):
            str_date = date_input
        elif isinstance(date_input,int):
            str_date = str(date_input)
        elif isinstance(date_input,datetime.datetime):
            str_date = date_input.strftime("%Y%m%d")

        sql_sentence = "select s_info_windcode,trade_dt,s_dq_close " \
                       "from wind_filesync.AShareEODPrices " \
                       "where s_info_windcode = :code and trade_dt = :dt"
        self.curs.execute(sql_sentence,code=code_input,dt=str_date)
        fetch_data = self.curs.fetchall()
        stock_close_df = pd.DataFrame(fetch_data, columns=['股票代码', '日期', '收盘价'])
        stock_close_df['日期'] = pd.to_datetime(stock_close_df['日期'], format="%Y%m%d")
        stock_close_df.set_index('股票代码', inplace=True)
        return stock_close_df

    def get_index_close(self,index_type,start_date_input,end_date_input=None):
        if isinstance(start_date_input,str):
            str_start_date = start_date_input
        elif isinstance(start_date_input,int):
            str_start_date = str(start_date_input)
        elif isinstance(start_date_input,datetime.datetime):
            str_start_date = start_date_input.strftime("%Y%m%d")

        index_code_dict = {'IH':'000016.SH','IF':'000300.SH','IC':'000905.SH'}
        if not end_date_input:
            sql_sentence = "select s_info_windcode,trade_dt,s_dq_close " \
                           "from wind_filesync.AIndexEODPrices " \
                           "where s_info_windcode = :code and trade_dt = :dt"
            self.curs.execute(sql_sentence, code=index_code_dict[index_type], dt=str_start_date)
        else:
            if isinstance(end_date_input, str):
                str_end_date = end_date_input
            elif isinstance(end_date_input, int):
                str_end_date = str(end_date_input)
            elif isinstance(end_date_input, datetime.datetime):
                str_end_date = end_date_input.strftime("%Y%m%d")
            sql_sentence = "select s_info_windcode,trade_dt,s_dq_close " \
                           "from wind_filesync.AIndexEODPrices " \
                           "where s_info_windcode = :code and trade_dt >= :s_dt and trade_dt <= :e_dt"
            self.curs.execute(sql_sentence,code=index_code_dict[index_type],s_dt=str_start_date,e_dt=str_end_date)
        fetch_data = self.curs.fetchall()
        stock_close_df = pd.DataFrame(fetch_data, columns=['股票代码', '日期', '收盘价'])
        stock_close_df['日期'] = pd.to_datetime(stock_close_df['日期'], format="%Y%m%d")
        stock_close_df.set_index('股票代码', inplace=True)
        return stock_close_df

    def get_eps_df(self, code_list):
        print('fetching eps data...')
        sql_sentence = "select s_info_windcode,trade_dt,NET_PROFIT_PARENT_COMP_TTM,TOT_SHR_TODAY " \
                       "from wind_filesync.AShareEODDerivativeIndicator " \
                       "where trade_dt >20160101 and s_info_windcode in " + str(tuple(code_list))
        self.curs.execute(sql_sentence)
        fetch_data = self.curs.fetchall()
        eps_df = pd.DataFrame(fetch_data, columns=['code', 'date', 'net_profit', 'total_shares'])
        eps_df['date'] = pd.to_datetime(eps_df['date'], format="%Y%m%d")
        eps_df.sort_values(by='date', ascending=True, inplace=True)
        eps_df.set_index('date', inplace=True)
        eps_df['eps'] = eps_df['net_profit'] / eps_df['total_shares']
        print('eps data got!')
        return eps_df


if __name__ == "__main__":
    a = rdf_data()
    d = a.get_index_constituent_df('IF',20110801)
    print(d)