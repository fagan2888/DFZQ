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

    def date_preprocess(self,date_input):
        if isinstance(date_input,str):
            str_date = date_input
        elif isinstance(date_input,int):
            str_date = str(date_input)
        elif isinstance(date_input,datetime.datetime):
            str_date = date_input.strftime("%Y%m%d")
        return str_date

    # 维护回测数据用
    def get_ohlc(self,code_list=None,start_input=None,end_input=None):
        query = "select S_INFO_WINDCODE,TRADE_DT,S_DQ_PRECLOSE,S_DQ_OPEN,S_DQ_HIGH,S_DQ_LOW,S_DQ_CLOSE,S_DQ_VOLUME," \
                "S_DQ_AMOUNT,S_DQ_TRADESTATUS " \
                "from wind_filesync.AShareEODPrices"

        if (not code_list) and (not start_input) and (not end_input):
            pass
        else:
            query = query + " where"
            if code_list:
                str_filter = str(tuple(code_list))
                str_filter = " and S_INFO_WINDCODE in {0}".format(str_filter)
            else:
                str_filter = ""

            if start_input and (not end_input):
                str_start = self.date_preprocess(start_input)
                query = query + " TRADE_DT>={0}".format(str_start) + str_filter
            elif (not start_input) and end_input:
                str_end = self.date_preprocess(end_input)
                query = query + " TRADE_DT<{0}".format(str_end) + str_filter
            elif start_input and end_input:
                str_start = self.date_preprocess(start_input)
                str_end = self.date_preprocess(end_input)
                query = query + " TRADE_DT>={0} and TRADE_DT<{1}".format(str_start,str_end) + str_filter
            else:
                query = query + str_filter[4:]

        self.curs.execute(query)
        ohlc = pd.DataFrame(self.curs.fetchall(),columns=['code','date','preclose','open','high','low',
                                                          'close','volume','amount','status'])
        return ohlc


    # 维护回测数据用
    # 目前只维护50，300，500
    def get_index_ohlc(self,start_input=None,end_input=None):
        query = "select S_INFO_WINDCODE,TRADE_DT,S_DQ_PRECLOSE,S_DQ_OPEN,S_DQ_HIGH,S_DQ_LOW,S_DQ_CLOSE,S_DQ_VOLUME," \
                "S_DQ_AMOUNT " \
                "from wind_filesync.AIndexEODPrices"
        if start_input and (not end_input):
            str_start = self.date_preprocess(start_input)
            query = query + " where TRADE_DT >={0}" \
                            " and S_INFO_WINDCODE in ('000016.SH','000300.SH','000905.SH')".format(str_start)
        elif (not start_input) and end_input:
            str_end = self.date_preprocess(end_input)
            query = query + " where TRADE_DT <{0}" \
                            " and S_INFO_WINDCODE in ('000016.SH','000300.SH','000905.SH')".format(str_end)
        elif start_input and end_input:
            str_start = self.date_preprocess(start_input)
            str_end = self.date_preprocess(end_input)
            query = query + " where TRADE_DT >={0} and TRADE_DT <{1}" \
                            " and S_INFO_WINDCODE in ('000016.SH','000300.SH','000905.SH')".format(str_start,str_end)
        else:
            query = query + " where S_INFO_WINDCODE in ('000016.SH','000300.SH','000905.SH')"
        self.curs.execute(query)
        ohlc_index = pd.DataFrame(self.curs.fetchall(),columns=['code','date','preclose','open','high','low',
                                                                'close','volume','amount'])
        ohlc_index['status'] = '交易'
        return ohlc_index


    # 维护回测数据用
    # 目前只维护50，300，500
    def get_futures_ohlc(self,start_input=None,end_input=None):
        query = "select S_INFO_WINDCODE,TRADE_DT,S_DQ_PRESETTLE,S_DQ_OPEN,S_DQ_HIGH,S_DQ_LOW,S_DQ_CLOSE,S_DQ_VOLUME," \
                "S_DQ_AMOUNT " \
                "from wind_filesync.CIndexFuturesEODPrices"
        if start_input and (not end_input):
            str_start = self.date_preprocess(start_input)
            query = query + " where TRADE_DT >={0}" \
                            " and S_INFO_WINDCODE in " \
                            "('IH00.CFE','IH01.CFE','IF00.CFE','IF01.CFE','IC00.CFE','IC01.CFE')".format(str_start)
        elif (not start_input) and end_input:
            str_end = self.date_preprocess(end_input)
            query = query + " where TRADE_DT <{0}" \
                            " and S_INFO_WINDCODE in " \
                            "('IH00.CFE','IH01.CFE','IF00.CFE','IF01.CFE','IC00.CFE','IC01.CFE')".format(str_end)
        elif start_input and end_input:
            str_start = self.date_preprocess(start_input)
            str_end = self.date_preprocess(end_input)
            query = query + " where TRADE_DT >={0} and TRADE_DT <{1}" \
                            " and S_INFO_WINDCODE in " \
                            "('IH00.CFE','IH01.CFE','IF00.CFE','IF01.CFE','IC00.CFE','IC01.CFE')".format(str_start,str_end)
        else:
            query = query + " where S_INFO_WINDCODE in " \
                            "('IH00.CFE','IH01.CFE','IF00.CFE','IF01.CFE','IC00.CFE','IC01.CFE')"
        self.curs.execute(query)
        ohlc_futures = pd.DataFrame(self.curs.fetchall(),columns=['code','date','preclose','open','high','low',
                                                                'close','volume','amount'])
        ohlc_futures['status'] = '交易'
        return ohlc_futures


    # 维护回测数据用
    # 获取复权信息
    def get_EX_right_dvd(self,start_input=None,end_input=None):
        query = "select S_INFO_WINDCODE, EX_DATE, CASH_DIVIDEND_RATIO, BONUS_SHARE_RATIO, RIGHTSISSUE_RATIO, " \
                "RIGHTSISSUE_PRICE, CONVERSED_RATIO, CONSOLIDATE_SPLIT_RATIO, SEO_PRICE, SEO_RATIO " \
                "from wind_filesync.AShareEXRightDividendRecord "
        if start_input and (not end_input):
            str_start = self.date_preprocess(start_input)
            query = query + " where EX_DATE >={0}".format(str_start)
        elif (not start_input) and end_input:
            str_end = self.date_preprocess(end_input)
            query = query + " where EX_DATE <{0}".format(str_end)
        elif start_input and end_input:
            str_start = self.date_preprocess(start_input)
            str_end = self.date_preprocess(end_input)
            query = query + " where EX_DATE >={0} and EX_DATE <{1}".format(str_start,str_end)
        else:
            pass
        self.curs.execute(query)
        ex_right = pd.DataFrame(self.curs.fetchall(),columns=['code','date','cash_dvd_ratio','bonus_share_ratio',
            'rightissue_ratio','rightissue_price','conversed_ratio','split_ratio','seo_price','seo_ratio'])
        return ex_right


    # 维护回测数据用
    def get_dvd_data(self,code_list=None,start_input=None,end_input=None):
        query = "select S_INFO_WINDCODE, CASH_DVD_PER_SH_PRE_TAX, STK_DVD_PER_SH, DVD_PAYOUT_DT, LISTING_DT_OF_DVD_SHR " \
                "from wind_filesync.AShareDividend " \
                "where S_DIV_PROGRESS =3 "
        if (not code_list) and (not start_input) and (not end_input):
            pass
        else:
            query = query + " and"
            if code_list:
                str_filter = str(tuple(code_list))
                str_filter = " and S_INFO_WINDCODE in {0}".format(str_filter)
            else:
                str_filter = ""

            if start_input and (not end_input):
                str_start = self.date_preprocess(start_input)
                query = query + " (DVD_PAYOUT_DT>={0} or LISTING_DT_OF_DVD_SHR>={0})".format(str_start) + str_filter
            elif (not start_input) and end_input:
                str_end = self.date_preprocess(end_input)
                query = query + " (DVD_PAYOUT_DT<{0} or LISTING_DT_OF_DVD_SHR<{0})".format(str_end) + str_filter
            elif start_input and end_input:
                str_start = self.date_preprocess(start_input)
                str_end = self.date_preprocess(end_input)
                query = query + " (DVD_PAYOUT_DT>={0} or LISTING_DT_OF_DVD_SHR>={0}) and" \
                                " (DVD_PAYOUT_DT<{1} or LISTING_DT_OF_DVD_SHR<{1})".format(str_start, str_end) + str_filter
            else:
                query = query + str_filter[4:]

        self.curs.execute(query)
        dvd = pd.DataFrame(self.curs.fetchall(), columns=['code', 'dvd_pre_tax', 'dvd_shr', 'dvd_payout_dt', 'dvd_shr_listdt'])
        return dvd

    # 维护回测数据用
    def get_index_comp_in_period(self,index_type,start_input,end_input):
        IH_sentence = "select TRADE_DT,S_INFO_WINDCODE,S_CON_WINDCODE,weight " \
                      "from wind_filesync.AIndexSSE50Weight "
        IF_sentence = "select TRADE_DT,S_INFO_WINDCODE,S_CON_WINDCODE,i_weight " \
                      "from wind_filesync.AIndexHS300Weight "
        IC_sentence = "select TRADE_DT,S_INFO_WINDCODE,S_CON_WINDCODE,weight " \
                      "from wind_filesync.AIndexCSI500Weight "
        sentence_dict = {'IH':IH_sentence,'IF':IF_sentence,'IC':IC_sentence}
        query = sentence_dict[index_type]

        if start_input and (not end_input):
            str_start = self.date_preprocess(start_input)
            query = query + " where TRADE_DT >={0}".format(str_start)
        elif (not start_input) and end_input:
            str_end = self.date_preprocess(end_input)
            query = query + " where TRADE_DT <{0}".format(str_end)
        elif start_input and end_input:
            str_start = self.date_preprocess(start_input)
            str_end = self.date_preprocess(end_input)
            query = query + " where TRADE_DT >={0} and TRADE_DT <{1}".format(str_start,str_end)

        self.curs.execute(query)
        index_comp = pd.DataFrame(self.curs.fetchall(),columns=['date', 'index_code', 'stk_code', 'weight'])
        return index_comp


    def get_index_constituent_df(self,index_type,date_input):
        self.date_preprocess(date_input)

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
        print('from RDF: %s constituent got!' %index_type)
        return index_constituent_df

    def get_trading_calendar(self):
        get_calendar_sentence = "select TRADE_DAYS from wind_filesync.AShareCalendar " \
                                "where trade_days >= 20080101"
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
        str_date_start = self.date_preprocess(date_start_input)
        if not date_end_input:
            sql_sentence = 'select s_info_windcode,STK_DVD_PER_SH,CASH_DVD_PER_SH_PRE_TAX,' \
                           'S_DIV_PROGRESS,EQY_RECORD_DT,EX_DT,DVD_PAYOUT_DT,DVD_ANN_DT,S_DIV_PRELANDATE,report_period ' \
                           'from wind_filesync.AShareDividend ' \
                           'where s_div_progress <= 3 and S_DIV_PRELANDATE >= :start_date ' \
                           'and s_info_windcode in ' + str(tuple(constituent_list)) \
                           + 'order by eqy_record_dt'
            self.curs.execute(sql_sentence, start_date=str_date_start)
        else:
            str_date_end = self.date_preprocess(date_end_input)
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
        str_date = self.date_preprocess(date_input)
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
        stock_name_dict = self.get_stock_name(*code_list)
        right_issue_df['股票名称'] = None
        for idx,row in right_issue_df.iterrows():
            right_issue_df.loc[idx,'股票名称'] = stock_name_dict[row['股票代码']]
        right_issue_df = right_issue_df.loc[:, ['股票代码', '股票名称', '方案进度',
            '配股价格', '配股比例', '配股计划数量（万股）','配股实际数量（万股）', '募集资金（元）','股权登记日',
            '除权日', '配股上市日', '缴款起始日','缴款终止日', '配股实施公告日','配股结果公告日', '配股年度']]
        return right_issue_df

    def get_stock_close(self,code_input,date_input):
        str_date = self.date_preprocess(date_input)
        sql_sentence = "select s_info_windcode,trade_dt,s_dq_close " \
                       "from wind_filesync.AShareEODPrices " \
                       "where s_info_windcode = :code and trade_dt = :dt"
        self.curs.execute(sql_sentence,code=code_input,dt=str_date)
        fetch_data = self.curs.fetchall()
        stock_close_df = pd.DataFrame(fetch_data, columns=['股票代码', '日期', '收盘价'])
        stock_close_df['日期'] = pd.to_datetime(stock_close_df['日期'], format="%Y%m%d")
        stock_close_df.set_index('股票代码', inplace=True)
        return stock_close_df

    def get_index_close(self,index_type,date_input):
        str_date = self.date_preprocess(date_input)
        index_code_dict = {'IH':'000016.SH','IF':'000300.SH','IC':'000905.SH'}

        sql_sentence = "select s_info_windcode,trade_dt,s_dq_close " \
                       "from wind_filesync.AIndexEODPrices " \
                       "where s_info_windcode = :code and trade_dt = :dt"
        self.curs.execute(sql_sentence, code=index_code_dict[index_type], dt=str_date)
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

    def get_strange_trade(self,start_date,end_date):
        start_date = self.date_preprocess(start_date)
        end_date = self.date_preprocess(end_date)
        sql_sentence = "select s_info_windcode,S_STRANGE_BGDATE,S_STRANGE_ENDDATE,S_STRANGE_TRADERNAME," \
                       "S_STRANGE_TRADERAMOUNT,S_VARIANT_TYPE " \
                       "from wind_filesync.AShareStrangeTrade " \
                       "where S_STRANGE_ENDDATE >= :bg_dt and S_STRANGE_ENDDATE <= :ed_dt " \
                       "order by S_STRANGE_ENDDATE"
        self.curs.execute(sql_sentence,bg_dt=start_date,ed_dt=end_date)
        fetch_data = self.curs.fetchall()
        strange_trade_df = pd.DataFrame(fetch_data,
            columns=['code','start_date','end_date','trader_name','trader_amount','type'])
        return strange_trade_df

    def filter_st(self,date_input,*code_input):
        str_date = self.date_preprocess(date_input)
        code_list = []
        for code in code_input:
            code_list.append(code)
        sql_sentence = "select s_info_windcode,ENTRY_DT,REMOVE_DT " \
                       "from wind_filesync.AShareST " \
                       "where s_info_windcode in " + str(tuple(code_list))
        self.curs.execute(sql_sentence)
        fetchdata = self.curs.fetchall()
        st_df = pd.DataFrame(fetchdata,columns=['code','entry_date','remove_date'])
        if st_df.empty:
            return code_list
        else:
            sts = st_df.loc[(st_df['remove_date'].isna())|
                            ((st_df['entry_date']<=str_date)&(st_df['remove_date']>=str_date)),'code'].unique()
            filtered_list = list(set(code_list).difference(set(sts)))
            return filtered_list


    def get_st(self):
        sql_sentence = "select s_info_windcode,ENTRY_DT,REMOVE_DT " \
                       "from wind_filesync.AShareST"
        self.curs.execute(sql_sentence)
        fetchdata = self.curs.fetchall()
        st = pd.DataFrame(fetchdata,columns=['code','entry_date','exit_date'])
        return st

    def get_citics_lv1(self):
        sql_sentence = "select S_INFO_WINDCODE,s_con_windcode,s_con_indate,s_con_outdate " \
                       "from wind_filesync.AIndexMembersCITICS"
        self.curs.execute(sql_sentence)
        fetchdata = self.curs.fetchall()
        citics_lv1 = pd.DataFrame(fetchdata,columns=['index_lv1_code','stock_code','entry_date','exit_date'])
        industries_code_list = citics_lv1['index_lv1_code'].unique()
        sql_sentence = "select s_info_windcode, S_INFO_NAME " \
                       "from wind_filesync.AIndexDescription " \
                       "where s_info_windcode in " +str(tuple(industries_code_list))
        self.curs.execute(sql_sentence)
        fetchdata = self.curs.fetchall()
        industries = pd.DataFrame(fetchdata,columns=['index_lv1_code','industry_lv1_name'])
        industries.set_index('index_lv1_code',inplace=True)
        citics_lv1 = citics_lv1.join(industries,on='index_lv1_code')
        return citics_lv1

    def get_citics_lv2(self):
        sql_sentence = "select S_INFO_WINDCODE,s_con_windcode,s_con_indate,s_con_outdate " \
                       "from wind_filesync.AIndexMembersCITICS2"
        self.curs.execute(sql_sentence)
        fetchdata = self.curs.fetchall()
        citics_lv2 = pd.DataFrame(fetchdata, columns=['index_lv2_code', 'stock_code', 'entry_date', 'exit_date'])
        industries_code_list = citics_lv2['index_lv2_code'].unique()
        sql_sentence = "select s_info_windcode, S_INFO_NAME " \
                       "from wind_filesync.AIndexDescription " \
                       "where s_info_windcode in " +str(tuple(industries_code_list))
        self.curs.execute(sql_sentence)
        fetchdata = self.curs.fetchall()
        industries = pd.DataFrame(fetchdata, columns=['index_lv2_code', 'industry_lv2_name'])
        industries.set_index('index_lv2_code', inplace=True)
        citics_lv2 = citics_lv2.join(industries, on='index_lv2_code')
        return citics_lv2

    def get_citics_lv3(self):
        sql_sentence = "select S_INFO_WINDCODE,s_con_windcode,s_con_indate,s_con_outdate " \
                       "from wind_filesync.AIndexMembersCITICS3"
        self.curs.execute(sql_sentence)
        fetchdata = self.curs.fetchall()
        citics_lv3 = pd.DataFrame(fetchdata, columns=['index_lv3_code', 'stock_code', 'entry_date', 'exit_date'])
        industries_code_list = citics_lv3['index_lv3_code'].unique()
        sql_sentence = "select s_info_windcode, S_INFO_NAME " \
                       "from wind_filesync.AIndexDescription " \
                       "where s_info_windcode in " + str(tuple(industries_code_list))
        self.curs.execute(sql_sentence)
        fetchdata = self.curs.fetchall()
        industries = pd.DataFrame(fetchdata, columns=['index_lv3_code', 'industry_lv3_name'])
        industries.set_index('index_lv3_code', inplace=True)
        citics_lv3 = citics_lv3.join(industries, on='index_lv3_code')
        return citics_lv3

    def get_SW_lv1(self):
        sw_lv1_index = ['801010.SI','801020.SI','801030.SI','801040.SI','801050.SI','801080.SI','801110.SI',
                        '801120.SI','801130.SI','801140.SI','801150.SI','801160.SI','801170.SI','801180.SI',
                        '801200.SI','801210.SI','801230.SI','801710.SI','801720.SI','801730.SI','801740.SI',
                        '801750.SI','801760.SI','801770.SI','801780.SI','801790.SI','801880.SI','801890.SI']
        sql_sentence = "select S_INFO_WINDCODE, S_CON_WINDCODE, S_CON_INDATE, S_CON_OUTDATE " \
                       "from wind_filesync.SWIndexMembers " \
                       "where S_INFO_WINDCODE in " + str(tuple(sw_lv1_index))
        self.curs.execute(sql_sentence)
        fetchdata = self.curs.fetchall()
        sw_lv1 = pd.DataFrame(fetchdata, columns=['index_lv1_code','stock_code','entry_date','exit_date'])
        industries_code_list = sw_lv1['index_lv1_code'].unique()
        sql_sentence = "select S_INFO_WINDCODE, S_INFO_COMPNAME " \
                       "from wind_filesync.AIndexDescription " \
                       "where S_INFO_WINDCODE in " + str(tuple(industries_code_list))
        self.curs.execute(sql_sentence)
        fetchdata = self.curs.fetchall()
        industries = pd.DataFrame(fetchdata, columns=['index_lv1_code', 'industry_lv1_name'])
        industries.set_index('index_lv1_code', inplace=True)
        sw_lv1 = sw_lv1.join(industries, on='index_lv1_code')
        return sw_lv1

    def get_SW_lv2(self):
        sw_lv2_index = ['801011.SI','801012.SI','801013.SI','801014.SI','801015.SI','801016.SI','801017.SI',
                        '801018.SI','801021.SI','801022.SI','801023.SI','801024.SI','801032.SI','801033.SI',
                        '801034.SI','801035.SI','801036.SI','801037.SI','801041.SI','801051.SI','801053.SI',
                        '801054.SI','801055.SI','801072.SI','801073.SI','801074.SI','801075.SI','801076.SI',
                        '801081.SI','801082.SI','801083.SI','801084.SI','801085.SI','801092.SI','801093.SI',
                        '801094.SI','801101.SI','801102.SI','801111.SI','801112.SI','801123.SI','801124.SI',
                        '801131.SI','801132.SI','801141.SI','801142.SI','801143.SI','801144.SI','801151.SI',
                        '801152.SI','801153.SI','801154.SI','801155.SI','801156.SI','801161.SI','801162.SI',
                        '801163.SI','801164.SI','801171.SI','801172.SI','801173.SI','801174.SI','801175.SI',
                        '801176.SI','801177.SI','801178.SI','801181.SI','801182.SI','801191.SI','801192.SI',
                        '801193.SI','801194.SI','801202.SI','801203.SI','801204.SI','801205.SI','801211.SI',
                        '801212.SI','801213.SI','801214.SI','801215.SI','801222.SI','801223.SI','801231.SI',
                        '801711.SI','801712.SI','801713.SI','801721.SI','801722.SI','801723.SI','801724.SI',
                        '801725.SI','801731.SI','801732.SI','801733.SI','801734.SI','801741.SI','801742.SI',
                        '801743.SI','801744.SI','801751.SI','801752.SI','801761.SI','801881.SI']
        sql_sentence = "select S_INFO_WINDCODE, S_CON_WINDCODE, S_CON_INDATE, S_CON_OUTDATE " \
                       "from wind_filesync.SWIndexMembers " \
                       "where S_INFO_WINDCODE in " + str(tuple(sw_lv2_index))
        self.curs.execute(sql_sentence)
        fetchdata = self.curs.fetchall()
        sw_lv2 = pd.DataFrame(fetchdata, columns=['index_lv2_code', 'stock_code', 'entry_date', 'exit_date'])
        industries_code_list = sw_lv2['index_lv2_code'].unique()
        sql_sentence = "select S_INFO_WINDCODE, S_INFO_COMPNAME " \
                       "from wind_filesync.AIndexDescription " \
                       "where S_INFO_WINDCODE in " + str(tuple(industries_code_list))
        self.curs.execute(sql_sentence)
        fetchdata = self.curs.fetchall()
        industries = pd.DataFrame(fetchdata, columns=['index_lv2_code', 'industry_lv2_name'])
        industries.set_index('index_lv2_code', inplace=True)
        sw_lv2 = sw_lv2.join(industries, on='index_lv2_code')
        return sw_lv2