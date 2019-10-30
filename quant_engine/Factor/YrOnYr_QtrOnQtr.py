import cx_Oracle as oracle
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from rdf_data import rdf_data
import datetime
import dateutil.parser as dtparser

class YrOnYr_QtrOnQtr:
    def __init__(self):
        self.rdf = rdf_data()

    def getYOYorQOQ(self,currentValue,lastValue):
        if not lastValue or not currentValue or lastValue==0:
            return np.nan
        else:
            return (currentValue-lastValue)/abs(lastValue)

    def cal_factor(self,code,startDate=None,endDate=None):
        #S_FA_YOYROE: wind自带的yoy数据；S_QFA_ROE：单季roe；S_QFA_ROE_DEDUCTED：单季扣非roe；S_QFA_EPS：单季eps_basic
        database='wind_filesync.AShareFinancialIndicator'
        oracleStr="select ANN_DT ,REPORT_PERIOD," \
                  "S_FA_YOYROE, S_QFA_ROE, S_QFA_ROE_DEDUCTED, S_FA_EPS_DILUTED, S_FA_BPS, S_FA_GRPS " \
                  "FROM {0} ".format(database)
        # startDate 需要提前一年
        if startDate==None:
            oracleStr=oracleStr+"where S_INFO_WINDCODE='{0}' order by ANN_DT".format(code)
        elif endDate==None:
            startDate = (dtparser.parse(str(startDate)) - relativedelta(years=1)).strftime("%Y%m%d")
            oracleStr=oracleStr+"where S_INFO_WINDCODE='{0}' and ANN_DT>={1} order by ANN_DT".format(code,startDate)
        else:
            startDate = (dtparser.parse(str(startDate)) - relativedelta(years=1)).strftime("%Y%m%d")
            oracleStr=oracleStr+"where S_INFO_WINDCODE='{0}' and ANN_DT>={1} and ANN_DT<={2} order by ANN_DT".\
                format(code,startDate,endDate)
        self.rdf.curs.execute(oracleStr)
        mydata = pd.DataFrame(self.rdf.curs.fetchall(),columns=['Ann_date','Report_Period','ROE_YOY_wind','ROE_Q',
                                                                'ROE_deducted_Q','EPS_diluted','BPS','GRPS'])
        if mydata.empty:
            return mydata
        mydata['ROE_lastQ'] = mydata['ROE_Q'].shift()
        mydata['ROE_lastY'] = mydata['ROE_Q'].shift(4)
        mydata['ROE_deducted_lastQ'] = mydata['ROE_deducted_Q'].shift()
        mydata['ROE_deducted_lastY'] = mydata['ROE_deducted_Q'].shift(4)
        mydata['ROE_YOY'] = mydata.apply(lambda row: self.getYOYorQOQ(row['ROE_Q'],row['ROE_lastY']),axis=1)
        mydata['ROE_QOQ'] = mydata.apply(lambda row: self.getYOYorQOQ(row['ROE_Q'],row['ROE_lastQ']),axis=1)
        mydata['ROE_deducted_YOY'] = mydata.apply(lambda row: self.getYOYorQOQ(row['ROE_deducted_Q'], row['ROE_deducted_lastY']),axis=1)
        mydata['ROE_deducted_QOQ'] = mydata.apply(lambda row: self.getYOYorQOQ(row['ROE_deducted_Q'], row['ROE_deducted_lastQ']),axis=1)

        mydata['Ann_date']=pd.to_datetime(mydata['Ann_date'],format='%Y%m%d')
        mydata['Report_Period']=pd.to_datetime(mydata['Report_Period'],format='%Y%m%d')
        mydata['Report_Year'] = mydata.apply(lambda row:row['Report_Period'].strftime('%Y'),axis=1)

        to_concat_list = []
        year_list = mydata['Report_Year'].unique()
        for year in year_list:
            one_yr_data = mydata.loc[mydata['Report_Year']==year,:].copy()
            one_yr_data['EPS_diluted_Q'] = one_yr_data['EPS_diluted'] - one_yr_data['EPS_diluted'].shift()
            one_yr_data['EPS_diluted_Q'].iloc[0] = one_yr_data['EPS_diluted'].iloc[0]
            one_yr_data['GRPS_Q'] = one_yr_data['GRPS'] - one_yr_data['GRPS'].shift()
            one_yr_data['GRPS_Q'].iloc[0] = one_yr_data['GRPS'].iloc[0]
            to_concat_list.append(one_yr_data)

        mydata = pd.concat(to_concat_list)
        del year_list
        del one_yr_data
        del to_concat_list
        mydata.set_index('Ann_date',inplace=True,drop=True)
        mydata = mydata.loc[:,['ROE_YOY_wind','ROE_YOY','ROE_QOQ','ROE_deducted_YOY','ROE_deducted_QOQ','EPS_diluted_Q',
                               'GRPS_Q','BPS']]
        mydata = mydata.groupby(mydata.index).last()

        database = 'wind_filesync.AShareEODPrices'
        oracleStr = "select S_INFO_WINDCODE,TRADE_DT,S_DQ_CLOSE " \
                    "FROM {0} ".format(database)
        if startDate == None:
            oracleStr = oracleStr + "where S_INFO_WINDCODE='{0}' order by TRADE_DT".format(code)
        elif endDate == None:
            oracleStr = oracleStr + "where S_INFO_WINDCODE='{0}' and TRADE_DT>={1} order by TRADE_DT".format(code,startDate)
        else:
            oracleStr = oracleStr + "where S_INFO_WINDCODE='{0}' and TRADE_DT>={1} and TRADE_DT<={2} order by TRADE_DT".\
                format(code,startDate,endDate)
        self.rdf.curs.execute(oracleStr)

        price = self.rdf.curs.fetchall()
        price = pd.DataFrame(price, columns=['code', 'date', 'close'])
        price['date'] = pd.to_datetime(price['date'], format='%Y%m%d')
        price.set_index('date',inplace=True)
        mydata = pd.merge(mydata,price,how='outer',left_index=True,right_index=True)
        mydata = mydata.fillna(method='ffill',axis=0)
        mydata['PE'] = mydata['close']/mydata['EPS_diluted_Q']
        mydata['PB'] = mydata['close']/mydata['BPS']
        mydata['PS'] = mydata['close']/mydata['GRPS_Q']

        mydata.reset_index(inplace=True)
        col_names = mydata.columns.values
        col_names[0] = 'index'
        mydata.columns = col_names
        mydata['lastQ_date'] = mydata.apply(lambda row: row['index']-relativedelta(months=3),axis=1)
        mydata['lastY_date'] = mydata.apply(lambda row: row['index']-relativedelta(years=1),axis=1)

        to_merge_df = mydata.loc[:,['index','PE','PB','PS']].copy()
        to_merge_df.columns = ['lastQ_date','PE_lastQ','PB_lastQ','PS_lastQ']
        mydata = pd.merge(mydata,to_merge_df,how='left',left_on='lastQ_date',right_on='lastQ_date')
        to_merge_df.columns = ['lastY_date','PE_lastY','PB_lastY','PS_lastY']
        mydata = pd.merge(mydata,to_merge_df,how='left',left_on='lastY_date',right_on='lastY_date')
        mydata = mydata.fillna(method='ffill', axis=0)

        mydata['PE_YOY'] = mydata.apply(lambda row: self.getYOYorQOQ(row['PE'], row['PE_lastY']), axis=1)
        mydata['PE_QOQ'] = mydata.apply(lambda row: self.getYOYorQOQ(row['PE'], row['PE_lastQ']), axis=1)
        mydata['PB_YOY'] = mydata.apply(lambda row: self.getYOYorQOQ(row['PB'], row['PB_lastY']), axis=1)
        mydata['PB_QOQ'] = mydata.apply(lambda row: self.getYOYorQOQ(row['PB'], row['PB_lastQ']), axis=1)
        mydata['PS_YOY'] = mydata.apply(lambda row: self.getYOYorQOQ(row['PS'], row['PS_lastY']), axis=1)
        mydata['PS_QOQ'] = mydata.apply(lambda row: self.getYOYorQOQ(row['PS'], row['PS_lastQ']), axis=1)
        mydata.set_index('index', inplace=True, drop=True)
        ROE_YOY_wind = mydata['ROE_YOY_wind'].dropna()
        mydata = mydata.loc[:,['ROE_YOY','ROE_QOQ','ROE_deducted_YOY','ROE_deducted_QOQ',
                               'PE_YOY','PE_QOQ','PB_YOY','PB_QOQ','PS_YOY','PS_QOQ']]
        columns = mydata.columns
        to_concat_list = []
        for col in columns:
            if 'YOY' in col:
                tmp = mydata.loc[mydata.index[0]+relativedelta(years=1):,col].dropna()
                to_concat_list.append(tmp)
            elif 'QOQ' in col:
                tmp = mydata.loc[mydata.index[0]+relativedelta(months=3):, col].dropna()
                to_concat_list.append(tmp)
            else:
                pass
        to_concat_list.append(ROE_YOY_wind)
        mydata = pd.concat(to_concat_list,axis=1)
        mydata['code'] = code
        mydata = mydata.replace(np.inf,0)
        mydata = mydata.replace(-np.inf,0)
        mydata = mydata.where(pd.notnull(mydata),None)
        return mydata


if __name__ == '__main__':
    YOY_QOQ = YrOnYr_QtrOnQtr()
    a = YOY_QOQ.cal_factor('000625.SZ',20150701,20191014)
    a.to_csv('000625_SZ.csv')