from data_process import DataProcess
import pandas as pd
import numpy as np
from rdf_data import rdf_data
from influxdb_data import influxdbData
from data_process import DataProcess
import joblib

class StrategyBase:
    def __init__(self):
        self.rdf = rdf_data()
        self.influx = influxdbData()

    def get_basic_info(self,start,end,filter_ST=True,industry='citics_lv1_name',mkt_cap_field='ln_market_cap'):
        mkt_data = self.influx.getDataMultiprocess('DailyData_Gus','marketData',start,end,None)
        mkt_data = mkt_data.loc[(mkt_data['status'] != '停牌') & (pd.notnull(mkt_data['status'])), :]
        if filter_ST:
            mkt_data = mkt_data.loc[mkt_data['isST']==False,:]
        mkt_data['return'] = mkt_data['close']/mkt_data['preclose'] -1
        # 超过0.11或-0.11的return标记为异常数据，置为nan(新股本身剔除)
        mkt_data.loc[(mkt_data['return']>0.11) | (mkt_data['return']<-0.11),'return'] = np.nan
        mkt_data = mkt_data.dropna(subset=['return'])

        calendar = self.rdf.get_trading_calendar()
        trade_day = pd.DataFrame(mkt_data.index.unique().tolist(),columns=['date'])
        trade_day['former_trade_day'] = trade_day['date'].apply(lambda x: calendar[calendar<x].iloc[-1])
        trade_day['next_trade_day'] = trade_day['date'].apply(lambda x: calendar[calendar>x].iloc[0])
        trade_day.set_index('date',inplace=True)

        mkt_data = pd.merge(mkt_data,trade_day,left_index=True,right_index=True,how='left')
        mkt_data.set_index([mkt_data.index,'code'],inplace=True)
        mkt_data.index.names = ['date','code']
        rtn_data = mkt_data.loc[:,['former_trade_day','next_trade_day','return']]
        industry_data = pd.get_dummies(mkt_data[industry])
        # 过滤掉没有行业信息的数据
        industry_data = industry_data.loc[~(industry_data==0).all(axis=1),:]
        rtn_data.reset_index(inplace=True)
        industry_data.reset_index(inplace=True)
        mkt_data = pd.merge(rtn_data,industry_data,how='right',on=['date','code'])

        size_data = self.influx.getDataMultiprocess('DailyFactor_Gus', 'Size',start,end,None)
        size_data.index.names = ['date']
        size_data.reset_index(inplace=True)
        size_data = size_data.loc[:,['date','code',mkt_cap_field]]
        size_data[mkt_cap_field] = DataProcess.Z_standardize(size_data[mkt_cap_field])

        mkt_data = pd.merge(mkt_data,size_data,how='inner',on=['date','code'])
        print('basic info loaded!')
        return mkt_data


    @staticmethod
    def cross_section_remove_outlier(factor_data,factor_field,date):
        day_factor = factor_data.loc[date,:].copy()
        day_factor[factor_field] = DataProcess.remove_outlier(day_factor[factor_field])
        day_factor = day_factor.dropna(subset=[factor_field])
        return day_factor


    @staticmethod
    def cross_section_Z_standardize(factor_data,factor_field,date):
        day_factor = factor_data.loc[date, :].copy()
        day_factor[factor_field] = DataProcess.Z_standardize(day_factor[factor_field])
        day_factor = day_factor.dropna(subset=[factor_field])
        return day_factor


    @staticmethod
    def cross_section_rank_standardize(factor_data,factor_field,date):
        day_factor = factor_data.loc[date, :].copy()
        day_factor[factor_field] = DataProcess.rank_standardize(day_factor[factor_field])
        day_factor = day_factor.dropna(subset=[factor_field])
        return day_factor


    # factor.index 是date
    # mkt_data 的date在columns里
    def test_factor(self,factor_data,factor_field,stk_data,standardize='z',remove_outlier=True):
        dates = factor_data.index.unique()
        # 数据预处理
        if remove_outlier:
            df_list = joblib.Parallel(n_jobs=6)(joblib.delayed(StrategyBase.cross_section_remove_outlier)
                                                (factor_data,factor_field,date)
                                                for date in dates)
            factor_data = pd.concat(df_list,ignore_index=False)
            factor_data = factor_data.sort_index()
            print('outlier remove finish!')

        if standardize == 'z':
            df_list = joblib.Parallel(n_jobs=6)(joblib.delayed(StrategyBase.cross_section_Z_standardize)
                                                (factor_data, factor_field, date)
                                                for date in dates[0:50])
            factor_data = pd.concat(df_list, ignore_index=False)
            factor_data = factor_data.sort_index()
            print('Z_standardize finish!')
        elif standardize == 'rank':
            df_list = joblib.Parallel(n_jobs=6)(joblib.delayed(StrategyBase.cross_section_rank_standardize)
                                                (factor_data, factor_field, date)
                                                for date in dates)
            factor_data = pd.concat(df_list, ignore_index=False)
            factor_data = factor_data.sort_index()
            print('rank_standardize finish!')
        else:
            pass

        factor_data.index.names = ['date']
        factor_data.reset_index(inplace=True)
        stk_data = pd.merge(stk_data,factor_data,on=['date','code'])
        print('.')


if __name__ == '__main__':
    sb = StrategyBase()
    start = 20150101
    end = 20160101
    mkt_data = sb.get_basic_info(start,end)
    ep_cut = sb.influx.getDataMultiprocess('DailyFactor_Gus','Value',start,end,None)
    ep_cut = ep_cut.loc[:,['code','EPcut_TTM']]
    ep_cut = ep_cut.dropna(subset=['EPcut_TTM'])
    sb.test_factor(ep_cut,'EPcut_TTM',mkt_data,standardize='z',remove_outlier=False)