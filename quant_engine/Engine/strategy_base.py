import pandas as pd
import numpy as np
from rdf_data import rdf_data
from influxdb_data import influxdbData
from data_process import DataProcess
from joblib import Parallel,delayed,parallel_backend
import warnings
import statsmodels.api as sm

class StrategyBase:
    @staticmethod
    def cross_section_remove_outlier(factor_data,factor_field,dates):
        res = []
        for date in dates:
            day_factor = factor_data.loc[date,:].copy()
            day_factor.loc[:,factor_field] = DataProcess.remove_outlier(day_factor[factor_field])
            res.append(day_factor)
        dates_factor = pd.concat(res)
        return dates_factor


    @staticmethod
    def cross_section_Z_standardize(factor_data,factor_field,dates):
        res = []
        for date in dates:
            day_factor = factor_data.loc[date, :].copy()
            day_factor.loc[:,factor_field] = DataProcess.Z_standardize(day_factor[factor_field])
            res.append(day_factor)
        dates_factor = pd.concat(res)
        return dates_factor


    @staticmethod
    def cross_section_rank_standardize(factor_data,factor_field,dates):
        res = []
        for date in dates:
            day_factor = factor_data.loc[date, :].copy()
            day_factor.loc[:,factor_field] = DataProcess.rank_standardize(day_factor[factor_field])
            res.append(day_factor)
        dates_factor = pd.concat(res)
        return dates_factor


    @staticmethod
    def regression(mkt_data,factor_field,dates):
        res = []
        idsty_size_cols = mkt_data.columns.difference(['code','date','former_trade_day','next_trade_day','return',factor_field])
        for date in dates:
            day_code = mkt_data.loc[mkt_data['date']==date,'code']
            day_factor = mkt_data.loc[mkt_data['date']==date,factor_field]
            day_return = mkt_data.loc[mkt_data['date']==date,'return']
            day_idsty_size = mkt_data.loc[mkt_data['date']==date,idsty_size_cols]
            OLS_est = sm.OLS(day_factor, day_idsty_size).fit()
            day_filtered_factor = OLS_est.resid
            day_filtered_factor.name = factor_field
            RLM_est = sm.RLM(day_return, day_filtered_factor, M=sm.robust.norms.HuberT()).fit()
            day_RLM_para = RLM_est.params
            day_Tvalue = RLM_est.tvalues
            #day_filtered_factor = pd.concat([day_code,day_filtered_factor],axis=1)
            #day_filtered_factor['date'] = date
            RLM_result = pd.DataFrame({'Fvalue':day_RLM_para.iloc[0],'Tvalue':day_Tvalue.iloc[0]},index=[date])
            res.append(RLM_result)
        res_df = pd.concat(res)
        return res_df


    def __init__(self):
        self.rdf = rdf_data()
        self.influx = influxdbData()


    def get_basic_info(self,start,end,filter_st=True,industry='citics_lv1_name',mkt_cap_field='ln_market_cap'):
        mkt_data = self.influx.getDataMultiprocess('DailyData_Gus','marketData',start,end,None)
        mkt_data = mkt_data.loc[(mkt_data['status'] != '停牌') & (pd.notnull(mkt_data['status'])), :]

        if filter_st:
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
        size_data.columns = ['date','code','size']
        # mkt cap 标准化
        size_data['size'] = DataProcess.Z_standardize(size_data['size'])
        mkt_data = pd.merge(mkt_data,size_data,how='inner',on=['date','code'])
        print('basic info loaded!')
        return mkt_data


    # factor.index 是date
    # mkt_data 的date在columns里
    def test_factor(self,factor_data,factor_field,mkt_data,standardize='z',remove_outlier=True):
        # 数据预处理
        dates = factor_data.index.unique()
        split_dates = np.array_split(dates, 30)
        if remove_outlier:
            with parallel_backend('multiprocessing', n_jobs=-1):
                parallel_res = Parallel()(delayed(StrategyBase.cross_section_remove_outlier)
                                          (factor_data, factor_field,dates) for dates in split_dates)
            factor_data = pd.concat(parallel_res)
            print('outlier remove finish!')
        if standardize == 'z':
            with parallel_backend('multiprocessing', n_jobs=-1):
                parallel_res = Parallel()(delayed(StrategyBase.cross_section_Z_standardize)
                                          (factor_data, factor_field,dates) for dates in split_dates)
            factor_data = pd.concat(parallel_res)
            print('Z_standardize finish!')
        elif standardize == 'rank':
            with parallel_backend('multiprocessing', n_jobs=-1):
                parallel_res = Parallel()(delayed(StrategyBase.cross_section_rank_standardize)
                                          (factor_data, factor_field,dates) for dates in split_dates)
            factor_data = pd.concat(parallel_res)
            print('rank_standardize finish!')
        else:
            pass

        factor_data.index.names = ['date']
        factor_data.reset_index(inplace=True)
        mkt_data = pd.merge(mkt_data,factor_data,on=['date','code'])

        # 去除行业和市值，得到新因子
        dates = mkt_data['date'].unique()
        split_dates = np.array_split(dates,30)
        with parallel_backend('multiprocessing', n_jobs=-1):
            parallel_res = Parallel()(delayed(StrategyBase.regression)
                                      (mkt_data, factor_field, dates) for dates in split_dates)
        print('regression process finish!')
        RLM_res = pd.concat(parallel_res)
        F_over_0_pct = RLM_res.loc[RLM_res['Fvalue']>0,:].shape[0] / RLM_res.shape[0]
        avg_abs_T = abs(RLM_res['Tvalue']).sum() / RLM_res.shape[0]
        abs_T_over_2_pct = RLM_res.loc[abs(RLM_res['Tvalue'])>=2,:].shape[0] / RLM_res.shape[0]
        print('-'*30)
        print('REGRESSION RESULT: \n   F_over_0_pct: %f \n   avg_abs_T: %f \n   abs_T_over_2_pct: %f \n' %
              (F_over_0_pct,avg_abs_T,abs_T_over_2_pct))




if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    strategy = StrategyBase()
    start = 20130101
    end = 20160101
    mkt_data = strategy.get_basic_info(start,end)
    ep_cut = strategy.influx.getDataMultiprocess('DailyFactor_Gus','Value',start,end,['code','EPcut_TTM'])
    ep_cut = ep_cut.dropna(subset=['EPcut_TTM'])
    print('epcut loaded!')
    strategy.test_factor(ep_cut,'EPcut_TTM',mkt_data,standardize='z',remove_outlier=False)