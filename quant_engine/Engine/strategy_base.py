import global_constant
import pandas as pd
import numpy as np
from rdf_data import rdf_data
from influxdb_data import influxdbData
from data_process import DataProcess
from joblib import Parallel,delayed,parallel_backend
import warnings
import statsmodels.api as sm
from backtest_engine import BacktestEngine
import logging
import datetime


class StrategyBase:
    @staticmethod
    def cross_section_remove_outlier(factor_data,factor_field,dates):
        res = []
        for date in dates:
            day_factor = factor_data.loc[date,:].copy()
            # 滤去周末出财报造成周末有因子的情况
            if day_factor.shape[0] < 100:
               pass
            else:
                day_factor.loc[:,factor_field] = DataProcess.remove_outlier(day_factor[factor_field])
                res.append(day_factor)
        dates_factor = pd.concat(res)
        return dates_factor


    @staticmethod
    def cross_section_Z_standardize(factor_data,factor_field,dates):
        res = []
        for date in dates:
            day_factor = factor_data.loc[date, :].copy()
            # 滤去周末出财报造成周末有因子的情况
            if day_factor.shape[0] < 100:
               pass
            else:
                day_factor.loc[:,factor_field] = DataProcess.Z_standardize(day_factor[factor_field])
                res.append(day_factor)
        dates_factor = pd.concat(res)
        return dates_factor


    @staticmethod
    def cross_section_rank_standardize(factor_data,factor_field,dates):
        res = []
        for date in dates:
            day_factor = factor_data.loc[date, :].copy()
            # 滤去周末出财报造成周末有因子的情况
            if day_factor.shape[0] < 100:
                pass
            else:
                day_factor.loc[:,factor_field] = DataProcess.rank_standardize(day_factor[factor_field])
                res.append(day_factor)
        dates_factor = pd.concat(res)
        return dates_factor


    @staticmethod
    # 正交后的factor中date在columns里
    def job_orth(processed_factor,factor_field,dates):
        orth_factor_res = []
        idsty_size_cols = processed_factor.columns.difference(['code','date','former_1_day','next_1_day','return',factor_field])
        for date in dates:
            day_code       = processed_factor.loc[processed_factor['date']==date,'code']
            day_factor     = processed_factor.loc[processed_factor['date']==date,factor_field]
            day_idsty_size = processed_factor.loc[processed_factor['date']==date,idsty_size_cols]
            OLS_est           = sm.OLS(day_factor, day_idsty_size).fit()
            day_orthed_factor = OLS_est.resid
            day_orthed_factor.name = factor_field
            # 得到正交化后的因子值
            day_orthed_factor = pd.concat([day_code, day_orthed_factor], axis=1)
            day_orthed_factor['date'] = date
            orth_factor_res.append(day_orthed_factor)
        orth_factor = pd.concat(orth_factor_res)
        return orth_factor


    @staticmethod
    def job_T_test(processed_factor,factor_field,dates):
        RLM_res = []
        for date in dates:
            day_factor = processed_factor.loc[processed_factor['date'] == date, factor_field]
            day_return = processed_factor.loc[processed_factor['date'] == date, 'return']
            RLM_est = sm.RLM(day_return, day_factor, M=sm.robust.norms.HuberT()).fit()
            day_RLM_para = RLM_est.params
            day_Tvalue = RLM_est.tvalues
            day_RLM_result = pd.DataFrame({'Fvalue':day_RLM_para.iloc[0],'Tvalue':day_Tvalue.iloc[0]},index=[date])
            RLM_res.append(day_RLM_result)
        RLM_result = pd.concat(RLM_res)
        return RLM_result


    @staticmethod
    def get_group_weight(dates,groups,factor,factor_field):
        labels = []
        for i in range(1,groups+1):
            labels.append('group_'+str(i))
        res = []
        for date in dates:
            day_factor = factor.loc[factor['date']==date,:].copy()
            industries = day_factor['industry'].unique()
            for ind in industries:
                day_industry_factor = day_factor.loc[day_factor['industry']==ind,:].copy()
                # 行业成分不足10支票时，所有group配置一样
                if day_industry_factor.shape[0] < 10:
                    #day_industry_factor['weight'] = day_industry_factor['industry_weight'] / day_industry_factor.shape[0]
                    day_industry_factor['weight_in_industry'] = 100 / day_industry_factor.shape[0]
                    day_industry_factor['group'] = 'same group'
                else:
                    day_industry_factor['group'] = pd.qcut(day_industry_factor[factor_field],5,labels=labels)
                    group_counts = day_industry_factor['group'].value_counts()
                    #day_industry_factor['weight'] = \
                    #    day_industry_factor.apply(lambda row:row['industry_weight']/group_counts[row['group']],axis=1)
                    day_industry_factor['weight_in_industry'] = day_industry_factor.apply(lambda row:100/group_counts[row['group']],axis=1)
                res.append(day_industry_factor)
        res_df = pd.concat(res)
        res_df.set_index('next_1_day',inplace=True)
        return res_df


    @staticmethod
    def job_backtest(grouped_weight,group_name,cash_reserve,data_input,price_field='vwap',stock_capital=1000000,
                     stk_slippage=0,stk_fee=0,logger_lvl=logging.INFO):
        weight = grouped_weight.loc[grouped_weight['group']==group_name,['code','weight']].copy()
        BE = BacktestEngine(stock_capital,stk_slippage,stk_fee,group_name,logger_lvl)
        start = weight.index[0].strftime('%Y%m%d')
        end = weight.index[-1].strftime('%Y%m%d')
        portfolio_value = BE.run(weight,start,end,cash_reserve,data_input=data_input,price_field=price_field)
        portfolio_value.columns = group_name + '_' + portfolio_value.columns
        print('%s backtest finish!' %group_name)
        return portfolio_value



    def __init__(self):
        self.rdf = rdf_data()
        self.calendar = self.rdf.get_trading_calendar()
        self.influx = influxdbData()


    def get_former_trade_day(self,mkt_data,days):
        trade_day = pd.DataFrame(mkt_data.index.unique().tolist(), columns=['date'])
        field = 'former_' + str(days) + '_day'
        trade_day[field] = trade_day['date'].apply(lambda x: self.calendar[self.calendar < x].iloc[-1*days])
        trade_day.set_index('date', inplace=True)
        return trade_day


    def get_next_trade_day(self,mkt_data,days):
        trade_day = pd.DataFrame(mkt_data.index.unique().tolist(), columns=['date'])
        field = 'next_' + str(days) + '_day'
        trade_day[field] = trade_day['date'].apply(lambda x: self.calendar[self.calendar > x].iloc[days-1])
        trade_day.set_index('date', inplace=True)
        return trade_day


    # 股票行情预处理，过滤st，行业字段，市值字段
    def mkt_data_preprocess(self,mkt_data,filter_st=True,industry='citics_lv1_name',mkt_cap_field='ln_market_cap'):
        # 过滤停牌(停牌没有收益率)
        mkt_data = mkt_data.loc[(mkt_data['status'] != '停牌') & (pd.notnull(mkt_data['status'])), :]
        # 过滤st
        if filter_st:
            mkt_data = mkt_data.loc[mkt_data['isST']==False,:]
        mkt_data['return'] = mkt_data['close']/mkt_data['preclose'] -1
        # 超过0.11或-0.11的return标记为异常数据，置为nan(新股本身剔除)
        mkt_data = mkt_data.loc[(mkt_data['return']<0.11) | (mkt_data['return']>-0.11),:]
        # 计算former date 和 next date
        mkt_data = pd.merge(mkt_data,self.get_former_trade_day(mkt_data,1),left_index=True,right_index=True,how='left')
        mkt_data = pd.merge(mkt_data,self.get_next_trade_day(mkt_data,1),left_index=True,right_index=True,how='left')
        mkt_data.set_index([mkt_data.index,'code'],inplace=True)
        mkt_data.index.names = ['date','code']
        rtn_data = mkt_data.loc[:,['former_1_day','next_1_day','return']]
        industry_data = pd.get_dummies(mkt_data[industry])
        # 过滤掉没有行业信息的数据
        industry_data = industry_data.loc[~(industry_data==0).all(axis=1),:]
        rtn_data.reset_index(inplace=True)
        industry_data.reset_index(inplace=True)
        mkt_data = pd.merge(rtn_data,industry_data,on=['date','code'])
        # 今天的结果用于后一天交易，最后一天的数据不可用，所以end提前一天
        start = mkt_data['date'].iloc[0].strftime('%Y%m%d')
        end = mkt_data['next_1_day'].iloc[-1].strftime('%Y%m%d')
        size_data = self.influx.getDataMultiprocess('DailyFactor_Gus', 'Size',start,end,None)
        size_data = size_data.loc[:,['code',mkt_cap_field]]
        size_data.rename(columns={mkt_cap_field:'size'},inplace=True)
        # mkt cap 标准化
        dates = size_data.index.unique()
        split_dates = np.array_split(dates, 30)
        # mkt_cap 不需要remove outlier，否则银行板块全被抹平
        with parallel_backend('multiprocessing', n_jobs=4):
            parallel_res = Parallel()(delayed(StrategyBase.cross_section_Z_standardize)
                                      (size_data, 'size', dates) for dates in split_dates)
        size_data = pd.concat(parallel_res)
        size_data.index.names = ['date']
        size_data.reset_index(inplace=True)
        mkt_data = pd.merge(mkt_data,size_data,how='inner',on=['date','code'])
        return mkt_data


    # factor.index 是date
    # preprocessed_mkt_data 的date在columns里
    # 输出的正交化因子 date在columns里
    def orth_factor(self,factor_data,factor_field,preprocessed_mkt_data,standardize,remove_outlier):
        # 数据预处理
        dates = factor_data.index.unique()
        split_dates = np.array_split(dates, 30)
        if remove_outlier:
            with parallel_backend('multiprocessing', n_jobs=4):
                parallel_res = Parallel()(delayed(StrategyBase.cross_section_remove_outlier)
                                          (factor_data, factor_field,dates) for dates in split_dates)
            factor_data = pd.concat(parallel_res)
            print('outlier remove finish!')
        if standardize == 'z':
            with parallel_backend('multiprocessing', n_jobs=4):
                parallel_res = Parallel()(delayed(StrategyBase.cross_section_Z_standardize)
                                          (factor_data, factor_field,dates) for dates in split_dates)
            factor_data = pd.concat(parallel_res)
            print('Z_standardize finish!')
        elif standardize == 'rank':
            with parallel_backend('multiprocessing', n_jobs=4):
                parallel_res = Parallel()(delayed(StrategyBase.cross_section_rank_standardize)
                                          (factor_data, factor_field,dates) for dates in split_dates)
            factor_data = pd.concat(parallel_res)
            print('rank_standardize finish!')
        else:
            pass

        factor_data.index.names = ['date']
        factor_data.reset_index(inplace=True)
        processed_factor = pd.merge(preprocessed_mkt_data,factor_data,on=['date','code'])

        # 去除行业和市值，得到新因子
        dates = processed_factor['date'].unique()
        split_dates = np.array_split(dates,30)
        with parallel_backend('multiprocessing', n_jobs=4):
            parallel_res = Parallel()(delayed(StrategyBase.job_orth)
                                      (processed_factor, factor_field, dates) for dates in split_dates)
        orth_factor = pd.concat(parallel_res)
        return orth_factor


    def T_test(self,orth_factor,factor_field,preprocessed_mkt_data):
        processed_factor = pd.merge(preprocessed_mkt_data,orth_factor,on=['date','code'])
        dates = processed_factor['date'].unique()
        split_dates = np.array_split(dates, 30)
        with parallel_backend('multiprocessing', n_jobs=4):
            parallel_res = Parallel()(delayed(StrategyBase.job_T_test)
                                      (processed_factor, factor_field, dates) for dates in split_dates)
        RLM_result = pd.concat(parallel_res)
        F_over_0_pct = RLM_result.loc[RLM_result['Fvalue']>0,:].shape[0] / RLM_result.shape[0]
        avg_abs_T = abs(RLM_result['Tvalue']).sum() / RLM_result.shape[0]
        abs_T_over_2_pct = RLM_result.loc[abs(RLM_result['Tvalue'])>=2,:].shape[0] / RLM_result.shape[0]
        print('-'*30)
        print('REGRESSION RESULT: \n   F_over_0_pct: %f \n   avg_abs_T: %f \n   abs_T_over_2_pct: %f \n' %
              (F_over_0_pct,avg_abs_T,abs_T_over_2_pct))
        return RLM_result


        # 后续需添加因子收益率输出
        # 后续需添加因子ic值计算


    def group_factor(self,orth_factor,factor_field,mkt_data,groups=5,benchmark='IC',industry_field='citics_lv1_name'):
        benchmark_field = benchmark+'_weight'
        mkt_data.dropna(subset=[industry_field],inplace=True)
        mkt_data = mkt_data.loc[:,[benchmark_field,industry_field,'code','status']]
        next_1_day = self.get_next_trade_day(mkt_data,1)
        mkt_data = pd.merge(mkt_data,next_1_day,right_index=True,left_index=True,how='left')
        # 需用到后1天的权重信息
        # 如果这个股票今天停牌，则它不在今天的选股池内（就算明天复牌也不在）
        nxt_1_day_weight = mkt_data.loc[:,['code',benchmark_field]].copy()
        nxt_1_day_weight.reset_index(inplace=True)
        nxt_1_day_weight.rename(columns={'index':'next_1_day',benchmark_field:'next_1_day_'+benchmark_field},inplace=True)
        mkt_data.index.names = ['date']
        mkt_data.reset_index(inplace=True)
        mkt_data = pd.merge(mkt_data,nxt_1_day_weight,on=['next_1_day','code'],how='left')
        # 计算后一天的行业权重
        industry_weight = pd.DataFrame(mkt_data.groupby(['date', industry_field])['next_1_day_' + benchmark_field].sum())
        industry_weight.reset_index(inplace=True)
        industry_weight.rename(columns={'next_1_day_' + benchmark_field: 'industry_weight', industry_field: 'industry'},
                               inplace=True)
        industry_weight = industry_weight.loc[industry_weight['industry_weight']>0,:]
        mkt_data.rename(columns={industry_field: 'industry'},inplace=True)
        mkt_data = pd.merge(mkt_data,industry_weight,on=['date','industry'])
        # 合并得到当天因子，行业，状态，下一交易日benchmark权重，下一交易日行业总权重
        orth_factor = pd.merge(orth_factor,mkt_data,on=['date','code'])
        # 去掉当天停牌的股票
        orth_factor = orth_factor.loc[~((orth_factor['status']=='停牌')|pd.isnull(orth_factor['status'])),
                                      ['date','code','next_1_day',factor_field,'industry','industry_weight']]
        dates = orth_factor['date'].unique()
        split_dates = np.array_split(dates,30)
        with parallel_backend('multiprocessing', n_jobs=4):
            result_list =  Parallel()(delayed(StrategyBase.get_group_weight)(dates,groups,orth_factor,factor_field)
                                      for dates in split_dates)
        grouped_weight = pd.concat(result_list)
        grouped_weight = grouped_weight.sort_index()
        grouped_weight = grouped_weight.loc[grouped_weight['weight_in_industry']>0,:]
        filename = global_constant.ROOT_DIR + 'Backtest_Result/Factor_Group_Weight/' + \
                   factor_field + '_' + str(groups) + 'groups' + '.csv'
        grouped_weight.to_csv(filename,encoding='gbk')
        return grouped_weight

    '''
    def get_actual_weight(self,grouped_weight,mkt_data,group):
        mkt_data = mkt_data.loc[:,['code','status','open','high','low','close']].copy()
        mkt_data.index.names = 'date'
        mkt_data.reset_index(inplace=True)
        grouped_weight.drop('date',axis=1,inplace=True)
        grp_weight = grouped_weight.loc[grouped_weight['group']==group,:]
        grp_weight.index.names = 'date'
        grp_weight.reset_index(inplace=True)
        dates = grp_weight['date'].unique()
        # 记录最新的持仓股票和量
        latest_postions = pd.DataFrame()
        for date in dates:
            day_grp_weight = grp_weight.loc[grp_weight[date]==date,:]
            day_grp_weight = pd.merge(day_grp_weight,mkt_data,on=['date','code'],how='left')
            if latest_postions.empty:
                # 仓位为空时，认为买入了当天没有停牌且没有涨停的股票
                day_grp_weight = \
                    day_grp_weight.loc[~((day_grp_weight['status']=='停牌')|(pd.isnull(day_grp_weight['status']))) &
                                       ~((day_grp_weight['high']==round(day_grp_weight['open']*1.1,2)) &
                                         (day_grp_weight['high']==day_grp_weight['low'])),:]
            else:
    '''




    def group_backtest(self,capital,grouped_weight,groups,f_name):
        start = grouped_weight.index.min().strftime('%Y%m%d')
        end = grouped_weight.index.max().strftime('%Y%m%d')
        bt_data = self.influx.getDataMultiprocess('DailyData_Gus','marketData',start,end)
        group = []
        for i in range(1,groups+1):
            group.append('group_'+str(i))
        with parallel_backend('multiprocessing', n_jobs=4):
            parallel_res = Parallel()(delayed(StrategyBase.job_backtest)
                                      (grouped_weight=grouped_weight,group_name=g,cash_reserve=0,stock_capital=capital,
                                       logger_lvl=logging.ERROR,data_input=bt_data)
                                      for g in group)
        tot_res = pd.concat(parallel_res,axis=1)
        filename = global_constant.ROOT_DIR + '/Backtest_Result/Factor_Test/' + f_name + '.csv'
        tot_res.to_csv(filename,encoding='gbk')
        print('group backtest finish!')


    def run_factor_test(self,mkt_data,factor_data,factor_field,save_name,groups=5,standardize='z',remove_outlier=True,
                        benchmark='IC',industry_field='citics_lv1_name',capital=5000000):
        preprocessed_mkt_data = self.mkt_data_preprocess(mkt_data)
        print('mkt data preprocessing finish')
        orth_factor = self.orth_factor(factor_data,factor_field,preprocessed_mkt_data,standardize,remove_outlier)
        print('factor orthing finish')
        T_test_result = self.T_test(orth_factor,factor_field,preprocessed_mkt_data)
        print('T-test finish')
        grouped_weight = self.group_factor(orth_factor,factor_field,mkt_data,groups,benchmark,industry_field)
        print('factor grouping finish')
        self.group_backtest(capital,grouped_weight,groups,save_name)




if __name__ == '__main__':
    print(datetime.datetime.now())
    warnings.filterwarnings("ignore")
    strategy = StrategyBase()

    start = 20120101
    end = 20160901
    mkt_data = strategy.influx.getDataMultiprocess('DailyData_Gus', 'marketData', start, end, None)
    factor = strategy.influx.getDataMultiprocess('DailyFactor_Gus','Value',start,end,['code','EPcut_TTM'])
    factor = factor.dropna(subset=['EPcut_TTM'])
    print('factor loaded!')
    strategy.run_factor_test(mkt_data,factor,'EPcut_TTM','EPcut')