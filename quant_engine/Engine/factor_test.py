import global_constant
import pandas as pd
import numpy as np
from rdf_data import rdf_data
from influxdb_data import influxdbData
from data_process import DataProcess
from joblib import Parallel,delayed,parallel_backend
import warnings
import statsmodels.api as sm
from industry_neutral_engine import IndustryNeutralEngine
import logging
import datetime


class FactorTest:
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
        idsty_size_cols = \
            processed_factor.columns.difference(['code','date','next_1_day','return','next_return',factor_field])
        for date in dates:
            day_code       = processed_factor.loc[processed_factor['date']==date,'code']
            day_factor     = processed_factor.loc[processed_factor['date']==date,factor_field]
            day_idsty_size = processed_factor.loc[processed_factor['date']==date,idsty_size_cols]
            OLS_est        = sm.OLS(day_factor, day_idsty_size).fit()
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
        F = []
        T = []
        for date in dates:
            day_factor = processed_factor.loc[processed_factor['date'] == date, factor_field]
            day_return = processed_factor.loc[processed_factor['date'] == date, 'next_return']
            RLM_est = sm.RLM(day_return, day_factor, M=sm.robust.norms.HuberT()).fit()
            day_RLM_para = RLM_est.params
            day_Tvalue = RLM_est.tvalues
            F.append(day_RLM_para.iloc[0])
            T.append(day_Tvalue.iloc[0])
        return np.array([F,T])


    @staticmethod
    def job_IC(processed_factor, factor_field, dates):
        day_IC = []
        IC_date = []
        for date in dates:
            day_factor = processed_factor.loc[processed_factor['date'] == date, factor_field]
            day_return = processed_factor.loc[processed_factor['date'] == date, 'next_return']
            day_IC.append(np.corrcoef(day_factor,day_return)[0,1])
            IC_date.append(date)
        return pd.Series(day_IC,index=IC_date)


    # 等权所用的分组job
    @staticmethod
    def job_group_equal_weight(dates,groups,factor,factor_field):
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
                    day_industry_factor['weight_in_industry'] = 100 / day_industry_factor.shape[0]
                    day_industry_factor['group'] = 'same group'
                else:
                    day_industry_factor['group'] = pd.qcut(day_industry_factor[factor_field], 5, labels=labels)
                    group_counts = day_industry_factor['group'].value_counts()
                    day_industry_factor['weight_in_industry'] = day_industry_factor.apply(
                        lambda row:100/group_counts[row['group']],axis=1)
                res.append(day_industry_factor)
        res_df = pd.concat(res)
        res_df.set_index('next_1_day',inplace=True)
        return res_df


    # 市值加权所用的分组job
    @staticmethod
    def job_group_nonequal_weight(dates,groups,factor,factor_field):
        labels = []
        for i in range(1, groups + 1):
            labels.append('group_' + str(i))
        res = []
        for date in dates:
            day_factor = factor.loc[factor['date']==date,:].copy()
            industries = day_factor['industry'].unique()
            for ind in industries:
                day_industry_factor = day_factor.loc[day_factor['industry']==ind,:].copy()
                day_industry_size_sum = day_industry_factor['size'].sum()
                # 行业成分不足10支票时，所有group配置一样
                if day_industry_factor.shape[0] < 10:
                    day_industry_factor['weight_in_industry'] = 100/day_industry_size_sum * day_industry_factor['size']
                    day_industry_factor['group'] = 'same group'
                else:
                    day_industry_factor['group'] = pd.qcut(day_industry_factor[factor_field], 5, labels=labels)
                    group_size_sum = day_industry_factor.groupby('group')['size'].sum()
                    day_industry_factor['weight_in_industry'] = \
                        day_industry_factor.apply(lambda row: 100/group_size_sum[row['group']]*row['size'], axis=1)
                res.append(day_industry_factor)
        res_df = pd.concat(res)
        res_df.set_index('next_1_day', inplace=True)
        return res_df


    @staticmethod
    def job_backtest(group_name,grouped_weight,benchmark,adj_interval,cash_reserve,price_field,indu_field,data_input,
                     stock_capital,stk_slippage,stk_fee,logger_lvl=logging.INFO):
        weight = grouped_weight.loc[grouped_weight['group']==group_name,['code','industry','weight_in_industry']].copy()
        BE = IndustryNeutralEngine(stock_capital,stk_slippage,stk_fee,save_name=group_name,logger_lvl=logger_lvl)
        start = weight.index[0].strftime('%Y%m%d')
        end = weight.index[-1].strftime('%Y%m%d')
        portfolio_value = BE.run(weight,start,end,benchmark,adj_interval,cash_reserve,price_field,indu_field,data_input)
        portfolio_value.rename(columns={'TotalValue': group_name+'_TotalValue'},inplace=True)
        portfolio_value = pd.DataFrame(portfolio_value[group_name+'_TotalValue'])
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


    # 股票行情预处理，过滤st，行业字段变哑变量，增加市值字段
    def mkt_data_preprocess(self,mkt_data,size_data):
        # 过滤停牌(停牌没有收益率)
        mkt_data = mkt_data.loc[(mkt_data['status'] != '停牌') & (pd.notnull(mkt_data['status'])), :]
        # 过滤st
        if self.filter_st:
            mkt_data = mkt_data.loc[mkt_data['isST']==False,:]
        mkt_data['return'] = mkt_data['close']/mkt_data['preclose'] -1
        # 计算 next date
        mkt_data = pd.merge(mkt_data,self.get_next_trade_day(mkt_data,1),left_index=True,right_index=True,how='left')
        rtn_data = mkt_data.loc[:, ['code', 'next_1_day', 'return']]
        rtn_data.index.names = ['date']
        rtn_data.reset_index(inplace=True)
        nxt_rtn_data = rtn_data.copy()
        nxt_rtn_data = nxt_rtn_data.loc[:, ['date', 'code', 'return']]
        nxt_rtn_data.rename(columns={'date':'next_1_day','return':'next_return'},inplace=True)
        rtn_data = pd.merge(rtn_data,nxt_rtn_data,on=['next_1_day','code'],how='left')
        mkt_data.set_index([mkt_data.index,'code'],inplace=True)
        mkt_data.index.names = ['date','code']
        industry_data = pd.get_dummies(mkt_data[self.industry_field])
        # 过滤掉没有行业信息的数据
        industry_data = industry_data.loc[~(industry_data==0).all(axis=1),:]
        industry_data.reset_index(inplace=True)
        mkt_data = pd.merge(rtn_data,industry_data,on=['date','code'])
        size_data = size_data.loc[:,['code',self.mkt_cap_field]]
        size_data.rename(columns={self.mkt_cap_field:'size'},inplace=True)
        # mkt cap 标准化
        dates = size_data.index.unique()
        split_dates = np.array_split(dates, 10)
        # mkt_cap 不需要remove outlier，否则银行板块全被抹平
        with parallel_backend('multiprocessing', n_jobs=4):
            parallel_res = Parallel()(delayed(FactorTest.cross_section_Z_standardize)
                                      (size_data, 'size', dates) for dates in split_dates)
        size_data = pd.concat(parallel_res)
        size_data.index.names = ['date']
        size_data.reset_index(inplace=True)
        mkt_data = pd.merge(mkt_data,size_data,on=['date','code'],how='left')
        return mkt_data


    # factor.index 是date
    # preprocessed_mkt_data 的date在columns里
    # 输出的中性化因子 date在columns里
    def orth_factor(self,factor_data,preprocessed_mkt_data):
        # 数据预处理
        dates = factor_data.index.unique()
        split_dates = np.array_split(dates, 30)
        if self.remove_outlier:
            with parallel_backend('multiprocessing', n_jobs=4):
                parallel_res = Parallel()(delayed(FactorTest.cross_section_remove_outlier)
                                          (factor_data, self.factor_field, dates) for dates in split_dates)
            factor_data = pd.concat(parallel_res)
            dates = factor_data.index.unique()
            split_dates = np.array_split(dates, 30)
            print('outlier remove finish!')
        if self.standardize == 'z':
            with parallel_backend('multiprocessing', n_jobs=4):
                parallel_res = Parallel()(delayed(FactorTest.cross_section_Z_standardize)
                                          (factor_data, self.factor_field, dates) for dates in split_dates)
            factor_data = pd.concat(parallel_res)
            print('Z_standardize finish!')
        elif self.standardize == 'rank':
            with parallel_backend('multiprocessing', n_jobs=4):
                parallel_res = Parallel()(delayed(FactorTest.cross_section_rank_standardize)
                                          (factor_data, self.factor_field, dates) for dates in split_dates)
            factor_data = pd.concat(parallel_res)
            print('rank_standardize finish!')
        else:
            pass
        factor_data.index.names = ['date']
        factor_data.reset_index(inplace=True)
        processed_factor = pd.merge(preprocessed_mkt_data,factor_data,on=['date','code'])
        # 去除行业和市值，得到新因子值
        dates = processed_factor['date'].unique()
        split_dates = np.array_split(dates,10)
        with parallel_backend('multiprocessing', n_jobs=4):
            parallel_res = Parallel()(delayed(FactorTest.job_orth)
                                      (processed_factor, self.factor_field, dates) for dates in split_dates)
        orth_factor = pd.concat(parallel_res)
        return orth_factor


    def validity_check(self,orth_factor,preprocessed_mkt_data,T_test=True):
        processed_factor = pd.merge(preprocessed_mkt_data,orth_factor,on=['date','code'])
        # 超过0.11或-0.11的return标记为异常数据，置为nan(新股本身剔除)
        processed_factor = processed_factor.loc[(processed_factor['next_return']<0.11) &
                                                (processed_factor['next_return']>-0.11),:]
        dates = processed_factor['date'].unique()
        split_dates = np.array_split(dates, 10)
        if T_test:
            # T检验
            with parallel_backend('multiprocessing', n_jobs=4):
                parallel_res = Parallel()(delayed(FactorTest.job_T_test)
                                          (processed_factor, self.factor_field, dates) for dates in split_dates)
            # 第一行F，第二行T
            RLM_result = np.concatenate(parallel_res,axis=1)
            Fvalues = RLM_result[0]
            Tvalues = RLM_result[1]
            F_over_0_pct = Fvalues[Fvalues>0].shape[0] / Fvalues.shape[0]
            avg_abs_T = abs(Tvalues).mean()
            abs_T_over_2_pct = abs(Tvalues)[abs(Tvalues)>=2].shape[0] / Tvalues.shape[0]
            self.summary_dict['F_over_0_pct'] = F_over_0_pct
            self.summary_dict['avg_abs_T'] = avg_abs_T
            self.summary_dict['abs_T_over_2_pct'] = abs_T_over_2_pct
            print('-' * 30)
            print('REGRESSION RESULT: \n   F_over_0_pct: %f \n   avg_abs_T: %f \n   abs_T_over_2_pct: %f \n' %
                  (F_over_0_pct, avg_abs_T, abs_T_over_2_pct))
        # 计算IC
        with parallel_backend('multiprocessing', n_jobs=4):
            parallel_res = Parallel()(delayed(FactorTest.job_IC)
                                      (processed_factor, self.factor_field, dates) for dates in split_dates)
        IC = pd.concat(parallel_res)
        IC_over_0_pct = IC[IC>0].shape[0] / IC.shape[0]
        abs_IC_over_20pct_pct = abs(IC)[abs(IC)>0.02].shape[0] / IC.shape[0]
        IR = IC.mean() / IC.std()
        self.summary_dict['IC_mean'] = IC.mean()
        self.summary_dict['IC_std'] = IC.std()
        self.summary_dict['IC_over_0_pct'] = IC_over_0_pct
        self.summary_dict['abs_IC_over_20pct_pct'] = abs_IC_over_20pct_pct
        self.summary_dict['IR'] = IR
        print('-' * 30)
        print('ICIR RESULT: \n   IC mean: %f \n   IC std: %f \n   IC_over_0_pct: %f \n   '
              'abs_IC_over_20pct_pct: %f \n   IR: %f \n' %
              (IC.mean(), IC.std(), IC_over_0_pct, abs_IC_over_20pct_pct, IR))
        IC.name = 'IC'
        return IC


    # 此处weight_field为行业内权重分配的field
    def group_factor(self,orth_factor,mkt_data,size_data,weight_field=None):
        benchmark_field = self.benchmark + '_weight'
        mkt_data.dropna(subset=[self.industry_field],inplace=True)
        mkt_data = mkt_data.loc[:,[benchmark_field,self.industry_field,'code','status']]
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
        industry_weight = pd.DataFrame(mkt_data.groupby(['date', self.industry_field])['next_1_day_' + benchmark_field].sum())
        industry_weight.reset_index(inplace=True)
        industry_weight.rename(columns={'next_1_day_' + benchmark_field: 'industry_weight',
                                        self.industry_field: 'industry'}, inplace=True)
        industry_weight = industry_weight.loc[industry_weight['industry_weight']>0,:]
        mkt_data.rename(columns={self.industry_field: 'industry'},inplace=True)
        mkt_data = pd.merge(mkt_data,industry_weight,on=['date','industry'])
        # 合并得到当天因子，行业，状态，下一交易日benchmark权重，下一交易日行业总权重
        orth_factor = pd.merge(orth_factor,mkt_data,on=['date','code'])
        # 如果没有size_field, 就等权
        if not weight_field:
            # 去掉当天停牌的股票
            orth_factor = orth_factor.loc[~((orth_factor['status']=='停牌')|pd.isnull(orth_factor['status'])),
                                          ['date','code','next_1_day',self.factor_field,'industry','industry_weight']]
            dates = orth_factor['date'].unique()
            split_dates = np.array_split(dates,30)
            with parallel_backend('multiprocessing', n_jobs=4):
                result_list = Parallel()(delayed(FactorTest.job_group_equal_weight)
                                        (dates,self.groups,orth_factor,self.factor_field) for dates in split_dates)
            grouped_weight = pd.concat(result_list)
            grouped_weight = grouped_weight.loc[grouped_weight['weight_in_industry']>0,:].sort_index()
            filename = global_constant.ROOT_DIR + 'Backtest_Result/Factor_Group_Weight/' + \
                       self.factor_field + '_' + str(self.groups) + 'groups_equal' + '.csv'
            grouped_weight.to_csv(filename,encoding='gbk')
        # 如果有weight_field，按照市值加权
        else:
            size_data = size_data.loc[:,['code',weight_field]]
            size_data.rename(columns={weight_field:'size'},inplace=True)
            size_data.index.names = ['date']
            size_data.reset_index(inplace=True)
            orth_factor = pd.merge(orth_factor,size_data,on=['date','code'])
            # 去掉当天停牌的股票
            orth_factor = orth_factor.loc[~((orth_factor['status'] == '停牌') | pd.isnull(orth_factor['status'])),
                                          ['date','code','next_1_day',self.factor_field,'size','industry','industry_weight']]
            dates = orth_factor['date'].unique()
            split_dates = np.array_split(dates, 30)
            with parallel_backend('multiprocessing', n_jobs=4):
                result_list = Parallel()(delayed(FactorTest.job_group_nonequal_weight)
                                         (dates, self.groups, orth_factor, self.factor_field) for dates in split_dates)
            grouped_weight = pd.concat(result_list)
            grouped_weight = grouped_weight.sort_index()
            grouped_weight = grouped_weight.loc[grouped_weight['weight_in_industry']>0, :]
            filename = global_constant.ROOT_DIR + 'Backtest_Result/Factor_Group_Weight/' + \
                       self.factor_field + '_' + str(self.groups) + 'groups_' + weight_field + '.csv'
            grouped_weight.to_csv(filename, encoding='gbk')
        return grouped_weight


    def group_backtest(self,grouped_weight,weight_mode):
        self.summary_dict['WeightMode'] = weight_mode
        start = grouped_weight.index.min().strftime('%Y%m%d')
        end = grouped_weight.index.max().strftime('%Y%m%d')
        bt_data = self.influx.getDataMultiprocess('DailyData_Gus','marketData',start,end)
        group = []
        for i in range(1,self.groups+1):
            group.append('group_'+str(i))
        with parallel_backend('multiprocessing', n_jobs=5):
            parallel_res = Parallel()(delayed(FactorTest.job_backtest)
                                      (g, grouped_weight, self.benchmark, self.adj_interval, self.cash_reserve,
                                       self.price_field, self.industry_field, bt_data, self.capital,
                                       self.stk_slippage, self.stk_fee, self.logger_lvl) for g in group)
        tot_res = pd.concat(parallel_res,axis=1)
        # 合并指数value
        comparing_index = ['000300.SH', '000905.SH']
        index_value = bt_data.loc[bt_data['code'].isin(comparing_index),['code','close']]
        index_value.set_index([index_value.index,'code'],inplace=True)
        index_value = index_value.unstack()['close']
        index_value['300+500'] = 0.5*index_value['000300.SH'] + 0.5*index_value['000905.SH']
        for col in index_value.columns:
            index_value[col] = index_value[col]/index_value[col].iloc[0] * self.capital
        tot_res = pd.merge(tot_res,index_value,left_index=True,right_index=True)
        filename = global_constant.ROOT_DIR +'/Backtest_Result/Group_Value/'+ self.save_name + '_' + weight_mode + '.csv'
        tot_res.to_csv(filename, encoding='gbk')
        # 计算指标
        self.summary_dict['TimePeriod'] = tot_res.index[0].strftime('%Y%m%d')+'~'+tot_res.index[-1].strftime('%Y%m%d')
        self.summary_dict['AnnRet_high_group'] = DataProcess.calc_ann_return(tot_res[group[-1]+'_TotalValue'])
        self.summary_dict['AnnRet_300'] = DataProcess.calc_ann_return(tot_res['000300.SH'])
        self.summary_dict['AnnRet_500'] = DataProcess.calc_ann_return(tot_res['000905.SH'])
        self.summary_dict['AnnRet_800'] = DataProcess.calc_ann_return(tot_res['300+500'])
        self.summary_dict['alpha_300'] = DataProcess.calc_alpha_ann_return(tot_res[group[-1]+'_TotalValue'],tot_res['000300.SH'])
        self.summary_dict['alpha_500'] = DataProcess.calc_alpha_ann_return(tot_res[group[-1]+'_TotalValue'],tot_res['000905.SH'])
        self.summary_dict['alpha_800'] = DataProcess.calc_alpha_ann_return(tot_res[group[-1]+'_TotalValue'],tot_res['300+500'])
        self.summary_dict['MDD_high_group'] = DataProcess.calc_max_draw_down(tot_res[group[-1]+'_TotalValue'])
        self.summary_dict['MDD_300'] = DataProcess.calc_max_draw_down(tot_res['000300.SH'])
        self.summary_dict['MDD_500'] = DataProcess.calc_max_draw_down(tot_res['000905.SH'])
        self.summary_dict['MDD_800'] = DataProcess.calc_max_draw_down(tot_res['300+500'])
        self.summary_dict['sharpe_high_group'] = DataProcess.calc_sharpe(tot_res[group[-1]+'_TotalValue'])
        self.summary_dict['sharpe_300'] = DataProcess.calc_sharpe(tot_res['000300.SH'])
        self.summary_dict['sharpe_500'] = DataProcess.calc_sharpe(tot_res['000905.SH'])
        self.summary_dict['sharpe_800'] = DataProcess.calc_sharpe(tot_res['300+500'])
        self.summary_dict['sharpe_alpha_300'] = DataProcess.calc_alpha_sharpe(tot_res[group[-1]+'_TotalValue'],tot_res['000300.SH'])
        self.summary_dict['sharpe_alpha_500'] = DataProcess.calc_alpha_sharpe(tot_res[group[-1]+'_TotalValue'],tot_res['000905.SH'])
        self.summary_dict['sharpe_alpha_800'] = DataProcess.calc_alpha_sharpe(tot_res[group[-1]+'_TotalValue'],tot_res['300+500'])
        return self.summary_dict.copy()

    def generate_report(self,*summary_dicts):
        reps = []
        for summary_dict in summary_dicts:
            rep = {self.factor_field: summary_dict}
            rep = pd.DataFrame(rep)
            rep = rep.reindex(index=['TimePeriod','WeightMode','F_over_0_pct','avg_abs_T','abs_T_over_2_pct','IC_mean',
                                     'IC_std','IC_over_0_pct','abs_IC_over_20pct_pct','IR','AnnRet_high_group',
                                     'AnnRet_300','AnnRet_500','AnnRet_800','alpha_300','alpha_500','alpha_800',
                                     'MDD_high_group','MDD_300','MDD_500','MDD_800','sharpe_high_group','sharpe_300',
                                     'sharpe_500','sharpe_800','sharpe_alpha_300','sharpe_alpha_500','sharpe_alpha_800'])
            reps.append(rep)
        rep = pd.concat(reps,axis=1)
        filename = global_constant.ROOT_DIR + '/Backtest_Result/Factor_Report/' + self.save_name + '.csv'
        rep.to_csv(filename, encoding='gbk')


    def run_factor_test(self,mkt_data,factor_data,factor_field,size_data,save_name,groups=5,filter_st=True,
                        standardize='z',remove_outlier=True,mkt_cap_field='ln_market_cap',benchmark='IC',
                        industry_field='citics_lv1_name',capital=5000000,cash_reserve=0.03,stk_slippage=0.001,
                        stk_fee=0.0001,price_field='vwap',adj_interval=5,logger_lvl=logging.ERROR):

        self.factor_field = factor_field
        self.save_name = save_name
        self.groups = groups
        self.filter_st = filter_st
        self.standardize = standardize
        self.remove_outlier = remove_outlier
        self.mkt_cap_field = mkt_cap_field
        self.benchmark = benchmark
        self.industry_field = industry_field
        self.capital = capital
        self.cash_reserve = cash_reserve
        self.stk_slippage = stk_slippage
        self.stk_fee = stk_fee
        self.price_field = price_field
        self.adj_interval = adj_interval
        self.logger_lvl = logger_lvl
        self.summary_dict = {}
        #---------------------------------------------------------------

        preprocessed_mkt_data = self.mkt_data_preprocess(mkt_data,size_data)
        print('mkt data preprocessing finish')
        print('-'*30)
        orth_factor = self.orth_factor(factor_data,preprocessed_mkt_data)
        print('factor orthing finish')
        print('-' * 30)
        self.validity_check(orth_factor,preprocessed_mkt_data)
        print('validity checking finish')
        print('-' * 30)
        grouped_equal_weight = self.group_factor(orth_factor,mkt_data,None,None)
        grouped_nonequal_weight = self.group_factor(orth_factor,mkt_data,size_data,'market_cap')
        print('factor grouping finish')
        print('-' * 30)
        summary1 = self.group_backtest(grouped_equal_weight,'equal')
        summary2 = self.group_backtest(grouped_nonequal_weight,'market_cap')
        print('group backtest finish')
        print('-' * 30)
        self.generate_report(summary1,summary2)
        print('report got')


if __name__ == '__main__':
    print(datetime.datetime.now())
    warnings.filterwarnings("ignore")
    strategy = FactorTest()

    start = 20120101
    end = 20121231
    mkt_data = strategy.influx.getDataMultiprocess('DailyData_Gus', 'marketData', start, end, None)
    mkt_data = mkt_data.loc[pd.notnull(mkt_data['IF_weight'])|pd.notnull(mkt_data['IC_weight']),:]
    factor = strategy.influx.getDataMultiprocess('DailyFactor_Gus','Growth',start,end,['code','EPcut_TTM_growthY'])
    factor = factor.dropna(subset=['EPcut_TTM_growthY'])
    print('factor loaded!')
    size_data = strategy.influx.getDataMultiprocess('DailyFactor_Gus','Size',start,end,None)
    strategy.run_factor_test(mkt_data,factor,'EPcut_TTM_growthY',size_data,'EPcut_TTM_growthY',benchmark='IF',
                             mkt_cap_field='ln_market_cap')