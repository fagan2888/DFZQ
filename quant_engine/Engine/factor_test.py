import global_constant
import pandas as pd
import numpy as np
from rdf_data import rdf_data
from influxdb_data import influxdbData
from data_process import DataProcess
from joblib import Parallel, delayed, parallel_backend
import warnings
import statsmodels.api as sm
from industry_neutral_engine import IndustryNeutralEngine
import logging
import datetime


class FactorTest:
    @staticmethod
    def job_T_test(processed_factor, factor_field, dates):
        F = []
        T = []
        for date in dates:
            day_factor = processed_factor.loc[processed_factor['date'] == date, factor_field]
            day_return = processed_factor.loc[processed_factor['date'] == date, 'next_period_return']
            RLM_est = sm.RLM(day_return, day_factor, M=sm.robust.norms.HuberT()).fit()
            day_RLM_para = RLM_est.params
            day_Tvalue = RLM_est.tvalues
            F.append(day_RLM_para.iloc[0])
            T.append(day_Tvalue.iloc[0])
        return np.array([F, T])

    @staticmethod
    def job_IC(processed_factor, factor_field, dates):
        day_IC = []
        IC_date = []
        for date in dates:
            day_factor = processed_factor.loc[processed_factor['date'] == date, factor_field]
            day_return = processed_factor.loc[processed_factor['date'] == date, 'next_period_return']
            day_IC.append(np.corrcoef(day_factor, day_return)[0, 1])
            IC_date.append(date)
        return pd.Series(day_IC, index=IC_date)

    # 等权所用的分组job
    @staticmethod
    def job_group_equal_weight(dates, groups, factor, factor_field):
        labels = []
        for i in range(1, groups + 1):
            labels.append('group_' + str(i))
        res = []
        for date in dates:
            day_factor = factor.loc[factor['date'] == date, :].copy()
            industries = day_factor['industry'].unique()
            for ind in industries:
                day_industry_factor = day_factor.loc[day_factor['industry'] == ind, :].copy()
                # 行业成分不足10支票时，所有group配置一样
                if day_industry_factor.shape[0] < 10:
                    day_industry_factor['group'] = 'same group'
                    day_industry_factor['weight_in_industry'] = 100 / day_industry_factor.shape[0]
                else:
                    day_industry_factor['group'] = pd.qcut(day_industry_factor[factor_field], 5, labels=labels)
                    group_counts = day_industry_factor['group'].value_counts()
                    day_industry_factor['weight_in_industry'] = day_industry_factor.apply(
                        lambda row: 100 / group_counts[row['group']], axis=1)
                res.append(day_industry_factor)
        res_df = pd.concat(res)
        res_df.set_index('next_1_day', inplace=True)
        return res_df

    # 市值加权所用的分组job
    @staticmethod
    def job_group_nonequal_weight(dates, groups, factor, factor_field):
        labels = []
        for i in range(1, groups + 1):
            labels.append('group_' + str(i))
        res = []
        for date in dates:
            day_factor = factor.loc[factor['date'] == date, :].copy()
            industries = day_factor['industry'].unique()
            for ind in industries:
                day_industry_factor = day_factor.loc[day_factor['industry'] == ind, :].copy()
                day_industry_size_sum = day_industry_factor['size'].sum()
                # 行业成分不足10支票时，所有group配置一样
                if day_industry_factor.shape[0] < 10:
                    day_industry_factor['weight_in_industry'] = 100 / day_industry_size_sum * \
                                                                day_industry_factor['size']
                    day_industry_factor['group'] = 'same group'
                else:
                    day_industry_factor['group'] = pd.qcut(day_industry_factor[factor_field], 5, labels=labels)
                    group_size_sum = day_industry_factor.groupby('group')['size'].sum()
                    day_industry_factor['weight_in_industry'] = \
                        day_industry_factor.apply(lambda row: 100 / group_size_sum[row['group']] * row['size'], axis=1)
                res.append(day_industry_factor)
        res_df = pd.concat(res)
        res_df.set_index('next_1_day', inplace=True)
        return res_df

    @staticmethod
    def job_backtest(group_name, grouped_weight, benchmark, adj_interval, cash_reserve, price_field, indu_field,
                     data_input,
                     stock_capital, stk_slippage, stk_fee, logger_lvl=logging.INFO):
        weight = grouped_weight.loc[
            grouped_weight['group'] == group_name, ['code', 'industry', 'weight_in_industry']].copy()
        BE = IndustryNeutralEngine(stock_capital, stk_slippage, stk_fee, save_name=group_name, logger_lvl=logger_lvl)
        start = weight.index[0].strftime('%Y%m%d')
        end = weight.index[-1].strftime('%Y%m%d')
        portfolio_value = BE.run(weight, start, end, benchmark, adj_interval, cash_reserve, price_field, indu_field,
                                 data_input)
        portfolio_value.rename(columns={'TotalValue': group_name + '_TotalValue'}, inplace=True)
        portfolio_value = pd.DataFrame(portfolio_value[group_name + '_TotalValue'])
        print('%s backtest finish!' % group_name)
        return portfolio_value

    def __init__(self):
        self.rdf = rdf_data()
        self.calendar = self.rdf.get_trading_calendar()
        self.influx = influxdbData()

    # days 决定next_period_return 的周期
    def validity_check(self, neutral_factor, mkt_data, T_test=True):
        mkt_next_return = DataProcess.add_next_period_return(mkt_data, self.calendar, self.change_days)
        processed_factor = pd.merge(mkt_next_return, neutral_factor, on=['date', 'code'])
        # 超过0.11或-0.11的return标记为异常数据，置为nan(新股本身剔除)
        processed_factor.dropna(subset=['next_period_return'], inplace=True)
        processed_factor = processed_factor.loc[(processed_factor['next_period_return'] < 0.11) &
                                                (processed_factor['next_period_return'] > -0.11), :]
        dates = processed_factor['date'].unique()
        split_dates = np.array_split(dates, 10)
        if T_test:
            # T检验
            with parallel_backend('multiprocessing', n_jobs=4):
                parallel_res = Parallel()(delayed(FactorTest.job_T_test)
                                          (processed_factor, self.factor_field, dates) for dates in split_dates)
            # 第一行F，第二行T
            RLM_result = np.concatenate(parallel_res, axis=1)
            Fvalues = RLM_result[0]
            Tvalues = RLM_result[1]
            F_over_0_pct = Fvalues[Fvalues > 0].shape[0] / Fvalues.shape[0]
            avg_abs_T = abs(Tvalues).mean()
            abs_T_over_2_pct = abs(Tvalues)[abs(Tvalues) >= 2].shape[0] / Tvalues.shape[0]
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
        IC_over_0_pct = IC[IC > 0].shape[0] / IC.shape[0]
        abs_IC_over_20pct_pct = abs(IC)[abs(IC) > 0.02].shape[0] / IC.shape[0]
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
    def group_factor(self, neutral_factor, mkt_data, size_data, weight_field):
        mkt_data = mkt_data.loc[(mkt_data['status'] != '停牌') & (pd.notnull(mkt_data['status'])) &
                                (pd.notnull(mkt_data[self.industry_field])), ['code', self.industry_field]]
        idxs = mkt_data.index.unique()
        next_date_dict = {}
        for idx in idxs:
            next_date_dict.update(DataProcess.get_next_date(self.calendar, idx, 1))
        mkt_data['next_1_day'] = mkt_data.apply(lambda row: next_date_dict[row.name], axis=1)
        mkt_data.index.names = ['date']
        mkt_data.reset_index(inplace=True)
        mkt_data.rename(columns={self.industry_field: 'industry'}, inplace=True)
        merge_df = pd.merge(neutral_factor,mkt_data,on=['date','code'])
        res_dict = {}
        # 先按等权测试
        dates = merge_df['date'].unique()
        split_dates = np.array_split(dates, 10)
        with parallel_backend('multiprocessing', n_jobs=4):
            result_list = Parallel()(delayed(FactorTest.job_group_equal_weight)
                                     (dates, self.groups, merge_df, self.factor_field) for dates in split_dates)
        grouped_weight = pd.concat(result_list)
        grouped_weight = grouped_weight.loc[grouped_weight['weight_in_industry'] > 0, :].sort_index()
        filename = global_constant.ROOT_DIR + 'Backtest_Result/Factor_Group_Weight/' + \
                   self.factor_field + '_' + str(self.groups) + 'groups_EquW' + '.csv'
        grouped_weight.to_csv(filename, encoding='gbk')
        res_dict['EquW'] = grouped_weight.copy()
        # 再按市值加权测试
        size_data = size_data.loc[:, ['code', weight_field]]
        size_data.rename(columns={weight_field: 'size'}, inplace=True)
        size_data.index.names = ['date']
        size_data.reset_index(inplace=True)
        merge_df = pd.merge(merge_df, size_data, on=['date', 'code'])
        dates = merge_df['date'].unique()
        split_dates = np.array_split(dates, 30)
        with parallel_backend('multiprocessing', n_jobs=4):
            result_list = Parallel()(delayed(FactorTest.job_group_nonequal_weight)
                                     (dates, self.groups, merge_df, self.factor_field) for dates in split_dates)
        grouped_weight = pd.concat(result_list)
        grouped_weight = grouped_weight.loc[grouped_weight['weight_in_industry'] > 0, :].sort_index()
        filename = global_constant.ROOT_DIR + 'Backtest_Result/Factor_Group_Weight/' + \
                   self.factor_field + '_' + str(self.groups) + 'groups_' + weight_field + '.csv'
        grouped_weight.to_csv(filename, encoding='gbk')
        res_dict['NonEquW'] = grouped_weight.copy()
        return res_dict


    def group_backtest(self, grouped_weight_dict,mkt_data):
        for weight_mode in grouped_weight_dict.keys():
            print('%s groups is backtesting' %weight_mode)
            grouped_weight = grouped_weight_dict[weight_mode]
            group = []
            for i in range(1, self.groups + 1):
                group.append('group_' + str(i))
            with parallel_backend('multiprocessing', n_jobs=5):
                parallel_res = Parallel()(delayed(FactorTest.job_backtest)
                                          (g, grouped_weight, self.benchmark, self.adj_interval,
                                           self.cash_reserve,self.price_field, self.industry_field, mkt_data,
                                           self.capital,self.stk_slippage, self.stk_fee, self.logger_lvl)
                                          for g in group)
            tot_res = pd.concat(parallel_res, axis=1)
            # 合并指数value
            comparing_index = ['000300.SH', '000905.SH']
            index_value = mkt_data.loc[mkt_data['code'].isin(comparing_index), ['code', 'close']]
            index_value.set_index([index_value.index, 'code'], inplace=True)
            index_value = index_value.unstack()['close']
            index_value['300+500'] = 0.5 * index_value['000300.SH'] + 0.5 * index_value['000905.SH']
            for col in index_value.columns:
                index_value[col] = index_value[col] / index_value[col].iloc[0] * self.capital
            tot_res = pd.merge(tot_res, index_value, left_index=True, right_index=True)
            filename = global_constant.ROOT_DIR + '/Backtest_Result/Group_Value/' + self.save_name + '_' +\
                       weight_mode + '.csv'
            tot_res.to_csv(filename, encoding='gbk')
            self.summary_dict['AnnRet_high_group_' + weight_mode] = \
                DataProcess.calc_ann_return(tot_res[group[-1] + '_TotalValue'])
            self.summary_dict['alpha_300_' + weight_mode] = \
                DataProcess.calc_alpha_ann_return(tot_res[group[-1] + '_TotalValue'], tot_res['000300.SH'])
            self.summary_dict['alpha_500_' + weight_mode] = \
                DataProcess.calc_alpha_ann_return(tot_res[group[-1] + '_TotalValue'], tot_res['000905.SH'])
            self.summary_dict['alpha_800_' + weight_mode] = \
                DataProcess.calc_alpha_ann_return(tot_res[group[-1] + '_TotalValue'], tot_res['300+500'])
            self.summary_dict['MDD_high_group_' + weight_mode] = \
                DataProcess.calc_max_draw_down(tot_res[group[-1] + '_TotalValue'])
            self.summary_dict['sharpe_high_group_' + weight_mode] = \
                DataProcess.calc_sharpe(tot_res[group[-1] + '_TotalValue'])
            self.summary_dict['sharpe_alpha_300_' + weight_mode] = \
                DataProcess.calc_alpha_sharpe(tot_res[group[-1] + '_TotalValue'], tot_res['000300.SH'])
            self.summary_dict['sharpe_alpha_500_' + weight_mode] = \
                DataProcess.calc_alpha_sharpe(tot_res[group[-1] + '_TotalValue'], tot_res['000905.SH'])
            self.summary_dict['sharpe_alpha_800_' + weight_mode] = \
                DataProcess.calc_alpha_sharpe(tot_res[group[-1] + '_TotalValue'], tot_res['300+500'])
        # -------------------------------------
        self.summary_dict['Start_Time'] = tot_res.index[0].strftime('%Y%m%d')
        self.summary_dict['End_Time'] = tot_res.index[-1].strftime('%Y%m%d')
        self.summary_dict['AnnRet_300'] = DataProcess.calc_ann_return(tot_res['000300.SH'])
        self.summary_dict['AnnRet_500'] = DataProcess.calc_ann_return(tot_res['000905.SH'])
        self.summary_dict['AnnRet_800'] = DataProcess.calc_ann_return(tot_res['300+500'])
        self.summary_dict['MDD_300'] = DataProcess.calc_max_draw_down(tot_res['000300.SH'])
        self.summary_dict['MDD_500'] = DataProcess.calc_max_draw_down(tot_res['000905.SH'])
        self.summary_dict['MDD_800'] = DataProcess.calc_max_draw_down(tot_res['300+500'])
        self.summary_dict['sharpe_300'] = DataProcess.calc_sharpe(tot_res['000300.SH'])
        self.summary_dict['sharpe_500'] = DataProcess.calc_sharpe(tot_res['000905.SH'])
        self.summary_dict['sharpe_800'] = DataProcess.calc_sharpe(tot_res['300+500'])
        return self.summary_dict.copy()

    def generate_report(self, summary_dict):
        rep = {self.factor_field: summary_dict}
        rep = pd.DataFrame(rep)
        rep = rep.reindex(index=['Start_Time','End_Time', 'F_over_0_pct', 'avg_abs_T', 'abs_T_over_2_pct', 'IC_mean',
                                 'IC_std', 'IC_over_0_pct', 'abs_IC_over_20pct_pct', 'IR', 'AnnRet_high_group_EquW',
                                 'AnnRet_high_group_NonEquW', 'AnnRet_300', 'AnnRet_500', 'AnnRet_800',
                                 'alpha_300_EquW', 'alpha_300_NonEquW', 'alpha_500_EquW', 'alpha_500_NonEquW',
                                 'alpha_800_EquW', 'alpha_800_NonEquW', 'MDD_high_group_EquW', 'MDD_high_group_NonEquW',
                                 'MDD_300', 'MDD_500', 'MDD_800', 'sharpe_high_group_EquW', 'sharpe_high_group_NonEquW',
                                 'sharpe_300', 'sharpe_500', 'sharpe_800', 'sharpe_alpha_300_EquW',
                                 'sharpe_alpha_300_NonEquW', 'sharpe_alpha_500_EquW', 'sharpe_alpha_500_NonEquW',
                                 'sharpe_alpha_800_EquW', 'sharpe_alpha_800_NonEquW'])
        filename = global_constant.ROOT_DIR + '/Backtest_Result/Factor_Report/' + self.save_name + '.csv'
        rep.to_csv(filename, encoding='gbk')

    def run_factor_test(self, mkt_data, factor_data, factor_field, size_data, save_name, change_days=5, groups=5,
                        filter_st=True, standardize='z', remove_outlier=True, mkt_cap_field='ln_market_cap',
                        benchmark='IC', industry_field='citics_lv1_name', capital=5000000, cash_reserve=0.03,
                        stk_slippage=0.001, stk_fee=0.0001, price_field='vwap', adj_interval=5, logger_lvl=logging.ERROR):
        self.factor_field = factor_field
        self.save_name = save_name
        self.change_days = change_days
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
        # ---------------------------------------------------------------
        industry_dummies = DataProcess.get_industry_dummies(mkt_data)
        neutral_factor = DataProcess.neutralize(factor_data, factor_field, industry_dummies, size_data)
        print('factor neutralization finish')
        print('-' * 30)
        self.validity_check(neutral_factor, mkt_data, T_test=True)
        print('validity checking finish')
        print('-' * 30)
        grouped_weight_dict = self.group_factor(neutral_factor,mkt_data,size_data,'market_cap')
        print('factor grouping finish')
        print('-' * 30)
        summary_dict = self.group_backtest(grouped_weight_dict,mkt_data)
        print('group backtest finish')
        print('-' * 30)
        self.generate_report(summary_dict)
        print('report got')


if __name__ == '__main__':
    print(datetime.datetime.now())
    warnings.filterwarnings("ignore")
    strategy = FactorTest()

    start = 20120101
    end = 20131231
    mkt_data = strategy.influx.getDataMultiprocess('DailyData_Gus', 'marketData', start, end, None)
    factor = strategy.influx.getDataMultiprocess('DailyFactor_Gus', 'Growth', start, end, ['code', 'EPcut_TTM_growthY'])
    factor = factor.dropna(subset=['EPcut_TTM_growthY'])
    print('factor loaded!')
    size_data = strategy.influx.getDataMultiprocess('DailyFactor_Gus', 'Size', start, end, None)
    strategy.run_factor_test(mkt_data, factor, 'EPcut_TTM_growthY', size_data, 'EPcut_TTM_growthY', benchmark='IF',
                             mkt_cap_field='ln_market_cap')