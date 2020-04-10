import global_constant
import pandas as pd
import numpy as np
import os.path
from data_process import DataProcess
from joblib import Parallel, delayed, parallel_backend
import warnings
import statsmodels.api as sm
from industry_neutral_engine import IndustryNeutralEngine
from strategy_base import StrategyBase
import logging
import datetime


class FactorTest(StrategyBase):
    @staticmethod
    def JOB_T_test(processed_factor, factor_field, dates):
        F_return = []
        T_return = []
        F_alpha = []
        T_alpha = []
        for date in dates:
            day_factor = processed_factor.loc[processed_factor['date'] == date, factor_field]
            day_return = processed_factor.loc[processed_factor['date'] == date, 'next_period_return']
            day_alpha = processed_factor.loc[processed_factor['date'] == date, 'next_period_alpha']
            RLM_est = sm.RLM(day_return, day_factor, M=sm.robust.norms.HuberT()).fit()
            day_RLM_para = RLM_est.params
            day_Tvalue = RLM_est.tvalues
            F_return.append(day_RLM_para.iloc[0])
            T_return.append(day_Tvalue.iloc[0])
            RLM_est = sm.RLM(day_alpha, day_factor, M=sm.robust.norms.HuberT()).fit()
            day_RLM_para = RLM_est.params
            day_Tvalue = RLM_est.tvalues
            F_alpha.append(day_RLM_para.iloc[0])
            T_alpha.append(day_Tvalue.iloc[0])
        return np.array([F_return, T_return, F_alpha, T_alpha])

    @staticmethod
    def JOB_IC(processed_factor, factor_field, dates):
        day_IC = []
        IC_date = []
        for date in dates:
            day_factor = processed_factor.loc[processed_factor['date'] == date, factor_field]
            day_return = processed_factor.loc[processed_factor['date'] == date, 'next_period_return']
            day_IC.append(day_factor.corr(day_return, method='spearman'))
            IC_date.append(date)
        return pd.Series(day_IC, index=IC_date)

    # 等权所用的分组job
    @staticmethod
    def JOB_group_equal_weight(dates, groups, factor, factor_field):
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

    # 回测所用的工具函数
    @staticmethod
    def JOB_backtest(group_name, grouped_weight, benchmark, adj_interval, cash_reserve, price_field, indu_field,
                     data_input, stock_capital, stk_slippage, stk_fee, logger_lvl=logging.INFO):
        weight = grouped_weight.loc[
            (grouped_weight['group'] == group_name) | (grouped_weight['group'] == 'same_group'),
            ['code', 'industry', 'weight_in_industry']].copy()
        BE = IndustryNeutralEngine(stock_capital, stk_slippage, stk_fee, save_name=group_name, logger_lvl=logger_lvl)
        start = weight.index[0].strftime('%Y%m%d')
        end = weight.index[-1].strftime('%Y%m%d')
        portfolio_value = BE.run(weight, start, end, benchmark, adj_interval, cash_reserve, price_field, indu_field,
                                 data_input)
        portfolio_value.rename(columns={'TotalValue': group_name + '_TotalValue'}, inplace=True)
        portfolio_value = pd.DataFrame(portfolio_value[group_name + '_TotalValue'])
        print('%s backtest finish!' % group_name)
        return portfolio_value

    def __init__(self, strategy_name):
        super().__init__(strategy_name)
        self.calendar = self.rdf.get_trading_calendar()

    def init_log(self):
        # 配置log
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=self.logger_lvl)
        folder_dir = global_constant.ROOT_DIR + 'Backtest_Result/Log/{0}/'.format(self.strategy_name)
        if os.path.exists(folder_dir):
            pass
        else:
            os.makedirs(folder_dir.rstrip('/'))
        handler = logging.FileHandler(folder_dir + '{0}to{1}.log'.format(self.start, self.end))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    # days 决定next_period_return 的周期
    def validity_check(self, neutral_factor, mkt_data, T_test=True):
        mkt_next_return = DataProcess.add_next_period_return(mkt_data, self.calendar, self.adj_interval, self.benchmark)
        processed_factor = pd.merge(mkt_next_return, neutral_factor, on=['date', 'code'])
        # 超过0.11或-0.11的return标记为异常数据，置为nan(新股本身剔除)
        processed_factor.dropna(subset=['next_period_return'], inplace=True)
        processed_factor = processed_factor.loc[(processed_factor['next_period_return'] < 0.11 * self.adj_interval) &
                                                (processed_factor['next_period_return'] > -0.11 * self.adj_interval), :]
        dates = processed_factor['date'].unique()
        split_dates = np.array_split(dates, 10)
        if T_test:
            # T检验
            with parallel_backend('multiprocessing', n_jobs=4):
                parallel_res = Parallel()(delayed(FactorTest.JOB_T_test)
                                          (processed_factor, self.factor, dates) for dates in split_dates)
            # 第一行F，第二行T
            RLM_result = np.concatenate(parallel_res, axis=1)
            F_return_values = RLM_result[0]
            T_return_values = RLM_result[1]
            F_alpha_values = RLM_result[2]
            T_alpha_values = RLM_result[3]
            Fret_over_0_pct = F_return_values[F_return_values > 0].shape[0] / F_return_values.shape[0]
            Falp_over_0_pct = F_alpha_values[F_alpha_values > 0].shape[0] / F_alpha_values.shape[0]
            avg_abs_Tret = abs(T_return_values).mean()
            avg_abs_Talp = abs(T_alpha_values).mean()
            abs_Tret_over_2_pct = abs(T_return_values)[abs(T_return_values) >= 2].shape[0] / T_return_values.shape[0]
            abs_Talp_over_2_pct = abs(T_alpha_values)[abs(T_alpha_values) >= 2].shape[0] / T_alpha_values.shape[0]
            self.summary_dict['Fret_over_0_pct'] = Fret_over_0_pct
            self.summary_dict['Falp_over_0_pct'] = Falp_over_0_pct
            self.summary_dict['avg_abs_Tret'] = avg_abs_Tret
            self.summary_dict['avg_abs_Talp'] = avg_abs_Talp
            self.summary_dict['abs_Tret_over_2_pct'] = abs_Tret_over_2_pct
            self.summary_dict['abs_Talp_over_2_pct'] = abs_Talp_over_2_pct
            print('-' * 30)
            print('REGRESSION RESULT: \n   Fret_over_0_pct: %f \n   Falp_over_0_pct: %f \n   avg_abs_Tret: %f \n   '
                  'avg_abs_Talp: %f \n   abs_Tret_over_2_pct: %f \n   abs_Talp_over_2_pct: %f \n'
                  % (Fret_over_0_pct, Falp_over_0_pct, avg_abs_Tret, avg_abs_Talp,
                     abs_Tret_over_2_pct, abs_Talp_over_2_pct))
        # 计算IC
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            parallel_res = Parallel()(delayed(FactorTest.JOB_IC)
                                      (processed_factor, self.factor, dates) for dates in split_dates)
        IC = pd.concat(parallel_res)
        IC_over_0_pct = IC[IC > 0].shape[0] / IC.shape[0]
        abs_IC_over_20pct_pct = abs(IC)[abs(IC) > 0.02].shape[0] / IC.shape[0]
        IR = IC.mean() / IC.std()
        self.summary_dict['IC_mean'] = IC.mean()
        self.summary_dict['IC_std'] = IC.std()
        self.summary_dict['IC_over_0_pct'] = IC_over_0_pct
        self.summary_dict['abs_IC_over_20pct_pct'] = abs_IC_over_20pct_pct
        self.summary_dict['IR'] = IR
        self.summary_dict['ICIR'] = IC.mean() / IR
        print('-' * 30)
        print('ICIR RESULT: \n   IC mean: %f \n   IC std: %f \n   IC_over_0_pct: %f \n   '
              'abs_IC_over_20pct_pct: %f \n   IR: %f \n   ICIR: %f \n' %
              (IC.mean(), IC.std(), IC_over_0_pct, abs_IC_over_20pct_pct, IR, IC.mean() / IR))
        IC.name = 'IC'
        return IC

    # 此处weight_field为行业内权重分配的field
    def group_factor(self, neutral_factor, mkt_data):
        mkt_data = mkt_data.loc[(mkt_data['status'] != '停牌') & (pd.notnull(mkt_data['status'])) &
                                (pd.notnull(mkt_data[self.industry])), ['code', self.industry]]
        idxs = mkt_data.index.unique()
        next_date_dict = {}
        for idx in idxs:
            next_date_dict.update(DataProcess.get_next_date(self.calendar, idx, 1))
        mkt_data['next_1_day'] = mkt_data.apply(lambda row: next_date_dict[row.name], axis=1)
        mkt_data.index.names = ['date']
        mkt_data.reset_index(inplace=True)
        mkt_data.rename(columns={self.industry: 'industry'}, inplace=True)
        # 组合得到当天未停牌因子的code，中性后的因子值，行业
        merge_df = pd.merge(neutral_factor, mkt_data, on=['date', 'code'])
        # 按等权测试
        dates = merge_df['date'].unique()
        split_dates = np.array_split(dates, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            result_list = Parallel()(delayed(FactorTest.JOB_group_equal_weight)
                                     (dates, self.groups, merge_df, self.factor) for dates in split_dates)
        grouped_weight = pd.concat(result_list)
        grouped_weight = grouped_weight.loc[grouped_weight['weight_in_industry'] > 0, :].sort_index()
        str_start = grouped_weight.index[0].strftime('%Y%m%d')
        str_end = grouped_weight.index[-1].strftime('%Y%m%d')
        folder_dir = global_constant.ROOT_DIR + '/Backtest_Result/Factor_Group_Weight/{0}/'.format(self.factor)
        if os.path.exists(folder_dir):
            pass
        else:
            os.makedirs(folder_dir.rstrip('/'))
        grouped_weight.to_csv(folder_dir + 'EquW{0}_{1}to{2}.csv'.format(str(self.groups), str_start, str_end),
                              encoding='gbk')
        return grouped_weight

    def group_backtest(self, grouped_weight):
        group = []
        for i in range(1, self.groups + 1):
            group.append('group_' + str(i))
        with parallel_backend('multiprocessing', n_jobs=self.groups):
            parallel_res = Parallel()(delayed(FactorTest.JOB_backtest)
                                      (g, grouped_weight, self.benchmark, self.adj_interval, self.cash_reserve,
                                       self.price_field, self.industry, self.mkt_data, self.capital,
                                       self.stk_slippage, self.stk_fee, self.logger_lvl) for g in group)
        tot_res = pd.concat(parallel_res, axis=1)
        # 合并指数value
        comparing_index = ['000300.SH', '000905.SH']
        index_value = self.mkt_data.loc[self.mkt_data['code'].isin(comparing_index), ['code', 'close']]
        index_value.set_index([index_value.index, 'code'], inplace=True)
        index_value = index_value.unstack()['close']
        index_value['ZZ800'] = 0.5 * index_value['000300.SH'] + 0.5 * index_value['000905.SH']
        for col in index_value.columns:
            index_value[col] = index_value[col] / index_value[col].iloc[0] * self.capital
        tot_res = pd.merge(tot_res, index_value, left_index=True, right_index=True)
        tot_res['accum_300_alpha'] = (tot_res['group_5_TotalValue'] - tot_res['000300.SH']) / self.capital
        tot_res['accum_500_alpha'] = (tot_res['group_5_TotalValue'] - tot_res['000905.SH']) / self.capital
        tot_res['accum_800_alpha'] = (tot_res['group_5_TotalValue'] - tot_res['ZZ800']) / self.capital
        str_start = tot_res.index[0].strftime('%Y%m%d')
        str_end = tot_res.index[-1].strftime('%Y%m%d')

        folder_dir = global_constant.ROOT_DIR + '/Backtest_Result/Group_Value/{0}/'.format(self.factor)
        if os.path.exists(folder_dir):
            pass
        else:
            os.makedirs(folder_dir.rstrip('/'))
        tot_res.to_csv(folder_dir + 'Value_{0}to{1}.csv'.format(str_start, str_end), encoding='gbk')
        # -------------------------------------
        self.summary_dict['AnnRet'] = \
            DataProcess.calc_ann_return(tot_res[group[-1] + '_TotalValue'])
        self.summary_dict['alpha_300'] = \
            DataProcess.calc_alpha_ann_return(tot_res[group[-1] + '_TotalValue'], tot_res['000300.SH'])
        self.summary_dict['alpha_500'] = \
            DataProcess.calc_alpha_ann_return(tot_res[group[-1] + '_TotalValue'], tot_res['000905.SH'])
        self.summary_dict['alpha_800'] = \
            DataProcess.calc_alpha_ann_return(tot_res[group[-1] + '_TotalValue'], tot_res['ZZ800'])
        self.summary_dict['MDD'] = \
            DataProcess.calc_max_draw_down(tot_res[group[-1] + '_TotalValue'])
        self.summary_dict['sharpe_ratio'] = \
            DataProcess.calc_sharpe(tot_res[group[-1] + '_TotalValue'])
        self.summary_dict['sharpe_alpha_300'] = \
            DataProcess.calc_alpha_sharpe(tot_res[group[-1] + '_TotalValue'], tot_res['000300.SH'])
        self.summary_dict['sharpe_alpha_500'] = \
            DataProcess.calc_alpha_sharpe(tot_res[group[-1] + '_TotalValue'], tot_res['000905.SH'])
        self.summary_dict['sharpe_alpha_800'] = \
            DataProcess.calc_alpha_sharpe(tot_res[group[-1] + '_TotalValue'], tot_res['ZZ800'])
        print('-' * 30)
        # -------------------------------------
        self.summary_dict['Start_Time'] = tot_res.index[0].strftime('%Y%m%d')
        self.summary_dict['End_Time'] = tot_res.index[-1].strftime('%Y%m%d')
        return self.summary_dict.copy()

    def generate_report(self, summary_dict):
        rep = {self.factor: summary_dict}
        rep = pd.DataFrame(rep)
        rep = rep.reindex(
            index=['Start_Time', 'End_Time', 'IC_mean', 'IC_std', 'IR', 'ICIR', 'AnnRet', 'alpha_300', 'alpha_500',
                   'alpha_800', 'MDD', 'sharpe_ratio', 'sharpe_alpha_300', 'sharpe_alpha_500', 'sharpe_alpha_800',
                   'abs_Talp_over_2_pct', 'IC_over_0_pct', 'abs_IC_over_20pct_pct', 'Fret_over_0_pct',
                   'Falp_over_0_pct', 'avg_abs_Tret', 'avg_abs_Talp', 'abs_Tret_over_2_pct']).T
        str_start = rep['Start_Time'].iloc[0]
        str_end = rep['End_Time'].iloc[0]
        folder_dir = global_constant.ROOT_DIR + '/Backtest_Result/Factor_Report/{0}/'.format(self.factor)
        if os.path.exists(folder_dir):
            pass
        else:
            os.makedirs(folder_dir.rstrip('/'))
        rep.to_csv(folder_dir + '{0}to{1}.csv'.format(str_start, str_end), encoding='gbk')

    def run_factor_test(
            # 初始化参数
            self, start, end, benchmark, select_range, industry, size_field,
            # 因子参数
            measure, factor, direction, if_fillna,
            # 回测参数
            adj_interval=5, groups=5, capital=5000000, cash_reserve=0.03, stk_slippage=0.001,
            stk_fee=0.0001, price_field='vwap', logger_lvl=logging.ERROR):

        self.factor = factor
        self.groups = groups
        self.industry = industry
        self.capital = capital
        self.cash_reserve = cash_reserve
        self.stk_slippage = stk_slippage
        self.stk_fee = stk_fee
        self.price_field = price_field
        self.adj_interval = adj_interval
        self.logger_lvl = logger_lvl
        self.summary_dict = {}
        # ---------------------------------------------------------------
        # 策略初始化
        self.initialize_strategy(start, end, benchmark, select_range, industry, size_field)
        print('initialization finish')
        print('-' * 30)
        # 因子数据
        self.factor_data = self.process_factor(measure, factor, direction, if_fillna)
        # 有效性检验
        self.validity_check(self.factor_data, self.mkt_data, T_test=True)
        print('validity checking finish')
        print('-' * 30)
        # 分组
        grouped_weight = self.group_factor(self.factor_data, self.mkt_data)
        print('factor grouping finish')
        print('-' * 30)
        # 回测
        summary_dict = self.group_backtest(grouped_weight)
        print('group backtest finish')
        print('-' * 30)
        # 生成报告
        self.generate_report(summary_dict)
        print('report got')


if __name__ == '__main__':
    dt_start = datetime.datetime.now()
    warnings.filterwarnings("ignore")

    start = 20120101
    end = 20171231
    measurements = ['Analyst', 'Analyst']
    factors = ['TPER', 'PEG']
    directions = [1, -1]
    if_fillnas = [False, False]
    benchmark = 300
    select_range = 800
    industry = 'improved_lv1'
    size_field = 'ln_market_cap'

    for i in range(len(factors)):
        factor = factors[i]
        measurement = measurements[i]
        direction = directions[i]
        if_fillna = if_fillnas[i]
        test = FactorTest(factor)
        test.run_factor_test(start, end, benchmark, select_range, industry, size_field, measurement, factor, direction,
                             if_fillna)
    print('Test finish! Time token: ', datetime.datetime.now()-dt_start)