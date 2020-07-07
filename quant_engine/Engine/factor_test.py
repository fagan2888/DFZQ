import global_constant
import pandas as pd
import numpy as np
import os.path
from data_process import DataProcess
from joblib import Parallel, delayed, parallel_backend
import statsmodels.api as sm
from backtest_engine import BacktestEngine
import logging
import datetime
from factor_test_CONFIG import TEST_CONFIG, CATEGORY_WEIGHT, FACTOR_WEIGHT
from strategy_base import StrategyBase
from dateutil.relativedelta import relativedelta


class FactorTest(StrategyBase):
    @staticmethod
    def JOB_T_test(processed_factor, factor_field, dates):
        F_alpha = []
        T_alpha = []
        for date in dates:
            day_factor = processed_factor.loc[processed_factor['date'] == date, factor_field].values
            day_alpha = processed_factor.loc[processed_factor['date'] == date, 'alpha'].values
            RLM_est = sm.RLM(day_alpha, day_factor, M=sm.robust.norms.HuberT()).fit()
            day_RLM_para = RLM_est.params
            day_Tvalue = RLM_est.tvalues
            F_alpha.append(day_RLM_para[0])
            T_alpha.append(day_Tvalue[0])
        return np.array([F_alpha, T_alpha])

    @staticmethod
    def JOB_IC(processed_factor, factor_field, dates):
        day_IC = []
        IC_date = []
        for date in dates:
            day_factor = processed_factor.loc[processed_factor['date'] == date, factor_field]
            day_return = processed_factor.loc[processed_factor['date'] == date, 'alpha']
            day_IC.append(day_factor.corr(day_return, method='spearman'))
            IC_date.append(date)
        return pd.Series(day_IC, index=IC_date)

    # 等权分配所用的分组job
    @staticmethod
    def JOB_group_factor(dates, groups, factor, factor_field, size_field):
        labels = []
        for i in range(1, groups + 1):
            labels.append('group_' + str(i))
        res = []
        for date in dates:
            day_factor = factor.loc[date, :].copy()
            industries = day_factor['industry'].unique()
            for ind in industries:
                day_industry_factor = day_factor.loc[day_factor['industry'] == ind, :].copy()
                # 行业成分不足10支票时，所有group配置一样
                if day_industry_factor.shape[0] < 10:
                    day_industry_factor['group'] = 'same group'
                    day_industry_factor['weight'] = \
                        day_industry_factor['industry_weight'] / day_industry_factor[size_field].sum() * \
                        day_industry_factor[size_field]
                else:
                    day_industry_factor['group'] = pd.qcut(day_industry_factor[factor_field], 5, labels=labels)
                    group_size = day_industry_factor.groupby('group')[size_field].sum().to_dict()
                    day_industry_factor['group_size'] = day_industry_factor['group'].map(group_size)
                    day_industry_factor['weight'] = day_industry_factor['industry_weight'] / \
                                                    day_industry_factor['group_size'] * day_industry_factor[size_field]
                res.append(day_industry_factor)
        res_df = pd.concat(res)
        return res_df

    # 回测所用的工具函数
    @staticmethod
    def JOB_backtest(backtest_engine, group_name, grouped_weight):
        weight = grouped_weight.loc[
            (grouped_weight['group'] == group_name) | (grouped_weight['group'] == 'same_group'),
            ['code', 'weight']].copy()
        start = weight.index[0].strftime('%Y%m%d')
        end = weight.index[-1].strftime('%Y%m%d')
        portfolio_value = backtest_engine.run(weight, start, end)
        portfolio_value = portfolio_value.loc[:, ['TotalValue', 'AccumAlpha']]
        portfolio_value.rename(columns={'TotalValue': group_name + '_TotalValue',
                                        'AccumAlpha': group_name + '_AccumAlpha'}, inplace=True)
        print('%s backtest finish!' % group_name)
        return portfolio_value

    # 重写 init_log
    def init_log(self):
        # 配置log
        self.logger = logging.getLogger('FactorTest')
        self.logger.setLevel(level=self.logger_lvl)
        handler = logging.FileHandler(self.folder_dir + 'Test_result.log')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    # 重写 initialize_strategy
    def initialize_strategy(self):
        self.n_jobs = global_constant.N_STRATEGY
        self.start = TEST_CONFIG['start']
        self.end = TEST_CONFIG['end']
        self.benchmark = TEST_CONFIG['benchmark']
        self.select_range = TEST_CONFIG['select_range']
        self.industry = TEST_CONFIG['industry']
        self.adj_interval = TEST_CONFIG['adj_interval']
        mkt_end = (pd.to_datetime(str(self.end)) + relativedelta(months=1)).strftime('%Y%m%d')
        self.data_prepare(self.start, mkt_end)
        # 获取 benchmark 的 行业权重
        self.bm_indu_weight = \
            pd.merge(self.bm_stk_wgt.reset_index(), self.industry_data.reset_index(), on=['date', 'code'])
        self.bm_indu_weight = self.bm_indu_weight.groupby(['date', 'industry'])['weight'].sum()
        self.bm_indu_weight = self.bm_indu_weight.reset_index().set_index('date')
        # 获取 benchmark 的 行情
        self.bm_mkt = self.mkt_data.loc[self.mkt_data['code'] == self.benchmark_code, :].copy()
        # 获取 流通市值
        self.float_mv = self.influx.getDataMultiprocess(global_constant.FACTOR_DB, 'Size', self.start, self.end,
                                                        ['code', 'float_market_cap'])
        self.float_mv.index.names = ['date']
        self.init_log()

    # 重写 __init__
    def __init__(self):
        # set save name
        self.save_name = ''
        for cate in CATEGORY_WEIGHT:
            for fct in FACTOR_WEIGHT[cate]:
                if self.save_name:
                    self.save_name = self.save_name + '+'
                self.save_name = self.save_name + '{0}({1})'.format(fct[2], str(fct[5]))
        self.folder_dir = global_constant.ROOT_DIR + 'Factor_Test/{0}/'.format(self.save_name)
        if os.path.exists(self.folder_dir):
            pass
        else:
            os.makedirs(self.folder_dir.rstrip('/'))
        super().__init__(self.save_name)

    def factors_combination(self):
        self.logger.info('COMBINE PARAMETERS: ')
        categories = []
        for category in CATEGORY_WEIGHT.keys():
            self.logger.info('-%s  weight: %i' % (category, CATEGORY_WEIGHT[category]))
            parameters_list = FACTOR_WEIGHT[category]
            factors_in_category = []
            for db, measure, factor, direction, fillna, weight, check_rp, neutralize in parameters_list:
                self.logger.info('  -Factor: %s   DIR:%i,   FILLNA:%s,   CHECKRP:%s,   NEUTRALIZE:%s' %
                                 (factor, direction, str(fillna), str(check_rp), str(neutralize)))
                self.logger.info('  -Factor: %s  weight:%i' % (factor, weight))
                factor_df = self.process_factor(db, measure, factor, direction, fillna, check_rp, neutralize)
                factor_df[factor] = factor_df[factor] * weight
                factor_df.set_index(['date', 'code'], inplace=True)
                factors_in_category.append(factor_df)
            category_df = pd.concat(factors_in_category, join='inner', axis=1)
            category_df[category] = category_df.sum(axis=1)
            category_df = category_df.reset_index()
            category_df = DataProcess.standardize(category_df, category, False, self.n_jobs)
            category_df[category] = CATEGORY_WEIGHT[category] * category_df[category]
            category_df.set_index(['date', 'code'], inplace=True)
            categories.append(category_df)
        merged_df = pd.concat(categories, join='inner', axis=1)
        merged_df['overall'] = merged_df[list(CATEGORY_WEIGHT.keys())].sum(axis=1)
        merged_df = merged_df.reset_index()
        merged_df = pd.merge(merged_df, self.industry_data.reset_index(), how='inner', on=['date', 'code'])
        merged_df = pd.merge(merged_df, self.float_mv.reset_index(), how='inner', on=['date', 'code'])
        merged_df = merged_df.set_index('date')
        folder_dir = global_constant.ROOT_DIR + 'Factor_Test/{0}/'.format(self.save_name)
        merged_df.to_csv(folder_dir + 'FactorsCombination.csv', encoding='gbk')
        print('Factors combination finish...')
        return merged_df

    def get_alpha_df(self):
        next_date_dict = {}
        for date in self.mkt_data.index.unique().strftime('%Y%m%d')[:-self.adj_interval]:
            next_date_dict.update(DataProcess.get_next_date(self.calendar, date, self.adj_interval))
        # get stk return
        stk_ret_df = self.mkt_data.copy()
        stk_ret_df['next_period_date'] = stk_ret_df.index.strftime('%Y%m%d')
        stk_ret_df['next_period_date'] = stk_ret_df['next_period_date'].map(next_date_dict)
        stk_ret_df = stk_ret_df.dropna(subset=['next_period_date'])
        stk_ret_df['next_period_date'] = pd.to_datetime(stk_ret_df['next_period_date'])
        stk_ret_df['fq_close'] = stk_ret_df['adj_factor'] * stk_ret_df['close']
        stk_ret_df.reset_index(inplace=True)
        stk_ret_df = stk_ret_df.loc[:, ['date', 'code', 'status', 'fq_close', 'next_period_date']]
        next_mkt_data = stk_ret_df.copy()
        next_mkt_data = next_mkt_data.loc[:, ['date', 'code', 'fq_close']]
        next_mkt_data.rename(columns={'date': 'next_period_date', 'fq_close': 'next_fq_close'}, inplace=True)
        stk_ret_df = pd.merge(stk_ret_df, next_mkt_data, on=['next_period_date', 'code'])
        stk_ret_df['return'] = stk_ret_df['next_fq_close'] / stk_ret_df['fq_close'] - 1
        stk_ret_df = stk_ret_df.loc[(stk_ret_df['return'] < 1.1 ** self.adj_interval - 1) &
                                    (stk_ret_df['return'] > 0.9 ** self.adj_interval - 1), :]
        stk_ret_df = stk_ret_df.loc[:, ['date', 'code', 'status', 'return']]
        # get benchmark return
        bm_ret_df = self.bm_mkt.copy()
        bm_ret_df['next_period_date'] = bm_ret_df.index.strftime('%Y%m%d')
        bm_ret_df['next_period_date'] = bm_ret_df['next_period_date'].map(next_date_dict)
        bm_ret_df = bm_ret_df.dropna(subset=['next_period_date'])
        bm_ret_df['next_period_date'] = pd.to_datetime(bm_ret_df['next_period_date'])
        bm_ret_df['fq_close'] = bm_ret_df['adj_factor'] * bm_ret_df['close']
        bm_ret_df.reset_index(inplace=True)
        bm_ret_df = bm_ret_df.loc[:, ['date', 'fq_close', 'next_period_date']]
        next_bm_data = bm_ret_df.copy()
        next_bm_data = next_bm_data.loc[:, ['date', 'fq_close']]
        next_bm_data.rename(columns={'date': 'next_period_date', 'fq_close': 'next_fq_close'}, inplace=True)
        bm_ret_df = pd.merge(bm_ret_df, next_bm_data, on=['next_period_date'])
        bm_ret_df['idx_return'] = bm_ret_df['next_fq_close'] / bm_ret_df['fq_close'] - 1
        bm_ret_df = bm_ret_df.loc[:, ['date', 'idx_return']]
        alpha_df = pd.merge(stk_ret_df, bm_ret_df, on=['date'])
        alpha_df['alpha'] = alpha_df['return'] - alpha_df['idx_return']
        alpha_df.set_index('date', inplace=True)
        alpha_df = alpha_df.loc[:, ['code', 'status', 'alpha']]
        return alpha_df

    # days 决定next_period_return 的周期
    def validity_check(self, T_test=True):
        fct_data = self.factor_data.loc[:, ['code', 'overall']].copy()
        factor_alpha = pd.merge(fct_data.reset_index(), self.alpha_df.reset_index(), on=['date', 'code'])
        factor_alpha = factor_alpha.loc[factor_alpha['status'] != '停牌', :]
        dates = factor_alpha['date'].unique()
        split_dates = np.array_split(dates, self.n_jobs)
        if T_test:
            # T检验
            with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
                parallel_res = Parallel()(delayed(FactorTest.JOB_T_test)
                                          (factor_alpha, 'overall', dates) for dates in split_dates)
            # 第一行F，第二行T
            RLM_result = np.concatenate(parallel_res, axis=1)
            F_alpha_values = RLM_result[0]
            T_alpha_values = RLM_result[1]
            Falp_over_0_pct = F_alpha_values[F_alpha_values > 0].shape[0] / F_alpha_values.shape[0]
            avg_abs_Talp = abs(T_alpha_values).mean()
            abs_Talp_over_2_pct = abs(T_alpha_values)[abs(T_alpha_values) >= 2].shape[0] / T_alpha_values.shape[0]
            self.summary_dict['Falp_over_0_pct'] = Falp_over_0_pct
            self.summary_dict['avg_abs_Talp'] = avg_abs_Talp
            self.summary_dict['abs_Talp_over_2_pct'] = abs_Talp_over_2_pct
            print('-' * 30)
            print('REGRESSION RESULT: \n -Falp_over_0_pct: %f \n -avg_abs_Talp: %f \n -abs_Talp_over_2_pct: %f \n'
                  % (Falp_over_0_pct, avg_abs_Talp, abs_Talp_over_2_pct))
            # 计算IC
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            parallel_res = Parallel()(delayed(FactorTest.JOB_IC)
                                      (factor_alpha, 'overall', dates) for dates in split_dates)
        IC = pd.concat(parallel_res)
        IC_over_0_pct = IC[IC > 0].shape[0] / IC.shape[0]
        abs_IC_over_20pct_pct = abs(IC)[abs(IC) > 0.02].shape[0] / IC.shape[0]
        IR = IC.mean() / IC.std()
        ICIR = IC.mean() / IC.std() * np.sqrt(250 / self.adj_interval)
        self.summary_dict['IC_mean'] = IC.mean()
        self.summary_dict['IC_std'] = IC.std()
        self.summary_dict['IC_over_0_pct'] = IC_over_0_pct
        self.summary_dict['abs_IC_over_20pct_pct'] = abs_IC_over_20pct_pct
        self.summary_dict['IR'] = IR
        self.summary_dict['ICIR'] = ICIR
        print('-' * 30)
        print('ICIR RESULT: \n   IC mean: %f \n   IC std: %f \n   IC_over_0_pct: %f \n   '
              'abs_IC_over_20pct_pct: %f \n   IR: %f \n   ICIR: %f \n' %
              (IC.mean(), IC.std(), IC_over_0_pct, abs_IC_over_20pct_pct, IR, ICIR))
        IC.name = 'IC'
        return IC

    # 此处weight_field为行业内权重分配的field
    def group_factor(self):
        # 组合行业权重
        fct_data = self.factor_data.loc[:, ['code', 'industry', 'overall', 'float_market_cap']]
        next_date_dict = {}
        for date in fct_data.index.unique().strftime('%Y%m%d')[:-1]:
            next_date_dict.update(DataProcess.get_next_date(self.calendar, date, 1))
        fct_data['date'] = fct_data.index.strftime('%Y%m%d')
        fct_data['date'] = fct_data['date'].map(next_date_dict)
        fct_data = fct_data.dropna(subset=['date'])
        fct_data['date'] = pd.to_datetime(fct_data['date'])
        # 以前一天的因子值，市值 组合后一天的行业权重
        merge_df = pd.merge(fct_data, self.bm_indu_weight.reset_index(), how='inner', on=['date', 'industry'])
        merge_df.rename(columns={'weight': 'industry_weight'}, inplace=True)
        merge_df.set_index('date', inplace=True)
        # 按市值加权测试
        dates = merge_df.index.unique().strftime('%Y%m%d')
        split_dates = np.array_split(dates, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            result_list = Parallel()(delayed(FactorTest.JOB_group_factor)
                                     (dates, self.groups, merge_df, 'overall', 'float_market_cap')
                                     for dates in split_dates)
        grouped_weight = pd.concat(result_list)
        grouped_weight = grouped_weight.loc[grouped_weight['weight'] > 0, :].sort_index()
        str_start = grouped_weight.index[0].strftime('%Y%m%d')
        str_end = grouped_weight.index[-1].strftime('%Y%m%d')
        folder_dir = global_constant.ROOT_DIR + '/Factor_Test/{0}/'.format(self.save_name)
        grouped_weight.to_csv(folder_dir + 'GrpWgt_{0}to{1}.csv'.format(str_start, str_end), encoding='gbk')
        return grouped_weight

    def group_backtest(self, grouped_weight):
        group = []
        BE = BacktestEngine('FactorTest_{0}'.format(self.save_name), self.start, self.end, self.adj_interval,
                            self.benchmark, stock_capital=self.capital, logger_lvl=logging.DEBUG)
        for i in range(1, self.groups + 1):
            group.append('group_' + str(i))
        start = grouped_weight.index.unique().strftime('%Y%m%d')[0]
        end = grouped_weight.index.unique().strftime('%Y%m%d')[-1]
        pvs = []
        for g in group:
            weight = grouped_weight.loc[
                (grouped_weight['group'] == g) | (grouped_weight['group'] == 'same_group'),
                ['code', 'weight']].copy()
            portfolio_value, _ = BE.run(weight, start, end)
            portfolio_value = portfolio_value.loc[:, ['TotalValue', 'BenchmarkValue', 'AccumAlpha']]
            portfolio_value.rename(columns={'TotalValue': 'TotalValue_{0}'.format(g),
                                            'BenchmarkValue': 'BenchmarkValue_{0}'.format(g),
                                            'AccumAlpha': 'AccumAlpha_{0}'.format(g)}, inplace=True)
            pvs.append(portfolio_value)
            # 重置 stk_portfolio
            BE.stk_portfolio.reset_portfolio(self.capital)
        tot_res = pd.concat(pvs, axis=1)
        tot_res['long_short'] = tot_res['TotalValue_{0}'.format(group[-1])].pct_change().fillna(0) - \
                                tot_res['TotalValue_{0}'.format(group[0])].pct_change().fillna(0)
        tot_res['long_short'] = (tot_res['long_short'] + 1).cumprod()
        folder_dir = global_constant.ROOT_DIR + '/Factor_Test/{0}/'.format(self.save_name)
        tot_res.to_csv(folder_dir + 'Value_{0}to{1}.csv'.format(self.start, self.end), encoding='gbk')
        print('-' * 30)
        # -------------------------------------
        self.summary_dict['Start_Time'] = start
        self.summary_dict['End_Time'] = end
        self.summary_dict['AnnAlpha'] = DataProcess.calc_alpha_ann_return(
            tot_res['TotalValue_{0}'.format(group[-1])], tot_res['BenchmarkValue_{0}'.format(group[-1])])
        self.summary_dict['AlphaMDD'] = DataProcess.calc_alpha_max_draw_down(
            tot_res['TotalValue_{0}'.format(group[-1])], tot_res['BenchmarkValue_{0}'.format(group[-1])])
        self.summary_dict['AlphaSharpe'] = DataProcess.calc_alpha_sharpe(
            tot_res['TotalValue_{0}'.format(group[-1])], tot_res['BenchmarkValue_{0}'.format(group[-1])])
        return self.summary_dict.copy()

    def generate_report(self):
        fields = ['Start_Time', 'End_Time', 'IC_mean', 'IC_std', 'IR', 'ICIR', 'AnnAlpha', 'AlphaMDD', 'AlphaSharpe',
                  'IC_over_0_pct', 'abs_IC_over_20pct_pct', 'Falp_over_0_pct', 'avg_abs_Talp', 'abs_Talp_over_2_pct']
        self.logger.info('TEST RESULT:')
        for field in fields:
            self.logger.info('{0}:   {1}'.format(field, self.summary_dict[field]))
        self.logger.info('*' * 30)

    def run(self, adj_interval=5, groups=5, capital=5000000, cash_reserve=0.03, stk_slippage=0.001,
            stk_fee=0.0001, price_field='vwap', logger_lvl=logging.INFO):
        self.groups = groups
        self.capital = capital
        self.cash_reserve = cash_reserve
        self.stk_slippage = stk_slippage
        self.stk_fee = stk_fee
        self.price_field = price_field
        self.adj_interval = adj_interval
        self.logger_lvl = logger_lvl
        self.summary_dict = {}
        # ---------------------------------------------------------------
        # initialize strategy
        self.initialize_strategy()
        # alpha 数据
        self.alpha_df = self.get_alpha_df()
        # 因子数据
        self.factor_data = self.factors_combination()
        # 有效性检验
        self.validity_check(T_test=True)
        print('validity checking finish')
        print('-' * 30)
        # 分组
        grouped_weight = self.group_factor()
        print('factor grouping finish')
        print('-' * 30)
        # 回测
        self.group_backtest(grouped_weight)
        print('group backtest finish')
        print('-' * 30)
        # 生成报告
        print(self.summary_dict)
        self.generate_report()
        print('report got')


if __name__ == '__main__':
    dt_start = datetime.datetime.now()
    ft = FactorTest()
    ft.run()
    print('Test finish! Time token: ', datetime.datetime.now() - dt_start)