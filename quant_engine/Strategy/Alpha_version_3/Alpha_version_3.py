# 相比version_2 加入了 surprise因子 和 analyst因子

from strategy_base import StrategyBase
from Alpha_version_3_CONFIG import STRATEGY_CONFIG, CATEGORY_WEIGHT, FACTOR_WEIGHT
from global_constant import N_JOBS
import pandas as pd
import numpy as np
from data_process import DataProcess
from joblib import Parallel, delayed, parallel_backend
import datetime
import cvxpy as cp


class alpha_version_3(StrategyBase):
    def __init__(self, strategy_name):
        super().__init__(strategy_name)

    def initialize_strategy(self):
        start = STRATEGY_CONFIG['start']
        end = STRATEGY_CONFIG['end']
        benchmark = STRATEGY_CONFIG['benchmark']
        select_range = STRATEGY_CONFIG['select_range']
        industry = STRATEGY_CONFIG['industry']
        size_field = STRATEGY_CONFIG['size_field']
        super().initialize_strategy(start, end, benchmark, select_range, industry, size_field)
        self.capital = STRATEGY_CONFIG['capital']
        self.adj_interval = STRATEGY_CONFIG['adj_interval']
        self.opt_option = STRATEGY_CONFIG['opt_option']
        self.target_sigma = STRATEGY_CONFIG['target_sigma']
        self.risk_aversion = STRATEGY_CONFIG['risk_aversion']
        self.mv_max_exp = STRATEGY_CONFIG['mv_max_exp']
        self.mv_min_exp = STRATEGY_CONFIG['mv_min_exp']

    def get_factors(self, measure, factor, direction, if_fillna, weight):
        print('-Factor: %s is processing...' % factor)
        factor_df = self.process_factor(measure, factor, direction, if_fillna)
        if weight == 1:
            pass
        else:
            factor_df[factor] = factor_df[factor] * weight
        factor_df.set_index(['date', 'code'], inplace=True)
        return factor_df

    def factors_combination(self):
        categories = []
        for category in FACTOR_WEIGHT.keys():
            print('Category: %s is processing...' % category)
            parameters_list = FACTOR_WEIGHT[category]
            factors_in_category = []
            for measure, factor, direction, if_fillna, weight in parameters_list:
                factor_df = self.get_factors(measure, factor, direction, if_fillna, weight)
                factors_in_category.append(factor_df)
            category_df = pd.concat(factors_in_category, join='inner', axis=1)
            category_df[category] = category_df.sum(axis=1)
            category_df = category_df.reset_index().loc[:, ['date', 'code', category]]
            category_df = DataProcess.remove_and_Z(category_df, category, False, self.n_jobs)
            category_df[category] = CATEGORY_WEIGHT[category] * category_df[category]
            category_df.set_index(['date', 'code'], inplace=True)
            categories.append(category_df)
        merged_df = pd.concat(categories, join='inner', axis=1)
        merged_df['overall'] = merged_df.sum(axis=1)
        merged_df = merged_df.reset_index().loc[:, ['date', 'code', 'overall']].set_index('date')
        print('Factors combination finish...')
        return merged_df

    @staticmethod
    def opti_weight(factor, dates, adj_interval, opt_option, target_sigma, risk_aversion, mv_max_exp, mv_min_exp,
                    bm_stk_wgt, bm_indu_wgt, indu_dummies, z_size, risk_exp, risk_cov, spec_risk):
        for date in dates:
            day_factor = factor.loc[date, :].sort_values('code').copy()
            codes = day_factor['code'].values
            array_overall_exp = day_factor['overall'].values
            # -------------------------权重设置--------------------------
            # 设置基准权重
            day_bm_stk_wgt = bm_stk_wgt.loc[date, :].sort_values('code').copy()
            base_weight = pd.merge(day_factor.reset_index(), day_bm_stk_wgt.reset_index(),
                                   how='left', on=['date', 'code'])
            base_weight['weight'] = base_weight['weight'].fillna(0)
            array_base_weight = base_weight['weight'].values
            # 设置权重上下限
            upbound = base_weight['weight'].copy()
            #   基准权重大于1的话 可以的翻倍
            upbound[upbound > 1] = upbound[upbound > 1]
            #   基准权重小于1的话 只能到2
            upbound[upbound <= 1] = 2 - upbound[upbound <= 1]
            array_upbound = np.array(upbound)
            #   权重最少调整到0
            lowbound = np.array(-1 * base_weight['weight'])
            array_lowbound = np.array(lowbound)
            # -----------------------哑变量设置-------------------------
            day_indu_dummies = indu_dummies.loc[date, :].set_index('code')
            array_indu_dummies = day_indu_dummies.loc[codes, :].values
            indus = day_indu_dummies.columns
            indu_nums = len(indus)
            day_bm_indu_wgt = bm_indu_wgt.loc[date, :].set_index('industry')
            miss_indus = indus.difference(day_bm_indu_wgt.index)
            for miss_i in miss_indus:
                day_bm_indu_wgt.loc[miss_i] = 0
            array_bm_indu_wgt = day_bm_indu_wgt.loc[indus, 'weight'].values
            # ----------------------市值因子设置------------------------
            day_z_size = z_size.loc[date, :].set_index('code')
            array_z_size = day_z_size.loc[codes, 'size'].values
            # ----------------------跟踪误差设置------------------------
            day_risk_exp = risk_exp.loc[date, :].set_index('code')
            risk_factors = day_risk_exp.columns
            array_risk_exp = day_risk_exp.loc[codes, :].values
            day_risk_cov = risk_cov.loc[date, :].set_index('code')
            array_risk_cov = day_risk_cov.loc[risk_factors, risk_factors].values
            array_risk_cov = 0.5 * (array_risk_cov + array_risk_cov.T)
            day_spec_risk = spec_risk.loc[date, :].set_index('code')
            array_spec_risk = day_spec_risk.loc[codes, 'specific_risk'].values
            # ----------------------风险因子设置------------------------
            '''
            # 除 Market和Size 外的10个风险因子
            risk_factors = ['Trend', 'Beta', 'Volatility', 'Liquidity', 'Value',
                            'Growth', 'SOE', 'Uncertainty', 'Cubic size']
            '''
            # -----------------------变量设置---------------------------
            stk_num, risk_factors_num = array_risk_exp.shape
            # 相对权重
            solve_weight = cp.Variable(stk_num)
            tot_risk_exp = array_risk_exp.T * (solve_weight + array_base_weight)
            risk_variance = cp.quad_form(tot_risk_exp, array_risk_cov) + \
                            cp.sum_squares(cp.multiply(array_spec_risk, (solve_weight + array_base_weight)))
            overall_exp = array_overall_exp * solve_weight
            # -----------------目标函数优化约束条件----------------------
            cons = []
            if opt_option == 1:
                obj = overall_exp
                sigma = target_sigma / np.sqrt(250/adj_interval)
                cons.append(risk_variance <= sigma ** 2)
            else:
                obj = overall_exp - risk_aversion * risk_variance
            cons.append(solve_weight <= array_upbound)
            cons.append(solve_weight >= array_lowbound)
            # 行业中性
            for i in range(indu_nums):
                wgt_diff = array_bm_indu_wgt[i] - (array_indu_dummies[:, i].T * array_base_weight).sum()
                if wgt_diff < 0:
                    wgt_diff = 0
                #cons.append(cp.sum(array_indu_dummies[:, i].T * solve_weight) == wgt_diff)
                cons.append(cp.sum(array_indu_dummies[:, i].T * solve_weight) == 0)
            # 市值主动暴露
            cons.append(cp.sum(array_z_size * solve_weight) <= mv_max_exp)
            cons.append(cp.sum(array_z_size * solve_weight) >= mv_min_exp)
            # 优化
            prob = cp.Problem(cp.Maximize(obj), constraints=cons)
            argskw = {'mi_max_iters': 1000, 'feastol': 1e-3, 'abstol': 1e-3}
            try:
                prob.solve(solver='ECOS', **argskw)
            except:
                prob.solve(solver='SCS', **argskw)
            if prob.status in ('optimal', 'optimal_inaccurate'):
                opti_weight = np.array(solve_weight.value)
            else:
                print(prob.status)
                raise ValueError('优化失败')
            print(opti_weight)


    def run(self):
        self.initialize_strategy()
        overall_factor = self.factors_combination()
        dates = overall_factor.index.unique().strftime("%Y%m%d")
        alpha_version_3.opti_weight(
            overall_factor, dates, self.adj_interval, self.opt_option, self.target_sigma, self.risk_aversion,
            self.mv_max_exp, self.mv_min_exp, self.bm_stk_wgt, self.bm_indu_wgt, self.industry_dummies,
            self.z_size, self.risk_exp, self.risk_cov, self.spec_risk)


        print('.')
        '''
        engine = IndustryNeutralEngine(stock_capital=self.capital, save_name=self.strategy_name)
        bt_start = weight.index[0].strftime('%Y%m%d')
        bt_end = (weight.index[-1] - datetime.timedelta(days=1)).strftime('%Y%m%d')
        portfolio_value = engine.run(weight, bt_start, bt_end, benchmark='IF', adj_interval=5, data_input=self.mkt_data)
        portfolio_value = portfolio_value.loc[:, ['Balance', 'StockValue', 'TotalValue']]
        comparing_index = ['000300.SH']
        index_value = self.mkt_data.loc[self.mkt_data['code'].isin(comparing_index), ['code', 'close']]
        index_value.set_index([index_value.index, 'code'], inplace=True)
        index_value = index_value.unstack()['close']
        for col in index_value.columns:
            index_value[col] = index_value[col] / index_value[col].iloc[0] * self.capital
        portfolio_value = pd.merge(portfolio_value, index_value, left_index=True, right_index=True)
        portfolio_value['accum_alpha'] = \
            DataProcess.calc_accum_alpha(portfolio_value['TotalValue'], portfolio_value['000300.SH'])
        portfolio_value.to_csv(self.strategy_name + '_BACKTEST.csv', encoding='gbk')

        self.logger.info('Backtest finish time: %s' % datetime.datetime.now().strftime('%Y/%m/%d - %H:%M:%S'))
        self.logger.info('*' * 50)
        self.logger.info('PERFORMANCE:')
        self.logger.info('-ANN_return: %f' % DataProcess.calc_ann_return(portfolio_value['TotalValue']))
        self.logger.info('-MDD: %f' % DataProcess.calc_max_draw_down(portfolio_value['TotalValue']))
        self.logger.info('-sharpe: %f' % DataProcess.calc_sharpe(portfolio_value['TotalValue']))
        self.logger.info('-ANN_Alpha: %f' % DataProcess.calc_alpha_ann_return(
            portfolio_value['TotalValue'], portfolio_value['000300.SH']))
        self.logger.info('-Alpha_MDD: %f' % DataProcess.calc_alpha_max_draw_down(
            portfolio_value['TotalValue'], portfolio_value['000300.SH']))
        self.logger.info('-Alpha_sharpe: %f' % DataProcess.calc_alpha_sharpe(
            portfolio_value['TotalValue'], portfolio_value['000300.SH']))
        '''

if __name__ == '__main__':
    print(datetime.datetime.now())
    a = alpha_version_3('Alpha_version_3')
    kk = a.run()
    print(datetime.datetime.now())