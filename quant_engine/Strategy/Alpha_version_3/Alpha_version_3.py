# 相比version_2 加入了 surprise因子 和 analyst因子
# 使用优化器

from strategy_base import StrategyBase
from Alpha_version_3_CONFIG import STRATEGY_CONFIG, CATEGORY_WEIGHT, FACTOR_WEIGHT
import pandas as pd
import numpy as np
from data_process import DataProcess
from joblib import Parallel, delayed, parallel_backend
import datetime
import cvxpy as cp
from backtest_engine import BacktestEngine


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

    def get_z_size(self):
        if self.select_range == 300:
            range_z_size = pd.merge(self.idx_wgt_data.loc[self.idx_wgt_data['index_code'] == '000300.SH'].reset_index(),
                                    self.size_data.reset_index(), how='inner', on=['date', 'code'])
            range_z_size = range_z_size.loc[:, ['date', 'code', 'size']]
        elif self.select_range == 500:
            range_z_size = pd.merge(self.idx_wgt_data.loc[self.idx_wgt_data['index_code'] == '000905.SH'].reset_index(),
                                    self.size_data.reset_index(), how='inner', on=['date', 'code'])
            range_z_size = range_z_size.loc[:, ['date', 'code', 'size']]
        elif self.select_range == 800:
            range_z_size = pd.merge(self.idx_wgt_data.loc[self.idx_wgt_data['index_code']
                                    .isin(['000300.SH', '000905.SH'])].reset_index(),
                                    self.size_data.reset_index(), how='inner', on=['date', 'code'])
            range_z_size = range_z_size.loc[:, ['date', 'code', 'size']]
        else:
            range_z_size = self.size_data.reset_index()
        dates = range_z_size['date'].unique()
        split_dates = np.array_split(dates, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            parallel_res = Parallel()(delayed(DataProcess.JOB_cross_section_Z_score)
                                      (range_z_size, 'size', dates) for dates in split_dates)
        range_z_size = pd.concat(parallel_res)
        range_z_size.set_index('date', inplace=True)
        return range_z_size

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
    def JOB_opti_weight(factor, dates, adj_interval, opt_option, target_sigma, risk_aversion, mv_max_exp,
                        mv_min_exp, bm_stk_wgt, indu_dummies, z_size, risk_exp, risk_cov, spec_risk):
        dfs = []
        for date in dates:
            day_factor = factor.loc[date, :].sort_values('code').copy()
            # -------------------------权重设置--------------------------
            # 设置基准权重
            day_bm_stk_wgt = bm_stk_wgt.loc[date, :].sort_values('code').copy()
            base_weight = pd.merge(day_factor.reset_index(), day_bm_stk_wgt.reset_index(),
                                   how='outer', on=['date', 'code'])
            base_weight['base_weight'] = base_weight['weight'].fillna(0).values
            codes = base_weight['code'].values
            z_size_dict = dict(zip(z_size.loc[date, 'code'].values, z_size.loc[date, 'size'].values))
            base_weight['z_size'] = base_weight['code'].map(z_size_dict)
            # 设置权重上下限
            #   基准权重大于1的话 可以到 2*base_weight
            #   基准权重小于1的话 只能到 2
            #   没有overall或z_size因子值的 总权重为0
            conditions = [pd.isnull(base_weight['overall']).values | pd.isnull(base_weight['z_size']).values,
                          base_weight['base_weight'].values <= 1]
            choices = [-1 * base_weight['base_weight'].values, 2 - base_weight['base_weight'].values]
            array_upbound = np.select(conditions, choices, default=base_weight['base_weight'].values)
            array_lowbound = -1 * base_weight['base_weight'].values
            array_overall = base_weight['overall'].fillna(0).values
            array_z_size = base_weight['z_size'].fillna(0).values
            # -----------------------哑变量设置-------------------------
            day_indu_dummies = indu_dummies.loc[date, :].set_index('code')
            array_indu_dummies = day_indu_dummies.loc[codes, :].values
            indus = day_indu_dummies.columns
            # ----------------------风险因子设置------------------------
            day_risk_exp = risk_exp.loc[date, :].set_index('code')
            risk_factors = day_risk_exp.columns
            array_risk_exp = day_risk_exp.loc[codes, :].values
            day_risk_cov = risk_cov.loc[date, :].set_index('code')
            array_risk_cov = day_risk_cov.loc[risk_factors, risk_factors].values
            array_risk_cov = array_risk_cov * 21 * 0.0001
            array_risk_cov = 0.5 * (array_risk_cov + array_risk_cov.T)
            day_spec_risk = spec_risk.loc[date, :].set_index('code')
            array_spec_risk = day_spec_risk.loc[codes, 'specific_risk'].values
            array_spec_risk = array_spec_risk * np.sqrt(21) * 0.01
            # *********************************************************
            # ------------------------变量-----------------------------
            stk_num, risk_factors_num = array_risk_exp.shape
            # 相对权重
            solve_weight = cp.Variable(stk_num)
            cons = []
            cons.append(solve_weight <= array_upbound)
            cons.append(solve_weight >= array_lowbound)
            # ---------------------跟踪误差设置-------------------------
            tot_risk_exp = array_risk_exp.T * solve_weight / 100
            risk_variance = cp.quad_form(tot_risk_exp, array_risk_cov) + \
                            cp.sum_squares(cp.multiply(array_spec_risk, solve_weight / 100))
            overall_exp = array_overall * solve_weight
            if opt_option == 1:
                obj = overall_exp
                sigma = target_sigma / np.sqrt(250/adj_interval)
                cons.append(risk_variance <= sigma ** 2)
            else:
                obj = overall_exp - risk_aversion * risk_variance
            # ---------------------行业中性设置------------------------
            for i in range(len(indus)):
                cons.append(cp.sum(array_indu_dummies[:, i].T * solve_weight) == 0)
                pass
            # ---------------------风险因子暴露------------------------
            '''
            # 除 Market和Size 外的10个风险因子
            risk_factors = ['Trend', 'Beta', 'Volatility', 'Liquidity', 'Value',
                            'Growth', 'SOE', 'Uncertainty', 'Cubic size']
            '''
            # ---------------------市值主动暴露------------------------
            cons.append(cp.sum(array_z_size * solve_weight / 100) <= mv_max_exp)
            cons.append(cp.sum(array_z_size * solve_weight / 100) >= -mv_min_exp)
            # ********************************************************
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
                print(date)
                print(prob.status)
                raise ValueError('优化失败')
            base_weight['weight'] = np.round(base_weight['base_weight'].values + opti_weight, 3)
            base_weight = base_weight.loc[base_weight['weight'] > 0, ['date', 'code', 'weight']]
            print(date, 'opti finish!')
            dfs.append(base_weight)
        res_df = pd.concat(dfs)
        return res_df

    def run(self):
        start_time = datetime.datetime.now()
        # get data
        self.initialize_strategy()
        range_z_size = self.get_z_size()
        overall_factor = self.factors_combination()
        # get target weight
        dates = overall_factor.index.unique().strftime("%Y%m%d")
        split_dates = np.array_split(dates, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            parallel_res = \
                Parallel()(delayed(alpha_version_3.JOB_opti_weight)
                           (overall_factor, dates, self.adj_interval, self.opt_option, self.target_sigma,
                            self.risk_aversion, self.mv_max_exp, self.mv_min_exp, self.bm_stk_wgt,
                            self.industry_dummies, range_z_size, self.risk_exp, self.risk_cov, self.spec_risk)
                           for dates in split_dates)
        target_weight = pd.concat(parallel_res)
        target_weight.set_index('date', inplace=True)
        target_weight = target_weight.sort_index()
        target_weight.to_csv(self.folder_dir + 'TARGET_WEIGHT.csv', encoding='gbk')
        # backtest
        QE = BacktestEngine(save_name=self.strategy_name, stock_capital=STRATEGY_CONFIG['capital'])
        bt_start = target_weight.index[0].strftime('%Y%m%d')
        bt_end = (target_weight.index[-1] - datetime.timedelta(days=1)).strftime('%Y%m%d')
        QE.run(target_weight, bt_start, bt_end, self.adj_interval, self.benchmark)
        print('Time used:', datetime.datetime.now()-start_time)


if __name__ == '__main__':
    print(datetime.datetime.now())
    a = alpha_version_3('Alpha_version_3')
    kk = a.run()
    print(datetime.datetime.now())