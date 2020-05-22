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
        adj_interval = STRATEGY_CONFIG['adj_interval']
        super().initialize_strategy(start, end, benchmark, select_range, industry, adj_interval)
        self.logger.info('Strategy start time: %s' % datetime.datetime.now().strftime('%Y/%m/%d - %H:%M:%S'))
        self.logger.info('*' * 50)
        self.capital = STRATEGY_CONFIG['capital']
        self.target_sigma = STRATEGY_CONFIG['target_sigma']
        self.mv_max_exp = STRATEGY_CONFIG['mv_max_exp']
        self.mv_min_exp = STRATEGY_CONFIG['mv_min_exp']
        self.weight_intercept = STRATEGY_CONFIG['weight_intercept']
        self.n_codes = STRATEGY_CONFIG['n_codes']

    def get_factors(self, measure, factor, direction, if_fillna, weight):
        self.logger.info('  -Factor: %s is processing...' % factor)
        factor_df = self.process_factor(measure, factor, direction, if_fillna)
        factor_df[factor] = factor_df[factor] * weight
        factor_df.set_index(['date', 'code'], inplace=True)
        return factor_df

    def factors_combination(self):
        self.logger.info('STRATEGY PARAMETERS: ')
        categories = []
        for category in CATEGORY_WEIGHT.keys():
            self.logger.info('-%s' % category)
            parameters_list = FACTOR_WEIGHT[category]
            factors_in_category = []
            for measure, factor, direction, if_fillna, weight in parameters_list:
                factor_df = self.get_factors(measure, factor, direction, if_fillna, weight)
                factors_in_category.append(factor_df)
            category_df = pd.concat(factors_in_category, join='inner', axis=1)
            category_df[category] = category_df.sum(axis=1)
            category_df = category_df.reset_index().loc[:, ['date', 'code', category]]
            # 合并时无需再去极值
            category_df = DataProcess.standardize(category_df, category, False, self.n_jobs)
            category_df[category] = CATEGORY_WEIGHT[category] * category_df[category]
            category_df.set_index(['date', 'code'], inplace=True)
            categories.append(category_df)
        merged_df = pd.concat(categories, join='inner', axis=1)
        merged_df['overall'] = merged_df.sum(axis=1)
        merged_df = merged_df.reset_index().loc[:, ['date', 'code', 'overall']]
        merged_df = pd.merge(merged_df, self.code_range.reset_index(), how='right', on=['date', 'code'])
        merged_df.set_index('date', inplace=True)
        merged_df = merged_df.loc[:, ['code', 'overall', 'status', 'isST']]
        print('Factors combination finish...')
        return merged_df

    def get_next_bm_stk_wgt(self):
        next_bm_stk_wgt = self.bm_stk_wgt.copy()
        next_bm_stk_wgt['former_date'] = next_bm_stk_wgt.index.strftime('%Y%m%d')
        former_dict = dict(zip(self.calendar[1:], self.calendar[:-1]))
        next_bm_stk_wgt['former_date'] = next_bm_stk_wgt['former_date'].map(former_dict)
        next_bm_stk_wgt = next_bm_stk_wgt.dropna(subset=['former_date'])
        next_bm_stk_wgt['former_date'] = pd.to_datetime(next_bm_stk_wgt['former_date'])
        next_bm_stk_wgt.set_index('former_date', inplace=True)
        next_bm_stk_wgt.index.names = ['date']
        return next_bm_stk_wgt

    def get_base_weight(self, overall_factor, next_bm_stk_wgt):
        trade_dates = next_bm_stk_wgt.reset_index()['date'].unique()
        overall_factor = overall_factor.reset_index()
        # 需考虑到因子出现在不在交易日的情况
        overall_factor = overall_factor.loc[overall_factor['date'].isin(trade_dates), :]
        base_weight = pd.merge(overall_factor, next_bm_stk_wgt.reset_index(), how='outer', on=['date', 'code'])
        # --------------------------------------------------------------
        self.indus = self.industry_dummies.columns.difference(['code'])
        base_weight = pd.merge(base_weight, self.industry_dummies.reset_index(), how='left', on=['date', 'code'])
        # --------------------------------------------------------------
        self.risks = self.risk_exp.columns.difference(['code'])
        base_weight = pd.merge(base_weight, self.risk_exp.reset_index(), how='left', on=['date', 'code'])
        base_weight = pd.merge(base_weight, self.spec_risk.reset_index(), how='left', on=['date', 'code'])
        base_weight = pd.merge(base_weight, self.size_data.reset_index(), how='left', on=['date', 'code'])
        # --------------------------------------------------------------
        # 用于过滤没有行业的stk
        base_weight['sum_dummies'] = base_weight[self.indus].sum(axis=1)
        base_weight[self.indus] = base_weight[self.indus].fillna(0)
        # 用于过滤没有风险因子的stk
        base_weight['sum_risks'] = base_weight[self.risks].sum(axis=1)
        base_weight[self.risks] = base_weight[self.risks].fillna(0)
        base_weight['filled_spec_risk'] = base_weight['specific_risk'].fillna(0)
        base_weight['filled_z_mv'] = base_weight['z_mv'].fillna(0)
        base_weight['filled_overall'] = base_weight['overall'].fillna(0)
        base_weight['weight'] = base_weight['weight'].fillna(0)
        # --------------------------------------------------------------
        base_weight.set_index('date', inplace=True)
        return base_weight

    def shift_target_weight(self, target_weight):
        next_date_dict = dict(zip(self.calendar[:-1], self.calendar[1:]))
        target_weight['next_date'] = target_weight.index.strftime('%Y%m%d')
        target_weight['next_date'] = target_weight['next_date'].map(next_date_dict)
        target_weight['next_date'] = pd.to_datetime(target_weight['next_date'])
        target_weight.set_index('next_date', inplace=True)
        target_weight.index.names = ['date']
        return target_weight

    @staticmethod
    def JOB_opti_weight(base_weight, risk_cov, dummies_field, risks_field, dates, adj_interval,
                        target_sigma, mv_max_exp, mv_min_exp, weight_intercept):
        dfs = []
        fail_dates = []
        for date in dates:
            day_base_weight = base_weight.loc[date, :].copy()
            # -----------------------get array--------------------------
            array_codes = day_base_weight['code'].values
            array_overall = day_base_weight['filled_overall'].values
            array_base_weight = day_base_weight['weight'].values
            array_z_size = day_base_weight['filled_z_mv'].values
            array_indu_dummies = day_base_weight[dummies_field].values
            array_risk_exp = day_base_weight[risks_field].values
            array_spec_risk = day_base_weight['filled_spec_risk'].values
            # -------------------------set para-------------------------
            n_stk, n_risk = array_risk_exp.shape
            solve_weight = cp.Variable(n_stk)
            # -------------------------get cov--------------------------
            day_risk_cov = risk_cov.loc[date, :].set_index('code')
            array_risk_cov = day_risk_cov.loc[risks_field, risks_field].values
            array_risk_cov = 0.5 * (array_risk_cov + array_risk_cov.T)
            # ------------------------get bound-------------------------
            # 设置权重上下限
            #   权重上限 可以到 weight_intercept + 0.5 * base_weight
            #   没有overall 或 size 或 sum_dummies 或 sum_risks 或 specific_risk 因子值的 总权重为0
            array_upbound = np.where(
                (day_base_weight['status'] == '停牌').values | pd.notnull(day_base_weight['isST']).values |
                pd.isnull(day_base_weight['overall']).values | pd.isnull(day_base_weight['sum_dummies']).values |
                pd.isnull(day_base_weight['sum_risks']).values | pd.isnull(day_base_weight['specific_risk']).values |
                pd.isnull(day_base_weight['z_mv']).values,
                -1 * day_base_weight['weight'].values, weight_intercept + 0.5 * day_base_weight['weight'].values)
            array_lowbound = -1 * day_base_weight['weight'].values
            # ----------------------track error-------------------------
            overall_exp = array_overall * solve_weight / 100
            obj = overall_exp
            tot_risk_exp = array_risk_exp.T * solve_weight / 100
            variance = cp.quad_form(tot_risk_exp, array_risk_cov) + \
                       cp.sum_squares(cp.multiply(array_spec_risk, solve_weight / 100))
            sigma = target_sigma / np.sqrt(252 / adj_interval)
            # -----------------------set cons---------------------------
            cons = []
            #  权重上下缘
            cons.append(solve_weight <= array_upbound)
            cons.append(solve_weight >= array_lowbound)
            #  跟踪误差设置
            cons.append(variance <= sigma ** 2)
            #  行业中性设置
            for i in range(len(dummies_field)):
                cons.append(cp.sum(array_indu_dummies[:, i].T * solve_weight) == 0)
            #  市值主动暴露
            cons.append(cp.sum(array_z_size * solve_weight / 100) <= mv_max_exp)
            cons.append(cp.sum(array_z_size * solve_weight / 100) >= mv_min_exp)
            # -------------------------优化-----------------------------
            prob = cp.Problem(cp.Maximize(obj), constraints=cons)
            argskw = {'mi_max_iters': 1000, 'feastol': 1e-3, 'abstol': 1e-3}
            try:
                prob.solve(solver='ECOS', **argskw)
            except:
                prob.solve(solver='SCS', **argskw)
            if prob.status in ('optimal', 'optimal_inaccurate'):
                opti_weight = np.array(solve_weight.value)
            else:
                print('%s OPTI ERROR!\n -status: %s' % (date, prob.status))
                fail_dates.append(date)
                continue
            day_base_weight['weight'] = np.round(array_base_weight + opti_weight, 2)
            day_base_weight = day_base_weight.loc[day_base_weight['weight'] > 0.01, ['code', 'weight']]
            print('%s OPTI finish \n -n_codes: %i  n_selections: %i' %
                  (date, array_codes.shape[0], day_base_weight.shape[0]))
            dfs.append(day_base_weight)
        res_df = pd.concat(dfs)
        return [res_df, fail_dates]

    # 工具函数 当某天优化失败时，用前面最近一个优化成功的权重复制，如果这天相隔超过10天，标配
    @staticmethod
    def JOB_fill_df(target_weight, fail_dates, next_bm_stk_weight):
        idx = target_weight.index.unique()
        fill_dfs = []
        for date in fail_dates:
            if idx[idx < date].empty:
                fill_dfs.append(next_bm_stk_weight.loc[date, :].copy())
            else:
                fill_date = idx[idx < date][-1]
                if (fill_date - pd.to_datetime(date)).days <= 10:
                    fill_df = target_weight.loc[fill_date, :].copy()
                    fill_df.index.names = ['date']
                    fill_df.reset_index(inplace=True)
                    fill_df['date'] = pd.to_datetime(date)
                    fill_df.set_index('date', inplace=True)
                    fill_dfs.append(fill_df)
                else:
                    fill_dfs.append(next_bm_stk_weight.loc[date, :].copy())
        df = pd.concat(fill_dfs)
        return df

    @staticmethod
    def JOB_limit_n_stk(target_weight, dates, stk_count, n_codes, next_bm_stk_weight):
        revise_dfs = []
        for date in dates:
            # 如果优化后选股数量大于限制，标配
            if stk_count[date] > n_codes:
                tmp = stk_count[stk_count.index < date].copy()
                tmp = tmp[tmp <= n_codes]
                if tmp.empty:
                    revise_dfs.append(next_bm_stk_weight.loc[date, :].copy())
                else:
                    former_date = tmp[tmp <= n_codes].index[-1]
                    # 如果间隔 <= 10天，copy，否则 标配
                    if (pd.to_datetime(date) - former_date).days <= 10:
                        cp_df = target_weight.loc[former_date, :].copy()
                        cp_df.reset_index()
                        cp_df['date'] = pd.to_datetime(date)
                        cp_df.set_index('date', inplace=True)
                        revise_dfs.append(cp_df)
                    else:
                        revise_dfs.append(next_bm_stk_weight.loc[date, :].copy())
            else:
                revise_dfs.append(target_weight.loc[date, :].copy())
        df = pd.concat(revise_dfs)
        return df

    def run(self):
        start_time = datetime.datetime.now()
        # -----------------------------get data------------------------------
        self.initialize_strategy()
        # 使用 risk 中的 Size，避免不同的 size 造成误差
        # range_z_size = self.get_z_size()
        overall_factor = self.factors_combination()
        next_bm_stk_wgt = self.get_next_bm_stk_wgt()
        # get base weight
        base_weight = self.get_base_weight(overall_factor, next_bm_stk_wgt)
        # get target weight
        dates = base_weight.index.unique().strftime("%Y%m%d")
        split_dates = np.array_split(dates, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            parallel_res = \
                Parallel()(delayed(alpha_version_3.JOB_opti_weight)
                           (base_weight, self.risk_cov, self.indus, self.risks, dates, self.adj_interval,
                            self.target_sigma, self.mv_max_exp, self.mv_min_exp, self.weight_intercept)
                           for dates in split_dates)
        parallel_dfs = []
        fail_dates = []
        for res in parallel_res:
            parallel_dfs.append(res[0])
            fail_dates.extend(res[1])
        target_weight = pd.concat(parallel_dfs)
        target_weight.to_csv(self.folder_dir + 'RAW_TARGET_WEIGHT.csv', encoding='gbk')
        print('raw weight is ready...')
        target_weight = target_weight.sort_index()
        # ------------------------limit n_codes-----------------------------
        stk_count = target_weight.groupby('date')['code'].count()
        dates = target_weight.index.unique().strftime('%Y%m%d')
        split_dates = np.array_split(dates, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            parallel_res = Parallel()(delayed(alpha_version_3.JOB_limit_n_stk)
                                      (target_weight, dates, stk_count, self.n_codes, next_bm_stk_wgt)
                                      for dates in split_dates)
        target_weight = pd.concat(parallel_res)
        target_weight = target_weight.sort_index()
        # ----------------------fill target weight--------------------------
        if fail_dates:
            fill_df = alpha_version_3.JOB_fill_df(target_weight, fail_dates, next_bm_stk_wgt)
            target_weight = pd.concat([target_weight, fill_df])
            target_weight = target_weight.sort_index()
        # ---------------------------date shift-----------------------------
        target_weight = self.shift_target_weight(target_weight)
        # ------------------------------------------------------------------
        target_weight.to_csv(self.folder_dir + 'TARGET_WEIGHT.csv', encoding='gbk')
        print('target_weight is ready...')
        # --------------------------backtest--------------------------------
        bt_start = target_weight.index[0].strftime('%Y%m%d')
        bt_end = (target_weight.index[-1] - datetime.timedelta(days=1)).strftime('%Y%m%d')
        QE = BacktestEngine(self.strategy_name, bt_start, bt_end, self.adj_interval, self.benchmark,
                            stock_capital=self.capital)
        portfolio_value = QE.run(target_weight, bt_start, bt_end)
        self.logger.info('Strategy finish time: %s' % datetime.datetime.now().strftime('%Y/%m/%d - %H:%M:%S'))
        self.logger.info('*' * 50)
        self.logger.info('PERFORMANCE:')
        self.logger.info('-ANN_Alpha: %f' % DataProcess.calc_alpha_ann_return(
            portfolio_value['TotalValue'], portfolio_value['BenchmarkValue']))
        MDD, MDD_period = \
            DataProcess.calc_alpha_max_draw_down(portfolio_value['TotalValue'], portfolio_value['BenchmarkValue'])
        self.logger.info('-Alpha_MDD: %f' % MDD)
        self.logger.info('-Alpha_MDD period: %s - %s' % (MDD_period[0], MDD_period[1]))
        self.logger.info('-Alpha_sharpe: %f' % DataProcess.calc_alpha_sharpe(
            portfolio_value['TotalValue'], portfolio_value['BenchmarkValue']))
        print('Time used:', datetime.datetime.now() - start_time)


if __name__ == '__main__':
    print(datetime.datetime.now())
    a = alpha_version_3('FINQUAL_para1_0522')
    kk = a.run()
    print(datetime.datetime.now())
