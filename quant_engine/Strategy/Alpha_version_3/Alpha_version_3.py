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
        self.target_sigma = STRATEGY_CONFIG['target_sigma']
        self.mv_max_exp = STRATEGY_CONFIG['mv_max_exp']
        self.mv_min_exp = STRATEGY_CONFIG['mv_min_exp']
        self.n_codes = STRATEGY_CONFIG['n_codes']

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

    def get_next_bm_stk_wgt(self):
        next_bm_stk_wgt = self.bm_stk_wgt.copy()
        next_bm_stk_wgt['str_date'] = next_bm_stk_wgt.index.strftime('%Y%m%d')
        former_dict = dict(zip(self.calendar[1:], self.calendar[:-1]))
        next_bm_stk_wgt['former_date'] = next_bm_stk_wgt['str_date'].map(former_dict)
        next_bm_stk_wgt = next_bm_stk_wgt.dropna(subset=['former_date'])
        next_bm_stk_wgt['former_date'] = pd.to_datetime(next_bm_stk_wgt['former_date'])
        next_bm_stk_wgt.set_index('former_date', inplace=True)
        next_bm_stk_wgt.index.names = ['date']
        return next_bm_stk_wgt

    def get_base_weight(self, overall_factor, z_size, next_bm_stk_wgt):
        trade_dates = next_bm_stk_wgt.reset_index()['date'].unique()
        base_weight = pd.merge(overall_factor.reset_index(), next_bm_stk_wgt.reset_index(),
                               how='outer', on=['date', 'code'])
        base_weight['filled_overall'] = base_weight['overall'].fillna(0)
        # 需考虑到因子出现在不在交易日的情况
        base_weight = base_weight.loc[base_weight['date'].isin(trade_dates), :]
        base_weight['base_weight'] = base_weight['weight'].fillna(0)
        # --------------------------------------------------------------
        self.indus = self.industry_dummies.columns.difference(['code'])
        base_weight = pd.merge(base_weight, self.industry_dummies.reset_index(), how='left', on=['date', 'code'])
        # 用于过滤没有行业的stk
        base_weight['sum_dummies'] = base_weight[self.indus].sum(axis=1)
        base_weight[self.indus] = base_weight[self.indus].fillna(0)
        # --------------------------------------------------------------
        base_weight = pd.merge(base_weight, z_size.reset_index(), how='left', on=['date', 'code'])
        base_weight['filled_size'] = base_weight['size'].fillna(0)
        # --------------------------------------------------------------
        self.risks = self.risk_exp.columns.difference(['code'])
        base_weight = pd.merge(base_weight, self.risk_exp.reset_index(), how='left', on=['date', 'code'])
        # 用于过滤没有风险因子的stk
        base_weight['sum_risks'] = base_weight[self.risks].sum(axis=1)
        base_weight[self.risks] = base_weight[self.risks].fillna(0)
        # --------------------------------------------------------------
        base_weight = pd.merge(base_weight, self.spec_risk.reset_index(), how='left', on=['date', 'code'])
        base_weight['filled_spec_risk'] = base_weight['specific_risk'].fillna(0)
        # --------------------------------------------------------------
        base_weight.set_index('date', inplace=True)
        return base_weight

    @staticmethod
    def JOB_opti_weight(base_weight, risk_cov, dummies_field, risks_field, dates, adj_interval,
                        target_sigma, mv_max_exp, mv_min_exp):
        dfs = []
        fail_dates = []
        for date in dates:
            day_base_weight = base_weight.loc[date, :].copy()
            # -----------------------get array--------------------------
            array_codes = day_base_weight['code'].values
            array_overall = day_base_weight['filled_overall'].values
            array_base_weight = day_base_weight['base_weight'].values
            array_z_size = day_base_weight['filled_size'].values
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
            #   基准权重大于1的话 可以到 2 + 1.5 * base_weight
            #   基准权重小于1的话 只能到 2
            #   没有overall 或 size 或 sum_dummies 或 sum_risks 或 specific_risk 因子值的 总权重为0
            conditions = [pd.isnull(day_base_weight['overall']).values | pd.isnull(day_base_weight['size']).values |
                          pd.isnull(day_base_weight['sum_dummies']).values |
                          pd.isnull(day_base_weight['sum_risks']).values |
                          pd.isnull(day_base_weight['specific_risk']).values,
                          day_base_weight['base_weight'].values <= 1]
            choices = [-1 * day_base_weight['base_weight'].values, 2 - day_base_weight['base_weight'].values]
            array_upbound = np.select(conditions, choices, default=2 + 0.5 * day_base_weight['base_weight'].values)
            array_lowbound = -1 * day_base_weight['base_weight'].values
            # ----------------------track error-------------------------
            tot_risk_exp = array_risk_exp.T * solve_weight / 100
            risk_variance = cp.quad_form(tot_risk_exp, array_risk_cov) + \
                            cp.sum_squares(cp.multiply(array_spec_risk, solve_weight / 100))
            overall_exp = array_overall * solve_weight
            obj = overall_exp
            sigma = target_sigma / np.sqrt(250 / adj_interval)
            # -----------------------set cons---------------------------
            cons = []
            cons.append(solve_weight <= array_upbound)
            cons.append(solve_weight >= array_lowbound)
            #  跟踪误差设置
            cons.append(risk_variance <= sigma ** 2)
            #  行业中性设置
            for i in range(len(dummies_field)):
                cons.append(cp.sum(array_indu_dummies[:, i].T * solve_weight) == 0)
            #  市值主动暴露
            cons.append(cp.sum(array_z_size * solve_weight / 100) <= mv_max_exp)
            cons.append(cp.sum(array_z_size * solve_weight / 100) >= -mv_min_exp)
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
            day_base_weight['weight'] = np.round(array_base_weight + opti_weight, 3)
            day_base_weight = day_base_weight.loc[day_base_weight['weight'] > 0, ['code', 'weight']]
            print('%s OPTI finish \n -n_codes: %i  n_selections: %i' %
                  (date, array_codes.shape[0], day_base_weight.shape[0]))
            dfs.append(day_base_weight)
        res_df = pd.concat(dfs)
        return [res_df, fail_dates]

    # 工具函数 当某天优化失败时，用前面最近一个优化成功的权重复制
    @staticmethod
    def JOB_fill_df(target_weight, dates):
        idx = target_weight.index.unique()
        fill_dfs = []
        for date in dates:
            if idx[idx < date].empty:
                pass
            else:
                fill_date = idx[idx < date].iloc[-1]
                fill_df = target_weight.loc[fill_date, :].copy()
                fill_df.index.names = ['date']
                fill_df.reset_index(inplace=True)
                fill_df['date'] = pd.to_datetime(date)
                fill_df.set_index('date', inplace=True)
                fill_dfs.append(fill_df)
        df = pd.concat(fill_dfs)
        return df

    @staticmethod
    def JOB_limit_n_stk(target_weight, dates, stk_count, n_codes, next_bm_stk_weight):
        revise_dfs = []
        for date in dates:
            # 如果优化后选股数量大于限制，标配
            if stk_count[date] > n_codes:
                tmp = stk_count[stk_count.index < date].copy()
                if tmp.empty:
                    revise_dfs.append(next_bm_stk_weight.loc[date, :].copy())
                else:
                    former_date = tmp[tmp <= n_codes].index[-1]
                    # 如果间隔 <= 10天，copy，否则 标配
                    if (pd.to_datetime('date') - former_date).days <= 10:
                        revise_dfs.append(target_weight.loc[former_date, :].copy())
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
        range_z_size = self.get_z_size()
        overall_factor = self.factors_combination()
        next_bm_stk_wgt = self.get_next_bm_stk_wgt()
        # get base weight
        base_weight = self.get_base_weight(overall_factor, range_z_size, next_bm_stk_wgt)
        # get target weight
        dates = base_weight.index.unique().strftime("%Y%m%d")
        split_dates = np.array_split(dates, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            parallel_res = \
                Parallel()(delayed(alpha_version_3.JOB_opti_weight)
                           (base_weight, self.risk_cov, self.indus, self.risks, dates, self.adj_interval,
                            self.target_sigma, self.mv_max_exp, self.mv_min_exp)
                           for dates in split_dates)
        # ----------------------fill target weight--------------------------
        parallel_dfs = []
        fail_dates = []
        for res in parallel_res:
            parallel_dfs.append(res[0])
            fail_dates.extend(res[1])
        target_weight = pd.concat(parallel_dfs)
        if fail_dates:
            split_fail_dates = np.array_split(fail_dates, self.n_jobs)
            with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
                parallel_res = Parallel()(delayed(alpha_version_3.JOB_fill_df)
                                          (target_weight, dates) for dates in split_fail_dates)
            tot_fill_df = pd.concat(parallel_res)
            target_weight = pd.concat([target_weight, tot_fill_df])
        target_weight.to_csv(self.folder_dir + 'RAW_TARGET_WEIGHT.csv', encoding='gbk')
        # ------------------------limit n_codes-----------------------------
        stk_count = target_weight.groupby('date')['code'].count()
        dates = target_weight.index.unique().strftime('%Y%m%d')
        split_dates = np.array_split(dates, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            parallel_res = Parallel()(delayed(alpha_version_3.JOB_limit_n_stk)
                                      (target_weight, dates, stk_count, self.n_codes, next_bm_stk_wgt)
                                      for dates in split_dates)
        target_weight = pd.concat(parallel_res)
        target_weight.to_csv(self.folder_dir + 'TARGET_WEIGHT.csv', encoding='gbk')
        # --------------------------backtest--------------------------------
        QE = BacktestEngine(save_name=self.strategy_name, stock_capital=STRATEGY_CONFIG['capital'])
        bt_start = target_weight.index[0].strftime('%Y%m%d')
        bt_end = (target_weight.index[-1] - datetime.timedelta(days=1)).strftime('%Y%m%d')
        QE.run(target_weight, bt_start, bt_end, self.adj_interval, self.benchmark)
        print('Time used:', datetime.datetime.now() - start_time)


if __name__ == '__main__':
    print(datetime.datetime.now())
    a = alpha_version_3('Alpha_version_3')
    kk = a.run()
    print(datetime.datetime.now())
