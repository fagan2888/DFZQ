# 测试券商模型

import pandas as pd
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import datetime
from backtest_engine import BacktestEngine
from strategy_base import StrategyBase
from securities_opt_CONFIG import STRATEGY_CONFIG, CATEGORY_WEIGHT, FACTOR_WEIGHT
from data_process import DataProcess
import global_constant
import os.path
import cvxpy as cp

class SecTestOpt(StrategyBase):
    def __init__(self, strategy_name):
        self.save_name = ''
        for cate in CATEGORY_WEIGHT:
            for fct in FACTOR_WEIGHT[cate]:
                if self.save_name:
                    self.save_name = self.save_name + '+'
                self.save_name = self.save_name + '{0}({1})'.format(fct[2], str(fct[5] * CATEGORY_WEIGHT[cate]))
        self.folder_dir = global_constant.ROOT_DIR + 'Sec_Opt/{0}/'.format(self.save_name)
        if os.path.exists(self.folder_dir):
            pass
        else:
            os.makedirs(self.folder_dir.rstrip('/'))
        super().__init__(strategy_name)

    def initialize_strategy(self):
        self.n_jobs = global_constant.N_STRATEGY
        self.start = STRATEGY_CONFIG['start']
        self.end = STRATEGY_CONFIG['end']
        self.benchmark = STRATEGY_CONFIG['benchmark']
        self.select_range = STRATEGY_CONFIG['select_range']
        self.industry = STRATEGY_CONFIG['industry']
        self.adj_interval = STRATEGY_CONFIG['adj_interval']
        self.capital = STRATEGY_CONFIG['capital']
        self.lmd = STRATEGY_CONFIG['lambda']
        self.data_prepare(self.start, self.end)
        self.init_log()
        self.logger.info('Strategy start time: %s' % datetime.datetime.now().strftime('%Y/%m/%d - %H:%M:%S'))
        self.logger.info('*' * 50)

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
                self.logger.info('  -Factor: %s   weight:%i' % (factor, weight))
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
        merged_df = merged_df.set_index('date')
        folder_dir = global_constant.ROOT_DIR + 'Sec_Opt/{0}/'.format(self.save_name)
        merged_df.to_csv(folder_dir + 'FactorsCombination.csv', encoding='gbk')
        print('Factors combination finish...')
        merged_df = merged_df.loc[:, ['code', 'overall']]
        return merged_df

    def get_bm_stk_wgt(self):
        bm_sec = self.bm_stk_wgt.copy()
        bm_sec = pd.merge(bm_sec.reset_index(), self.industry_data.reset_index(), on=['date', 'code'])
        bm_sec = bm_sec.loc[bm_sec['industry'] == '证券Ⅱ(中信)', :]
        wgt_sum = bm_sec.groupby('date')['weight'].sum()
        wgt_sum = wgt_sum.to_dict()
        bm_sec['tot_wgt'] = bm_sec['date'].map(wgt_sum)
        bm_sec['weight'] = bm_sec['weight'] / bm_sec['tot_wgt'] * 100
        bm_sec.set_index('date', inplace=True)
        bm_sec = bm_sec.loc[:, ['code', 'weight']]
        return bm_sec

    def opt_portfolio(self, factor_df, bm_sec):
        later_dict = dict(zip(self.calendar[:-1], self.calendar[1:]))
        factor_df['next_date'] = factor_df.index.strftime('%Y%m%d')
        factor_df['next_date'] = factor_df['next_date'].map(later_dict)
        bm_df = bm_sec.copy()
        bm_df.rename(columns={'weight': 'bm_weight'}, inplace=True)
        bm_df['next_date'] = bm_df.index.strftime('%Y%m%d')
        merge = pd.merge(factor_df.reset_index(), bm_df, how='outer', on=['next_date', 'code'])
        merge['date'] = merge.groupby('next_date')['date'].fillna(method='ffill')
        merge['date'] = merge.groupby('next_date')['date'].fillna(method='bfill')
        merge = merge.dropna(subset=['date'])
        merge = merge.dropna(subset=['next_date'])
        self.risks = self.risk_exp.columns.difference(['code'])
        merge = pd.merge(merge, self.risk_exp.reset_index(), how='left', on=['date', 'code'])
        merge = pd.merge(merge, self.spec_risk.reset_index(), how='left', on=['date', 'code'])
        merge['overall'] = merge['overall'].fillna(-999)
        merge['bm_weight'] = merge['bm_weight'].fillna(0)
        merge[self.risks] = merge[self.risks].fillna(0)
        merge = merge.drop('date', axis=1)
        merge.set_index('next_date', inplace=True)
        merge.index = pd.to_datetime(merge.index)
        merge.index.names = ['date']
        dates = merge.index.unique().strftime('%Y%m%d')
        split_dates = np.array_split(dates, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            parallel_res = Parallel()(delayed(SecTestOpt.JOB_opt_weight)
                                      (dates, merge, self.risks, self.risk_cov, self.lmd)
                                      for dates in split_dates)
        parallel_dfs = []
        fail_dates = []
        for res in parallel_res:
            parallel_dfs.append(res[0])
            fail_dates.extend(res[1])
        target_weight = pd.concat(parallel_dfs)
        print('raw weight is ready...')
        target_weight = target_weight.sort_index()
        print('fail_dates: ', fail_dates)
        return target_weight

    @staticmethod
    def JOB_opt_weight(dates, base_weight, risks_field, risk_cov, lmd):
        dfs = []
        fail_dates = []
        for date in dates:
            day_base_weight = base_weight.loc[date, :].copy()
            # -----------------------get array--------------------------
            array_codes = day_base_weight['code'].values
            array_overall = day_base_weight['overall'].values
            array_base_weight = day_base_weight['bm_weight'].values
            array_risk_exp = day_base_weight.loc[:, risks_field].values
            array_spec_risk = day_base_weight['specific_risk'].values
            # -------------------------set para-------------------------
            n_stk, n_risk = array_risk_exp.shape
            solve_weight = cp.Variable(n_stk)
            # -------------------------get cov--------------------------
            day_risk_cov = risk_cov.loc[date, :].set_index('code')
            array_risk_cov = day_risk_cov.loc[risks_field, risks_field].values
            array_risk_cov = 0.5 * (array_risk_cov + array_risk_cov.T)
            # ------------------------get bound-------------------------
            # 设置权重上下限
            # 行业内绝对权重 20 或 1.5 * base_weight
            array_upbound = np.where(20 - array_base_weight > 0.5 * array_base_weight, 20 - array_base_weight,
                                     0.5 * array_base_weight)
            array_lowbound = -1 * day_base_weight['bm_weight'].values
            # ----------------------track error-------------------------
            overall_exp = array_overall * solve_weight / 100

            tot_risk_exp = array_risk_exp.T * solve_weight / 100
            variance = cp.quad_form(tot_risk_exp, array_risk_cov) + \
                       cp.sum_squares(cp.multiply(array_spec_risk, solve_weight / 100))
            obj = overall_exp - lmd * variance
            # -----------------------set cons---------------------------
            cons = []
            #  权重上下缘
            cons.append(solve_weight <= array_upbound)
            cons.append(solve_weight >= array_lowbound)
            cons.append(cp.sum(solve_weight) == 0)
            #  跟踪误差设置
            #cons.append(variance <= target_sigma ** 2)
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
            day_base_weight['weight'] = np.round(array_base_weight + opti_weight, 1)
            day_base_weight = day_base_weight.loc[day_base_weight['weight'] > 0.1, ['code', 'weight']]
            print('%s OPTI finish \n -n_codes: %i  n_selections: %i' %
                  (date, array_codes.shape[0], day_base_weight.shape[0]))
            dfs.append(day_base_weight)
        res_df = pd.concat(dfs)
        return [res_df, fail_dates]

    def run(self):
        start_time = datetime.datetime.now()
        self.initialize_strategy()
        self.code_range = self.code_range.loc[self.code_range['industry'] == '证券Ⅱ(中信)']
        overall_factor = self.factors_combination()
        bm_stk_wgt = self.get_bm_stk_wgt()
        target_weight = self.opt_portfolio(overall_factor, bm_stk_wgt)
        target_weight.to_csv(self.folder_dir + 'opt_wgt.csv', encoding='gbk')
        bm_stk_wgt.to_csv(self.folder_dir + 'bm_wgt.csv', encoding='gbk')
        # --------------------------backtest--------------------------------
        bt_start = target_weight.index[0].strftime('%Y%m%d')
        bt_end = target_weight.index[-1].strftime('%Y%m%d')
        QE = BacktestEngine(self.strategy_name, bt_start, bt_end, self.adj_interval, self.benchmark,
                            stock_capital=self.capital)
        pvs = []
        portfolio_value, _ = QE.run(target_weight, bt_start, bt_end)
        portfolio_value = portfolio_value.loc[:, ['TotalValue']]
        portfolio_value.rename(columns={'TotalValue': 'AlphaSec'}, inplace=True)
        pvs.append(portfolio_value)
        QE.stk_portfolio.reset_portfolio(self.capital)
        portfolio_value, _ = QE.run(bm_stk_wgt, bt_start, bt_end)
        portfolio_value = portfolio_value.loc[:, ['TotalValue']]
        portfolio_value.rename(columns={'TotalValue': 'BmSec'}, inplace=True)
        pvs.append(portfolio_value)
        sec_comparation = pd.concat(pvs, axis=1)
        sec_comparation['AccumAlpha'] = \
            DataProcess.calc_accum_alpha(sec_comparation['AlphaSec'], sec_comparation['BmSec']) - 1
        sec_comparation.to_csv(self.folder_dir + 'sec_comaration.csv', encoding='gbk')
        self.logger.info('sec Comparation:')
        self.logger.info('-ANN_Alpha: %f' % DataProcess.calc_alpha_ann_return(
            sec_comparation['AlphaSec'], sec_comparation['BmSec']))
        MDD, MDD_period = \
            DataProcess.calc_alpha_max_draw_down(sec_comparation['AlphaSec'], sec_comparation['BmSec'])
        self.logger.info('-Alpha_MDD: %f' % MDD)
        self.logger.info('-Alpha_MDD period: %s - %s' % (MDD_period[0], MDD_period[1]))
        self.logger.info('-Alpha_sharpe: %f' % DataProcess.calc_alpha_sharpe(
            sec_comparation['AlphaSec'], sec_comparation['BmSec']))
        print('Time used:', datetime.datetime.now() - start_time)



if __name__ == '__main__':
    print(datetime.datetime.now())
    a = SecTestOpt('SecTestOpt')
    kk = a.run()
    print(datetime.datetime.now())
