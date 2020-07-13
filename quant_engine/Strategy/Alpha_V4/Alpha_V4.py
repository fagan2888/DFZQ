import global_constant
from strategy_base import StrategyBase
from Alpha_V4_CONFIG import *
import pandas as pd
import numpy as np
from data_process import DataProcess
from joblib import Parallel, delayed, parallel_backend
import datetime
import cvxpy as cp
from backtest_engine import BacktestEngine
import os.path


class Alpha_V4(StrategyBase):
    def __init__(self, strategy_name):
        super().__init__(strategy_name)
        start = STRATEGY_CONFIG['start']
        end = STRATEGY_CONFIG['end']
        benchmark = STRATEGY_CONFIG['benchmark']
        select_range = STRATEGY_CONFIG['select_range']
        industry = STRATEGY_CONFIG['industry']
        adj_interval = STRATEGY_CONFIG['adj_interval']
        super().initialize_strategy(start, end, benchmark, select_range, industry, adj_interval)
        self.folder_dir = global_constant.ROOT_DIR + 'Strategy_Result/{0}/'.format(self.strategy_name)
        if os.path.exists(self.folder_dir):
            pass
        else:
            os.makedirs(self.folder_dir.rstrip('/'))
        self.init_log()
        self.logger.info('Strategy start time: %s' % datetime.datetime.now().strftime('%Y/%m/%d - %H:%M:%S'))
        self.logger.info('*' * 50)
        self.capital = STRATEGY_CONFIG['capital']
        self.lmd = STRATEGY_CONFIG['lambda']
        self.non_fin_const = STRATEGY_CONFIG['non_fin_const']
        self.non_fin_ratio = STRATEGY_CONFIG['non_fin_ratio']
        self.bank_const = STRATEGY_CONFIG['bank_const']
        self.bank_ratio = STRATEGY_CONFIG['bank_ratio']
        self.sec_const = STRATEGY_CONFIG['sec_const']
        self.sec_ratio = STRATEGY_CONFIG['sec_ratio']

    def factors_combination(self, category_weight, factor_weight, industry):
        # 单行业的需要重新设定 code_range
        tmp_code_range = self.code_range.copy()
        if industry:
            self.code_range = self.code_range.loc[self.code_range['industry'] == industry, :]
        else:
            self.code_range = self.code_range.loc[~self.code_range['industry'].isin(['银行(中信)', '证券Ⅱ(中信)'])]
        self.logger.info('COMBINE PARAMETERS: ')
        cates = []
        for cate in category_weight.keys():
            self.logger.info('-%s  weight: %i' % (cate, category_weight[cate]))
            parameters_list = factor_weight[cate]
            fcts_in_cate = []
            for db, measure, factor, direction, fillna, weight, check_rp, neutralize in parameters_list:
                self.logger.info('  -Factor: %s   DIR:%i,   FILLNA:%s,   CHECKRP:%s,   NEUTRALIZE:%s' %
                                 (factor, direction, str(fillna), str(check_rp), str(neutralize)))
                self.logger.info('  -Factor: %s   weight:%i' % (factor, weight))
                fct_df = self.process_factor(db, measure, factor, direction, fillna, check_rp, neutralize)
                fct_df[factor] = fct_df[factor] * weight
                fct_df.set_index(['date', 'code'], inplace=True)
                fcts_in_cate.append(fct_df)
            cate_df = pd.concat(fcts_in_cate, join='inner', axis=1)
            cate_df[cate] = cate_df.sum(axis=1)
            cate_df = cate_df.reset_index()
            cate_df = DataProcess.standardize(cate_df, cate, False, self.n_jobs)
            cate_df[cate] = category_weight[cate] * cate_df[cate]
            cate_df.set_index(['date', 'code'], inplace=True)
            cates.append(cate_df)
        merged_df = pd.concat(cates, join='inner', axis=1)
        merged_df['overall'] = merged_df[list(category_weight.keys())].sum(axis=1)
        merged_df = merged_df.reset_index()
        merged_df = pd.merge(merged_df, self.industry_data.reset_index(), how='inner', on=['date', 'code'])
        merged_df = merged_df.set_index('date')
        self.code_range = tmp_code_range
        print('Factors combination finish...')
        return merged_df

    def get_bm_idsty_wgt(self, industry):
        bm_idsty = pd.merge(self.bm_stk_wgt.reset_index(), self.industry_data.reset_index(), on=['date', 'code'])
        if industry:
            bm_idsty = bm_idsty.loc[bm_idsty['industry'] == industry, :]
            wgt_sum = bm_idsty.groupby('date')['weight'].sum()
            wgt_sum = wgt_sum.to_dict()
            bm_idsty['tot_wgt'] = bm_idsty['date'].map(wgt_sum)
            bm_idsty['weight'] = bm_idsty['weight'] / bm_idsty['tot_wgt'] * 100
        else:
            bm_idsty = bm_idsty.loc[~bm_idsty['industry'].isin(['银行(中信)', '证券Ⅱ(中信)']), :]
        bm_idsty.set_index('date', inplace=True)
        return bm_idsty

    def opt_weight(self, factor_df, benchmark_df, const, ratio):
        later_dict = dict(zip(self.calendar[:-1], self.calendar[1:]))
        factor_df['next_date'] = factor_df.index.strftime('%Y%m%d')
        factor_df['next_date'] = factor_df['next_date'].map(later_dict)
        bm_df = benchmark_df.copy()
        bm_df.rename(columns={'weight': 'bm_weight'}, inplace=True)
        bm_df['next_date'] = bm_df.index.strftime('%Y%m%d')
        bm_stk_wgt = bm_df.loc[:, ['next_date', 'code', 'bm_weight']].copy()
        merge = pd.merge(factor_df.reset_index(), bm_stk_wgt, how='outer', on=['next_date', 'code'])
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
        merge['specific_risk'] = merge['specific_risk'].fillna(0)
        merge = merge.drop('date', axis=1)
        merge.set_index('next_date', inplace=True)
        merge.index = pd.to_datetime(merge.index)
        merge.index.names = ['date']
        dates = merge.index.unique().strftime('%Y%m%d')
        split_dates = np.array_split(dates, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            parallel_res = Parallel()(delayed(Alpha_V4.JOB_opt_weight)
                                      (dates, merge, self.risks, self.risk_cov, self.lmd, const, ratio)
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

    # 工具函数，优化金融行业权重
    @staticmethod
    def JOB_opt_weight(dates, base_weight, risks_field, risk_cov, lmd, const, ratio):
        dfs = []
        fail_dates = []
        indus_field = risks_field.difference(['Beta', 'Cubic size', 'Growth', 'Liquidity', 'Market', 'SOE', 'Size',
                                             'Trend', 'Uncertainty', 'Value', 'Volatility'])
        for date in dates:
            day_base_weight = base_weight.loc[date, :].copy()
            # -----------------------get array--------------------------
            array_codes = day_base_weight['code'].values
            array_overall = day_base_weight['overall'].values
            array_base_weight = day_base_weight['bm_weight'].values
            array_risk_exp = day_base_weight.loc[:, risks_field].values
            array_indu = day_base_weight.loc[:, indus_field].values
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
            # 行业内绝对权重 max(const, (1+ratio) * base_weight)
            array_upbound = np.where(const - array_base_weight > ratio * array_base_weight, const - array_base_weight,
                                     ratio * array_base_weight)
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
            #  行业中性
            for i in range(len(indus_field)):
                cons.append(cp.sum(array_indu[:, i].T * solve_weight) == 0)
            #  跟踪误差设置
            # cons.append(variance <= target_sigma ** 2)
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

    def run(self):
        start_time = datetime.datetime.now()
        # -----------------------------get data------------------------------
        # 银行因子
        print('BANK Factors processing...')
        overall_bank = self.factors_combination(CATEGORY_BANK, FACTOR_BANK, '银行(中信)')
        overall_bank.to_csv(self.folder_dir + 'factors_bank.csv', encoding='gbk')
        overall_bank = overall_bank.loc[:, ['code', 'overall', 'industry']]
        bm_bank = self.get_bm_idsty_wgt('银行(中信)')
        # 券商因子
        print('SEC Factors processing...')
        overall_sec = self.factors_combination(CATEGORY_SEC, FACTOR_SEC, '证券Ⅱ(中信)')
        overall_sec.to_csv(self.folder_dir + 'factors_sec.csv', encoding='gbk')
        overall_sec = overall_sec.loc[:, ['code', 'overall', 'industry']]
        bm_sec = self.get_bm_idsty_wgt('证券Ⅱ(中信)')
        # 非金融因子
        print('NON FIN Factors processing...')
        overall_non_fin = self.factors_combination(CATEGORY_NON_FIN, FACTOR_NON_FIN, None)
        overall_non_fin.to_csv(self.folder_dir + 'factors_non_fin.csv', encoding='gbk')
        overall_non_fin = overall_non_fin.loc[:, ['code', 'overall', 'industry']]
        bm_non_fin = self.get_bm_idsty_wgt(None)
        print('ALL Factors finish!')
        # ---------------------------opt_weight------------------------------
        # 银行权重
        print('BANK Opt processing...')
        opt_bank_wgt = self.opt_weight(overall_bank, bm_bank, self.bank_const, self.bank_ratio)
        bank_tot_wgt = dict(zip(bm_bank.index.unique(), bm_bank['tot_wgt'].unique()))
        opt_bank_wgt['bank_wgt'] = opt_bank_wgt.index
        opt_bank_wgt['bank_wgt'] = opt_bank_wgt['bank_wgt'].map(bank_tot_wgt)
        opt_bank_wgt['weight'] = opt_bank_wgt['weight'] / 100 * opt_bank_wgt['bank_wgt']
        # opt_bank_wgt.to_csv(self.folder_dir + 'bank_weight.csv', encoding='gbk')
        # 券商权重
        print('SEC Opt processing...')
        # 行业内权重
        opt_sec_wgt = self.opt_weight(overall_sec, bm_sec, self.sec_const, self.sec_ratio)
        sec_tot_wgt = dict(zip(bm_sec.index.unique(), bm_sec['tot_wgt'].unique()))
        opt_sec_wgt['sec_wgt'] = opt_sec_wgt.index
        opt_sec_wgt['sec_wgt'] = opt_sec_wgt['sec_wgt'].map(sec_tot_wgt)
        opt_sec_wgt['weight'] = opt_sec_wgt['weight'] / 100 * opt_sec_wgt['sec_wgt']
        # 非金融权重
        print('NON FIN Opt processing...')
        opt_non_fin_wgt = self.opt_weight(overall_non_fin, bm_non_fin, self.non_fin_const, self.non_fin_ratio)
        # 组合权重
        target_weight = pd.concat(
            [opt_bank_wgt.reset_index(), opt_sec_wgt.reset_index(), opt_non_fin_wgt.reset_index()], ignore_index=True)
        target_weight = target_weight.set_index('date')
        target_weight.to_csv(self.folder_dir + 'target_weight.csv', encoding='gbk')
        print('target_weight is ready!')
        # --------------------------backtest--------------------------------
        bt_start = target_weight.index[0].strftime('%Y%m%d')
        bt_end = (target_weight.index[-1]).strftime('%Y%m%d')
        QE = BacktestEngine(self.strategy_name, bt_start, bt_end, self.adj_interval, self.benchmark,
                            stock_capital=self.capital)
        portfolio_value, _ = QE.run(target_weight, bt_start, bt_end)
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
    a = Alpha_V4('first_run')
    kk = a.run()
    print(datetime.datetime.now())
