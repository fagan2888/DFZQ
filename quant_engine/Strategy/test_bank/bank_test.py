# 测试银行模型

import pandas as pd
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import datetime
from backtest_engine import BacktestEngine
from strategy_base import StrategyBase
from bank_CONFIG import STRATEGY_CONFIG, CATEGORY_WEIGHT, FACTOR_WEIGHT
from data_process import DataProcess
import global_constant
import os.path

class BankTest(StrategyBase):
    def __init__(self, strategy_name):
        self.save_name = ''
        for cate in CATEGORY_WEIGHT:
            for fct in FACTOR_WEIGHT[cate]:
                if self.save_name:
                    self.save_name = self.save_name + '+'
                self.save_name = self.save_name + '{0}({1})'.format(fct[2], str(fct[5] * CATEGORY_WEIGHT[cate]))
        self.folder_dir = global_constant.ROOT_DIR + 'Bank_Test/{0}/'.format(self.save_name)
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
        self.n_selection = STRATEGY_CONFIG['n_selection']
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
        folder_dir = global_constant.ROOT_DIR + 'Bank_Test/{0}/'.format(self.save_name)
        merged_df.to_csv(folder_dir + 'FactorsCombination.csv', encoding='gbk')
        print('Factors combination finish...')
        return merged_df

    def select_codes(self, df):
        dates = df.index.unique().strftime('%Y%m%d')
        split_dates = np.array_split(dates, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            parallel_res = Parallel()(delayed(BankTest.JOB_select_codes)
                                      (df, dates, self.n_selection) for dates in split_dates)
        target_weight = pd.concat(parallel_res)
        target_weight = target_weight.sort_index()
        later_dict = dict(zip(self.calendar[:-1], self.calendar[1:]))
        target_weight['date'] = target_weight.index.strftime('%Y%m%d')
        target_weight['date'] = target_weight['date'].map(later_dict)
        target_weight.set_index('date', inplace=True)
        target_weight.index = pd.to_datetime(target_weight.index)
        return target_weight

    def get_next_bm_stk_wgt(self):
        next_bm_bank = self.bm_stk_wgt.copy()
        next_bm_bank['date'] = next_bm_bank.index.strftime('%Y%m%d')
        former_dict = dict(zip(self.calendar[1:], self.calendar[:-1]))
        next_bm_bank['date'] = next_bm_bank['date'].map(former_dict)
        next_bm_bank = next_bm_bank.dropna(subset=['date'])
        next_bm_bank['date'] = pd.to_datetime(next_bm_bank['date'])
        next_bm_bank = pd.merge(next_bm_bank, self.industry_data.reset_index(), on=['date', 'code'])
        next_bm_bank = next_bm_bank.loc[next_bm_bank['industry'] == '银行(中信)', :]
        wgt_sum = next_bm_bank.groupby('date')['weight'].sum()
        wgt_sum = wgt_sum.to_dict()
        next_bm_bank['tot_wgt'] = next_bm_bank['date'].map(wgt_sum)
        next_bm_bank['weight'] = next_bm_bank['weight'] / next_bm_bank['tot_wgt'] * 100
        next_bm_bank.set_index('date', inplace=True)
        next_bm_bank = next_bm_bank.loc[:, ['code', 'weight']]
        return next_bm_bank

    @staticmethod
    def JOB_select_codes(df, dates, n_selection):
        dfs = []
        for date in dates:
            day_df = df.loc[date, :]
            day_df = day_df.sort_values('overall', ascending=False).head(n_selection)
            day_df['weight'] = 100 / n_selection
            dfs.append(day_df)
        res_df = pd.concat(dfs)
        return res_df

    def run(self):
        start_time = datetime.datetime.now()
        self.initialize_strategy()
        self.code_range = self.code_range.loc[self.code_range['industry'] == '银行(中信)']
        self.code_range.reset_index(inplace=True)
        overall_factor = self.factors_combination()
        selection = self.select_codes(overall_factor)
        selection.to_csv(self.folder_dir + 'code_selection.csv', encoding='gbk')
        bm_stk_wgt = self.get_next_bm_stk_wgt()
        bm_stk_wgt.to_csv(self.folder_dir + 'bm_wgt.csv', encoding='gbk')
        # --------------------------backtest--------------------------------
        bt_start = selection.index[0].strftime('%Y%m%d')
        bt_end = bm_stk_wgt.index[-1].strftime('%Y%m%d')
        QE = BacktestEngine(self.strategy_name, bt_start, bt_end, self.adj_interval, self.benchmark,
                            stock_capital=self.capital)
        pvs = []
        portfolio_value = QE.run(selection, bt_start, bt_end)
        portfolio_value = portfolio_value.loc[:, ['TotalValue']]
        portfolio_value.rename(columns={'TotalValue': 'AlphaBank'}, inplace=True)
        pvs.append(portfolio_value)
        QE.stk_portfolio.reset_portfolio(self.capital)
        portfolio_value = QE.run(bm_stk_wgt, bt_start, bt_end)
        portfolio_value = portfolio_value.loc[:, ['TotalValue']]
        portfolio_value.rename(columns={'TotalValue': 'BmBank'}, inplace=True)
        pvs.append(portfolio_value)
        banks_comparation = pd.concat(pvs, axis=1)
        banks_comparation['AccumAlpha'] = \
            DataProcess.calc_accum_alpha(banks_comparation['AlphaBank'], banks_comparation['BmBank']) - 1
        banks_comparation.to_csv(self.folder_dir + 'banks_comaration.csv', encoding='gbk')
        self.logger.info('Bank Comparation:')
        self.logger.info('-ANN_Alpha: %f' % DataProcess.calc_alpha_ann_return(
            banks_comparation['AlphaBank'], banks_comparation['BmBank']))
        MDD, MDD_period = \
            DataProcess.calc_alpha_max_draw_down(banks_comparation['AlphaBank'], banks_comparation['BmBank'])
        self.logger.info('-Alpha_MDD: %f' % MDD)
        self.logger.info('-Alpha_MDD period: %s - %s' % (MDD_period[0], MDD_period[1]))
        self.logger.info('-Alpha_sharpe: %f' % DataProcess.calc_alpha_sharpe(
            banks_comparation['AlphaBank'], banks_comparation['BmBank']))
        print('Time used:', datetime.datetime.now() - start_time)



if __name__ == '__main__':
    print(datetime.datetime.now())
    a = BankTest('BankTest_new')
    kk = a.run()
    print(datetime.datetime.now())
