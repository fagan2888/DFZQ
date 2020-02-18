from strategy_base import StrategyBase
from Alpha_version_2_CONFIG import STRATEGY_CONFIG, CATEGORY_WEIGHT, FACTOR_WEIGHT
from industry_neutral_engine import IndustryNeutralEngine
from global_constant import N_JOBS
import pandas as pd
import numpy as np
from data_process import DataProcess
from joblib import Parallel, delayed, parallel_backend
import datetime


class alpha_version_2(StrategyBase):
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
        self.select_pct = STRATEGY_CONFIG['select_pct']
        self.capital = STRATEGY_CONFIG['capital']
        self.calendar = self.rdf.get_trading_calendar()
        self.weight_in_industry_threshold = STRATEGY_CONFIG['weight_in_industry_threshold']


    def factors_combination(self):
        categorys = []
        for category in FACTOR_WEIGHT.keys():
            print('Category: %s is processing...' %category)
            parameters_list = FACTOR_WEIGHT[category]
            factors_in_same_category = []
            for measure, factor, direction, if_fillna, weight in parameters_list:
                print('-Factor: %s is processing...' %factor)
                factor_df = self.process_factor(measure, factor, direction, if_fillna)
                if weight == 1:
                    pass
                else:
                    factor_df[factor] = factor_df[factor] * weight
                factor_df.set_index(['date', 'code'], inplace=True)
                factors_in_same_category.append(factor_df)
            category_df = pd.concat(factors_in_same_category, join='inner', axis=1)
            category_df[category] = category_df.sum(axis=1)
            category_df.reset_index(inplace=True)
            category_df = category_df.loc[:, ['date', 'code', category]]
            category_df = DataProcess.remove_and_Z(category_df, category, False, self.n_jobs)
            category_df[category] = CATEGORY_WEIGHT[category] * category_df[category]
            category_df.set_index(['date', 'code'], inplace=True)
            categorys.append(category_df)
        merged_df = pd.concat(categorys, join='inner', axis=1)
        merged_df['overall'] = merged_df.sum(axis=1)
        merged_df.reset_index(inplace=True)
        merged_df = pd.merge(merged_df, self.code_range.reset_index(), on=['date', 'code'])
        merged_df.rename(columns={self.industry: 'industry'}, inplace=True)
        print('Factors combination finish...')
        return merged_df

    def industry_count(self):
        if self.benchmark == 300:
            bm_industries = \
                self.mkt_data.loc[pd.notnull(self.mkt_data[self.industry]) & pd.notnull(self.mkt_data['IF_weight']),
                                  ['code', self.industry]].copy()
        elif self.benchmark == 500:
            bm_industries = \
                self.mkt_data.loc[pd.notnull(self.mkt_data[self.industry]) & pd.notnull(self.mkt_data['IC_weight']),
                                  ['code', self.industry]].copy()
        elif self.benchmark == 800:
            bm_industries = \
                self.mkt_data.loc[pd.notnull(self.mkt_data[self.industry]) &
                                  (pd.notnull(self.mkt_data['IF_weight']) | pd.notnull(self.mkt_data['IC_weight'])),
                                  ['code', self.industry]].copy()
        else:
            print('Benchmark not found!')
            raise NameError
        industry_count = bm_industries.groupby(['date', self.industry])['code'].count()
        industry_count = pd.DataFrame(industry_count).reset_index()
        industry_count.rename(columns={'code': 'count', self.industry: 'industry'}, inplace=True)
        industry_count['to_select'] = industry_count['count'] * self.select_pct
        # 每个行业至少选1个
        industry_count['to_select'] = \
            industry_count['to_select'].apply(lambda x: 0 if x == 0 else 1 if x <= 0.5 else round(x))
        return industry_count

    # select_code 的工具函数
    @staticmethod
    def JOB_select(industry_count, overall_factor, dates):
        res = []
        for date in dates:
            day_industry = industry_count.loc[industry_count['date'] == date, :].copy()
            day_overall = overall_factor.loc[overall_factor['date'] == date, :].copy()
            for idx, row in day_industry.iterrows():
                indu = row['industry']
                n_select = int(row['to_select'])
                res.append(day_overall.loc[day_overall['industry'] == indu, :].head(n_select))
        res = pd.concat(res)
        return res

    def select_code(self):
        overall_factor = self.factors_combination()
        overall_factor = overall_factor.sort_values('overall', ascending=False)
        industry_count = self.industry_count()
        dates = industry_count['date'].unique()
        split_dates = np.array_split(dates, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            parallel_res = Parallel()(delayed(alpha_version_2.JOB_select)
                                      (industry_count, overall_factor, dates) for dates in split_dates)
        selections = pd.concat(parallel_res)
        selections = selections.sort_values('date')
        return selections

    @staticmethod
    def JOB_get_stk_weight(df, dates, threshold):
        date_dfs = []
        for date in dates:
            date_df = df.loc[df['date'] == date, :].copy()
            indus = date_df['industry'].unique()
            indu_dfs = []
            for indu in indus:
                date_indu_df = date_df.loc[date_df['industry'] == indu, :].copy()
                n_stks = date_indu_df.shape[0]
                # 原始权重 >= max(预设阈值, 等权权重),  原始权重保留
                over_threshold_weight = date_indu_df.loc[
                    date_indu_df['weight_in_industry'] >= max(threshold, 100 / n_stks), 'weight_in_industry']
                # 如果所选票均符合原始权重的规则， 该行业等权
                if n_stks - over_threshold_weight.shape[0] == 0:
                    date_indu_df.loc[:, 'weight_in_industry'] = 100 / n_stks
                # 剩余权重被均分
                else:
                    date_indu_df.loc[pd.isnull(date_indu_df['weight_in_industry']) |
                                     (date_indu_df['weight_in_industry'] < max(threshold, 100 / n_stks)),
                                     'weight_in_industry'] = \
                        (100 - over_threshold_weight.sum()) / (n_stks - over_threshold_weight.shape[0])
                indu_dfs.append(date_indu_df)
            date_df = pd.concat(indu_dfs)
            date_dfs.append(date_df)
        res_df = pd.concat(date_dfs)
        return res_df


    def get_stk_weight(self):
        bm_dict = {300: 'IF_weight', 500: 'IC_weight'}
        bm_stk_weight = self.mkt_data.loc[pd.notnull(self.mkt_data[bm_dict[self.benchmark]]),
                                      ['code', self.industry, bm_dict[self.benchmark]]].copy()
        bm_stk_weight.rename(columns={self.industry: 'industry', bm_dict[self.benchmark]: 'benchmark_weight'}, inplace=True)
        bm_stk_weight.reset_index(inplace=True)
        bm_indu_weight = bm_stk_weight.groupby(['date', 'industry'])['benchmark_weight'].sum()
        bm_indu_weight = pd.DataFrame(bm_indu_weight).reset_index()
        bm_indu_weight.rename(columns={'benchmark_weight': 'bm_indu_weight'}, inplace=True)
        bm_stk_weight = pd.merge(bm_stk_weight, bm_indu_weight, on=['date', 'industry'])
        bm_stk_weight['weight_in_industry'] = bm_stk_weight['benchmark_weight'] / bm_stk_weight['bm_indu_weight'] * 100
        bm_stk_weight = bm_stk_weight.loc[:, ['date', 'code', 'weight_in_industry']]
        selections = self.select_code()
        selections = pd.merge(selections, bm_stk_weight, on=['date', 'code'], how='left')
        selections = selections.loc[:, ['date', 'code', 'industry', 'weight_in_industry']]
        dates = selections['date'].unique()
        split_dates = np.array_split(dates, N_JOBS)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            parallel_res = Parallel()(delayed(alpha_version_2.JOB_get_stk_weight)
                                      (selections, dates, self.weight_in_industry_threshold) for dates in split_dates)
        selections = pd.concat(parallel_res)
        selections.set_index('date', inplace=True)
        next_trade_date = {}
        for date in dates:
            next_trade_date.update(DataProcess.get_next_date(self.calendar, date, 1))
        selections['next_1_day'] = selections.apply(lambda row: next_trade_date[row.name], axis=1)
        selections.set_index('next_1_day', inplace=True)
        return selections


    def run(self):
        self.initialize_strategy()
        weight = self.get_stk_weight()
        weight.to_csv(self.strategy_name + '_TARGET_WEIGHT.csv', encoding='gbk')
        engine = IndustryNeutralEngine(stock_capital=self.capital, save_name='Alpha_version_2')
        bt_start = weight.index[0].strftime('%Y%m%d')
        bt_end = (weight.index[-1] - datetime.timedelta(days=1)).strftime('%Y%m%d')
        portfolio_value = engine.run(weight, bt_start, bt_end, benchmark='IF', adj_interval=5, data_input=self.mkt_data)
        portfolio_value = portfolio_value.loc[:, ['Balance', 'StockValue', 'TotalValue']]
        comparing_index = ['000300.SH', '000905.SH']
        index_value = self.mkt_data.loc[self.mkt_data['code'].isin(comparing_index), ['code', 'close']]
        index_value.set_index([index_value.index, 'code'], inplace=True)
        index_value = index_value.unstack()['close']
        index_value['300+500'] = 0.5 * index_value['000300.SH'] + 0.5 * index_value['000905.SH']
        for col in index_value.columns:
            index_value[col] = index_value[col] / index_value[col].iloc[0] * self.capital
        portfolio_value = pd.merge(portfolio_value, index_value, left_index=True, right_index=True)
        portfolio_value['accum_alpha_300'] = \
            DataProcess.calc_accum_alpha(portfolio_value['TotalValue'], portfolio_value['000300.SH'])
        portfolio_value['accum_alpha_500'] = \
            DataProcess.calc_accum_alpha(portfolio_value['TotalValue'], portfolio_value['000905.SH'])
        portfolio_value['accum_alpha_800'] = \
            DataProcess.calc_accum_alpha(portfolio_value['TotalValue'], portfolio_value['300+500'])
        portfolio_value.to_csv(self.strategy_name + '_backtest.csv', encoding='gbk')
        
        self.logger.info('Backtest finish time: %s' % datetime.datetime.now().strftime('%Y/%m/%d - %H:%M:%S'))
        self.logger.info('*' * 50)
        self.logger.info('PERFORMANCE:')
        self.logger.info('-ANN_return: %f' % DataProcess.calc_ann_return(portfolio_value['TotalValue']))
        self.logger.info('-MDD: %f' % DataProcess.calc_max_draw_down(portfolio_value['TotalValue']))
        self.logger.info('-sharpe: %f' % DataProcess.calc_sharpe(portfolio_value['TotalValue']))
        self.logger.info('-ANN_Alpha300: %f' % DataProcess.calc_alpha_ann_return(portfolio_value['TotalValue'],
                                                                                 portfolio_value['000300.SH']))
        self.logger.info('-Alpha300_MDD: %f' % DataProcess.calc_alpha_max_draw_down(portfolio_value['TotalValue'],
                                                                                    portfolio_value['000300.SH']))
        self.logger.info('-Alpha300_sharpe: %f' % DataProcess.calc_alpha_sharpe(portfolio_value['TotalValue'],
                                                                                portfolio_value['000300.SH']))
        self.logger.info('-ANN_Alpha500: %f' % DataProcess.calc_alpha_ann_return(portfolio_value['TotalValue'],
                                                                                 portfolio_value['000905.SH']))
        self.logger.info('-Alpha500_MDD: %f' % DataProcess.calc_alpha_max_draw_down(portfolio_value['TotalValue'],
                                                                                    portfolio_value['000905.SH']))
        self.logger.info('-Alpha500_sharpe: %f' % DataProcess.calc_alpha_sharpe(portfolio_value['TotalValue'],
                                                                                portfolio_value['000905.SH']))
        self.logger.info('-ANN_Alpha800: %f' % DataProcess.calc_alpha_ann_return(portfolio_value['TotalValue'],
                                                                                 portfolio_value['300+500']))
        self.logger.info('-Alpha800_MDD: %f' % DataProcess.calc_alpha_max_draw_down(portfolio_value['TotalValue'],
                                                                                    portfolio_value['300+500']))
        self.logger.info('-Alpha800_sharpe: %f' % DataProcess.calc_alpha_sharpe(portfolio_value['TotalValue'],
                                                                                portfolio_value['300+500']))
        

if __name__ == '__main__':
    print(datetime.datetime.now())
    a = alpha_version_2('Alpha_version_2')
    kk = a.run()
    print(datetime.datetime.now())