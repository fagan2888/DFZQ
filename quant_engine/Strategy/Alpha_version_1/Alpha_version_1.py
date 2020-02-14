from strategy_base import StrategyBase
from Alpha_version_1_CONFIG import STRATEGY_CONFIG, CATEGORY_WEIGHT, FACTOR_WEIGHT
from industry_neutral_engine import IndustryNeutralEngine
import pandas as pd
import numpy as np
from data_process import DataProcess
from joblib import Parallel, delayed, parallel_backend


class alpha_version_1(StrategyBase):
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

    def factors_combination(self):
        categorys = []
        for category in FACTOR_WEIGHT.keys():
            print('Category: %s is processing...' %category)
            parameters_list = FACTOR_WEIGHT[category]
            factors_in_same_category = []
            for measure, factor, direction, if_fillna, weight in parameters_list:
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
            parallel_res = Parallel()(delayed(alpha_version_1.JOB_select)
                                      (industry_count, overall_factor, dates) for dates in split_dates)
        selections = pd.concat(parallel_res)
        selections = selections.sort_values('date')
        return selections

    def get_stk_weight(self):
        selections = self.select_code()
        free_market_cap = \
            self.influx.getDataMultiprocess(self.factor_db, 'Size', self.start, self.end, ['code', 'free_market_cap'])
        free_market_cap.index.names = ['date']
        free_market_cap.reset_index(inplace=True)
        selections = pd.merge(selections, free_market_cap, on=['date', 'code'])
        industry_size_sum = selections.groupby(['date', 'industry'])['free_market_cap'].sum()
        industry_size_sum = pd.DataFrame(industry_size_sum).reset_index()
        industry_size_sum.rename(columns={'free_market_cap': 'industry_size_sum'}, inplace=True)
        selections = pd.merge(selections, industry_size_sum, on=['date', 'industry'])
        selections['weight_in_industry'] = selections['free_market_cap'] / selections['industry_size_sum'] * 100
        selections = selections.loc[:, ['date', 'code', 'industry', 'weight_in_industry']]
        selections.set_index('date', inplace=True)
        dates = selections.index.unique()
        calendar = self.rdf.get_trading_calendar()
        next_trade_date = {}
        for date in dates:
            next_trade_date.update(DataProcess.get_next_date(calendar, date, 1))
        selections['next_1_day'] = selections.apply(lambda row: next_trade_date[row.name], axis=1)
        selections.set_index('next_1_day', inplace=True)
        return selections

    def run(self):
        self.initialize_strategy()
        weight = self.get_stk_weight()
        engine = IndustryNeutralEngine(stock_capital=self.capital, save_name='Alpha_version_1')
        bt_start = weight.index[0].strftime('%Y%m%d')
        bt_end = weight.index[-2].strftime('%Y%m%d')
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
        portfolio_value.to_csv('pv.csv', encoding='gbk')
        print('ANN_return: ', DataProcess.calc_ann_return(portfolio_value['TotalValue']))
        print('Alpha300: ',
              DataProcess.calc_alpha_ann_return(portfolio_value['TotalValue'], portfolio_value['000300.SH']))
        print('Alpha500: ',
              DataProcess.calc_alpha_ann_return(portfolio_value['TotalValue'], portfolio_value['000905.SH']))
        print('Alpha800: ',
              DataProcess.calc_alpha_ann_return(portfolio_value['TotalValue'], portfolio_value['300+500']))

if __name__ == '__main__':
    a = alpha_version_1('Alpha_version_1')
    kk = a.run()