import global_constant
import pandas as pd
import numpy as np
from rdf_data import rdf_data
from influxdb_data import influxdbData
from data_process import DataProcess
from joblib import Parallel,delayed,parallel_backend
import cvxpy as cp
from industry_neutral_engine import IndustryNeutralEngine
import datetime
import os


class StrategyBase:
    def __init__(self):
        self.influx = influxdbData()
        self.rdf = rdf_data()
        self.factor_db = 'DailyFactor_Gus'
        self.root_dir = global_constant.ROOT_DIR + 'Strategy/'

    # 获得行情信息以及选股的范围
    # 行情信息为全市场，以免吸收合并出现没有行情的情况
    def data_prepare(self):
        self.mkt_data = self.influx.getDataMultiprocess('DailyData_Gus', 'marketData', self.start, self.end, None)
        if not self.range:
            self.code_range = self.mkt_data.loc[
                pd.notnull(self.mkt_data[self.industry]) &
                ~((self.mkt_data['status'] == '停牌') | pd.isnull(self.mkt_data['status'])),
                ['code', self.industry]].copy()
        elif self.range == 300:
            self.code_range = self.mkt_data.loc[
                pd.notnull(self.mkt_data[self.industry]) &
                pd.notnull(self.mkt_data['IF_weight']) &
                ~((self.mkt_data['status'] == '停牌') | pd.isnull(self.mkt_data['status'])),
                ['code', self.industry]].copy()
        elif self.range == 500:
            self.code_range = self.mkt_data.loc[
                pd.notnull(self.mkt_data[self.industry]) &
                pd.notnull(self.mkt_data['IC_weight']) &
                ~((self.mkt_data['status'] == '停牌') | pd.isnull(self.mkt_data['status'])),
                ['code', self.industry]].copy()
        elif self.range == 800:
            self.code_range = self.mkt_data.loc[
                pd.notnull(self.mkt_data[self.industry]) &
                (pd.notnull(self.mkt_data['IF_weight']) | pd.notnull(self.mkt_data['IC_weight'])) &
                ~((self.mkt_data['status'] == '停牌') | pd.isnull(self.mkt_data['status'])),
                ['code', self.industry]].copy()
        else:
            print('无效的range')
            raise NameError
        self.code_range.index.names = ['date']
        self.code_range.reset_index(inplace=True)
        self.size_data = \
            self.influx.getDataMultiprocess('DailyFactor_Gus', 'Size', self.start, self.end, ['code', self.size_field])
        self.size_data.index.names = ['date']
        self.industry_dummies = DataProcess.get_industry_dummies(self.code_range.set_index('date'), self.industry)
        print('all data are loaded! start processing...')
        print('-'*30)

    # factor 的输入: {因子大类(measurement): {因子名称(field): 因子方向}}
    # factor 处理完的输出 {因子大类(measurement): {因子名称(field): 方向调整后且正交化后的dataframe}}
    def process_factor(self):
        m_res = []
        for m in self.factor_weight_dict.keys():
            f_res = []
            for f in self.factor_weight_dict[m].keys():
                raw_df = self.influx.getDataMultiprocess(self.factor_db, m, self.start, self.end, ['code', f])
                raw_df.index.names = ['date']
                raw_df = raw_df.groupby(['date', 'code']).last()
                raw_df.reset_index(inplace=True)
                raw_in_range = pd.merge(raw_df, self.code_range, how='right', on=['date', 'code'])
                # 缺失的因子用行业中位数代替
                raw_in_range[f] = raw_in_range.groupby(['date', self.industry])[f].apply(lambda x: x.fillna(x.median()))
                raw_in_range.set_index('date', inplace=True)
                # 进行remove outlier, z score和中性化
                neutralized_df = DataProcess.neutralize(raw_in_range, f, self.industry_dummies.copy(),
                                                        self.size_data.copy(), self.size_field, self.n_jobs)
                neutralized_df[f] = neutralized_df[f] * self.factor_weight_dict[m][f]
                neutralized_df.set_index(['date', 'code'], inplace=True)
                f_res.append(neutralized_df)
                print('Factor: %s(%s) neutralization finish!' % (f, m))
            category = pd.concat(f_res, join='inner', axis=1)
            category[m] = category.sum(axis=1)
            category.reset_index(inplace=True)
            dates = category['date'].unique()
            split_dates = np.array_split(dates, self.n_jobs)
            with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
                parallel_res = Parallel()(delayed(DataProcess.JOB_cross_section_remove_and_Z)
                                          (category, m, dates) for dates in split_dates)
            category = pd.concat(parallel_res)
            category = category.loc[:, ['date', 'code', m]]
            category.set_index(['date', 'code'], inplace=True)
            m_res.append(category)
            print('Category: %s process finish!' %m)
        self.factor_overall = pd.concat(m_res, join='inner', axis=1)
        self.factor_overall['overall'] = self.factor_overall.sum(axis=1)
        self.factor_overall.reset_index(inplace=True)
        print('factor processing finish...')
        print('-'*30)
        return self.factor_overall

    def select_stks(self):
        if self.benchmark == 300:
            bm_industy_distribution = \
                self.mkt_data.loc[pd.notnull(self.mkt_data[self.industry]) & pd.notnull(self.mkt_data['IF_weight']),
                                  ['code', self.industry]].copy()
        elif self.benchmark == 500:
            bm_industy_distribution = \
                self.mkt_data.loc[pd.notnull(self.mkt_data[self.industry]) & pd.notnull(self.mkt_data['IC_weight']),
                                  ['code', self.industry]].copy()
        else:
            print('Benchmark not found!')
            raise NameError
        bm_industy_distribution.index.names = ['date']
        industry_count = bm_industy_distribution.groupby(['date', self.industry])['code'].count()
        industry_count = pd.DataFrame(industry_count).reset_index()
        industry_count.rename(columns={'code': 'count', self.industry: 'industry'}, inplace=True)
        industry_count['to_select'] = industry_count['count'] * self.select_pct
        industry_count['to_select'] = \
            industry_count['to_select'].apply(lambda x: 0 if x == 0 else 1 if x <= 0.5 else round(x))
        merge = pd.merge(self.factor_overall, self.code_range, on=['date', 'code'])
        merge.rename(columns={self.industry: 'industry'}, inplace=True)
        merge = merge.sort_values('overall', ascending=False)
        dates = merge['date'].unique()
        split_dates = np.array_split(dates, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            parallel_res = Parallel()(delayed(StrategyBase.JOB_select)
                                      (industry_count, merge, dates) for dates in split_dates)
        self.selections = pd.concat(parallel_res)
        print('stocks selections finish...')
        print('-'*30)
        return self.selections

    def opti_weight(self):
        # 市值因子标准化
        size = self.size_data.reset_index()
        dates = size['date'].unique()
        split_dates = np.array_split(dates, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            parallel_res = Parallel()(delayed(DataProcess.JOB_cross_section_Z_score)
                                      (size, self.size_field, dates) for dates in split_dates)
        size_Z = pd.concat(parallel_res)
        # 计算benchmark的市值暴露
        if self.benchmark == 300:
            bm_stk_weight = \
                self.mkt_data.loc[pd.notnull(self.mkt_data['IF_weight']), ['code', 'IF_weight', self.industry]].copy()
            bm_stk_weight.rename(columns={'IF_weight': 'weight', self.industry: 'industry'}, inplace=True)
        elif self.benchmark == 500:
            bm_stk_weight = \
                self.mkt_data.loc[pd.notnull(self.mkt_data['IC_weight']), ['code', 'IC_weight', self.industry]].copy()
            bm_stk_weight.rename(columns={'IC_weight': 'weight', self.industry: 'industry'}, inplace=True)
        else:
            print('Benchmark not found!')
            raise NameError
        dates = bm_stk_weight.index.unique()
        calendar = self.rdf.get_trading_calendar()
        next_date_dict = {}
        for d in dates:
            next_date_dict.update({d: calendar[calendar > d].iloc[0]})
        bm_stk_weight.index.names = ['date']
        bm_stk_weight.reset_index(inplace=True)
        # 获取下一交易日的指数权重
        bm_stk_weight['next_date'] = bm_stk_weight['date'].apply(lambda x: next_date_dict[x])
        nxt_bm_stk_weight = bm_stk_weight.loc[:, ['date', 'code', 'weight']].copy()
        nxt_bm_stk_weight.rename(columns={'date': 'next_date', 'weight': 'next_weight'}, inplace=True)
        bm_stk_weight = pd.merge(bm_stk_weight, nxt_bm_stk_weight, on=['next_date', 'code'])
        # 计算下一交易日的市值暴露
        bm_stk_weight = pd.merge(bm_stk_weight, size_Z, on=['date', 'code'])
        bm_stk_weight['size_exposure'] = bm_stk_weight['next_weight'] * bm_stk_weight[self.size_field] / 100
        bm_size_exposure = bm_stk_weight.groupby('date')['size_exposure'].sum()
        # 计算下一交易日的行业暴露
        bm_industry_exposure = bm_stk_weight.groupby(['date', 'industry'])['weight'].sum()
        bm_industry_exposure = pd.DataFrame(bm_industry_exposure).reset_index()
        # 开始优化权重
        selection = self.selections.loc[:, ['date', 'code', 'overall', 'industry']].copy()
        selection = pd.merge(selection, size_Z, on=['date', 'code'])
        selection.rename(columns={self.size_field: 'size'}, inplace=True)
        selection['next_date'] = selection['date'].apply(lambda x: next_date_dict[x])
        dates = bm_size_exposure.index.unique()
        split_dates = np.array_split(dates, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            parallel_res = Parallel()(delayed(StrategyBase.JOB_opti)
                                      (selection, bm_industry_exposure, bm_size_exposure, dates)
                                      for dates in split_dates)
        res_weight = pd.concat(parallel_res)
        folder_dir = self.root_dir + self.strategy_name
        if os.path.exists(folder_dir):
            pass
        else:
            os.makedirs(folder_dir)
        res_weight.to_csv(folder_dir+'/strategy_weight.csv', encoding='gbk')
        return res_weight

    # select stks 的工具函数
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

    # opti_weight 的工具函数
    @staticmethod
    def JOB_opti(selection, bm_industry_exposure, bm_size_exposure, dates):
        opti_res = []
        for date in dates:
            day_selection = selection.loc[selection['date'] == date, :].copy()
            selection_dummies = pd.get_dummies(day_selection['industry']).values
            day_selection_counts = day_selection.groupby('industry')['code'].count()
            # 统计选股的个数以确定最小权重, 最小权重=行业总权重/选股个数/2
            day_selection_counts.name = 'counts'
            day_selection_counts = pd.DataFrame(day_selection_counts).reset_index()
            day_bm_indu_exp = bm_industry_exposure.loc[bm_industry_exposure['date'] == date, :]
            indu_weight_limit = \
                pd.merge(day_bm_indu_exp.loc[:, ['industry', 'weight']], day_selection_counts, on='industry')
            indu_weight_limit['min_weight'] = indu_weight_limit['weight'] / indu_weight_limit['counts'] / 2
            indu_weight_limit['max_weight'] = indu_weight_limit['weight'] / indu_weight_limit['counts'] * 3
            selection_weight_limit = pd.merge(day_selection, indu_weight_limit, on='industry')
            selection_min_weight = selection_weight_limit['min_weight'].values
            selection_max_weight = selection_weight_limit['max_weight'].values
            # bm_size_exposure 是Series，index是date
            target_size_exp = bm_size_exposure[date]
            # weight 分配到行业哑变量上
            day_bm_indu_exp = pd.merge(day_bm_indu_exp, pd.get_dummies(day_bm_indu_exp['industry']),
                                       left_index=True, right_index=True)
            indu_cols = day_bm_indu_exp.columns.difference(['date', 'industry', 'weight'])
            target_indu_weight = (day_bm_indu_exp['weight'].values * day_bm_indu_exp[indu_cols].values).sum(axis=0)
            # 优化器
            w = cp.Variable(day_selection.shape[0])
            size_exp = w/100 * day_selection['size'].values
            factor_exp = w * day_selection['overall'].values
            industry_exp = w * selection_dummies
            obj = cp.Maximize(factor_exp)
            con = [industry_exp == target_indu_weight, cp.abs(size_exp - target_size_exp) <= 0.5,
                   w >= selection_min_weight, w <= selection_max_weight]
            problem = cp.Problem(obj, con)
            problem.solve(solver=cp.ECOS)
            res_weight = w.value
            if not isinstance(res_weight, np.ndarray):
                print(date, 'NO RES')
                if not opti_res:
                    continue
                else:
                    # 当日无解就用前一日的结果
                    tmp = opti_res[-1]
                    tmp['next_date'] = day_selection['next_date'].iloc[-1]
                    opti_res.append(tmp)
            else:
                day_selection = pd.merge(day_selection, day_bm_indu_exp.loc[:, ['date', 'industry', 'weight']],
                                         on=['date', 'industry'])
                day_selection.rename(columns={'weight': 'industry_weight'}, inplace=True)
                day_selection['weight'] = res_weight
                day_selection['weight_in_industry'] = day_selection['weight'] / day_selection['industry_weight'] * 100
                day_selection = day_selection.loc[:, ['next_date', 'code', 'industry', 'industry_weight', 'weight',
                                                      'weight_in_industry']]
                opti_res.append(day_selection)
        opti_res = pd.concat(opti_res).set_index('next_date')
        return opti_res



    def run(self, start, end, benchmark, range, select_pct, strategy_name, factor_weight_dict,
            industry='improved_lv1', size_field='ln_market_cap'):
        self.n_jobs = 4

        self.start = start
        self.end = end
        self.benchmark = benchmark
        self.range = range
        self.select_pct = select_pct
        self.industry = industry
        self.size_field = size_field
        self.factor_weight_dict = factor_weight_dict
        self.strategy_name = strategy_name
        self.data_prepare()
        self.process_factor()
        self.select_stks()
        self.opti_weight()


if __name__ == '__main__':
    s = StrategyBase()
    s.run(20120101, 20190801, 300, 800, 0.6, 'test',
          {'Value': {'BP': 0.33, 'EP_TTM': 0.33, 'SP': 0.33},
           'Turnover': {'std_free_turn_1m': -0.5, 'bias_std_free_turn_1m': -0.5},
           'FinancialQuality': {'ROE_ddt_Q': 1},
           'Growth': {'ROE_ddt_Q_growthY': 0.5, 'EP_TTM_growthQ': 0.5},
           'Momentum': {'free_exp_wgt_rtn_1m': -0.5, 'rtn_1m': -0.5},
           'Size': {'ln_market_cap': 1}})