import global_constant
import pandas as pd
import numpy as np
from rdf_data import rdf_data
from influxdb_data import influxdbData
from data_process import DataProcess


class StrategyBase:
    def __init__(self, strategy_name):
        self.influx = influxdbData()
        self.rdf = rdf_data()
        self.mkt_db = 'DailyData_Gus'
        self.mkt_measure = 'marketData'
        self.factor_db = 'DailyFactors_Gus'
        self.strategy_name = strategy_name

    # 获得行情信息以及选股的范围
    # 行情信息为全市场，以免吸收合并出现没有行情的情况
    # data_prepare 所生成的所有数据，date都是index
    def data_prepare(self):
        self.mkt_data = self.influx.getDataMultiprocess(self.mkt_db, self.mkt_measure, self.start, self.end, None)
        # 剔除没有行业或者状态为停牌(没有状态)或为st的股票
        self.code_range = self.mkt_data.loc[
                          pd.notnull(self.mkt_data[self.industry]) &
                          ~((self.mkt_data['status'] == '停牌') | pd.isnull(self.mkt_data['status'])) &
                          (self.mkt_data['isST'] == False), :].copy()
        if not self.select_range:
            self.code_range = self.code_range.loc[:, ['code', self.industry]]
        elif self.select_range == 300:
            self.code_range = self.code_range.loc[pd.notnull(self.code_range['IF_weight']), ['code', self.industry]]
        elif self.select_range == 500:
            self.code_range = self.code_range.loc[pd.notnull(self.code_range['IC_weight']), ['code', self.industry]]
        elif self.select_range == 800:
            self.code_range = \
                self.code_range.loc[pd.notnull(self.code_range['IF_weight']) | pd.notnull(self.code_range['IC_weight']),
                                    ['code', self.industry]]
        else:
            print('无效的range')
            raise NameError
        self.code_range.index.names = ['date']
        self.size_data = \
            self.influx.getDataMultiprocess(self.factor_db, 'Size', self.start, self.end, ['code', self.size_field])
        self.size_data.index.names = ['date']
        self.industry_dummies = DataProcess.get_industry_dummies(self.code_range, self.industry)
        self.industry_dummies.index.names = ['date']
        print('All mkt data are loaded!')
        print('-' * 30)

    def process_factor(self, measure, factor, direction, if_fillna=True):
        factor_df = self.influx.getDataMultiprocess(self.factor_db, measure, self.start, self.end, ['code', factor])
        factor_df.index.names = ['date']
        factor_df.reset_index(inplace=True)
        if direction == -1:
            factor_df[factor] = factor_df[factor] * -1
        code_range = self.code_range.copy().reset_index()
        # 缺失的因子用行业中位数代
        if if_fillna:
            factor_df = pd.merge(factor_df, code_range, how='right', on=['date', 'code'])
            factor_df[factor] = factor_df.groupby(['date', self.industry])[factor].apply(lambda x: x.fillna(x.median()))
        else:
            factor_df = pd.merge(factor_df, code_range, how='inner', on=['date', 'code'])
        factor_df.set_index('date', inplace=True)
        industry_dummies = self.industry_dummies.copy()
        size_data = self.size_data.copy()
        # 进行remove outlier, z score和中性化
        factor_df = \
            DataProcess.neutralize(factor_df, factor, industry_dummies, size_data, self.size_field, self.n_jobs)
        return factor_df

    def initialize_strategy(self, start, end, benchmark, select_range, industry, size_field):
        self.n_jobs = global_constant.N_JOBS
        self.start = start
        self.end = end
        self.benchmark = benchmark
        self.select_range = select_range
        self.industry = industry
        self.size_field = size_field
        self.folder_dir = global_constant.ROOT_DIR + 'Strategy/' + self.strategy_name + '/'
        self.data_prepare()