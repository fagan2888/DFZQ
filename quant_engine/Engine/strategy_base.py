import global_constant
import pandas as pd
import numpy as np
from rdf_data import rdf_data
from influxdb_data import influxdbData
from data_process import DataProcess
from joblib import Parallel,delayed,parallel_backend
import warnings
import statsmodels.api as sm
from industry_neutral_engine import IndustryNeutralEngine
import logging
import datetime


class StrategyBase:
    def __init__(self):
        self.influx = influxdbData()
        self.factor_db = 'DailyFactor_Gus'

    # 获得行情信息以及选股的范围
    # 行情信息为全市场，以免吸收合并出现没有行情的情况
    def get_market_data(self):
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
            self.influx.getDataMultiprocess('DailyFactor_Gus', 'Size', self.start, self.end, ['code', self.size_filed])

    # factor 的输入: {因子大类(measurement): {因子名称(field): 因子方向}}
    # factor 处理完的输出 {因子大类(measurement): {因子名称(field): 方向调整后且正交化后的dataframe}}
    def process_factor(self):
        df_dict = {}
        for m in self.factor_dict.keys():
            for f in self.factor_dict[m].keys():
                raw_df = self.influx.getDataMultiprocess(self.factor_db, m, self.start, self.end, ['code', f])
                raw_df.index.names = ['date']
                raw_df.reset_index(inplace=True)
                raw_in_range = pd.merge(raw_df, self.code_range, how='right', on=['date', 'code'])
                # 缺失的因子用行业中位数代替
                raw_in_range[f] = raw_in_range.groupby(['date', self.industry])[f].apply(lambda x: x.fillna(x.median()))
                df_dict[m] = {f: raw_in_range}

    @staticmethod
    def JOB_process_factor(industry_dummies, size_data, df_dict):



    def run(self, start, end, range, factor_dict, industry='improved_lv1', size_field='ln_market_cap'):
        self.start = start
        self.end = end
        self.range = range
        self.industry = industry
        self.size_field = size_field
        self.factor_dict = factor_dict
        self.get_market_data()
        self.process_factor()


if __name__ == '__main__':
    s = StrategyBase()
    s.run(20150101,20160101,None,{'Turnover':{'bias_std_free_turn_1m':1}})