import global_constant
import pandas as pd
import numpy as np
from rdf_data import rdf_data
from influxdb_data import influxdbData
from data_process import DataProcess
from joblib import Parallel,delayed,parallel_backend
from industry_neutral_engine import IndustryNeutralEngine
import logging
import datetime


class StrategyBase:
    def __init__(self):
        self.influx = influxdbData()
        self.factor_db = 'DailyFactor_Gus'

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
        self.size_data.reset_index(inplace=True)
        # size_data只取在范围内的
        self.size_data = pd.merge(self.size_data, self.code_range, on=['date','code'])
        self.size_data = self.size_data.loc[:, ['date','code',self.size_field]]
        self.size_data.set_index('date', inplace=True)
        self.industry_dummies = DataProcess.get_industry_dummies(self.code_range.set_index('date'), self.industry)
        print('all data are loaded! start processing...')

    # factor 的输入: {因子大类(measurement): {因子名称(field): 因子方向}}
    # factor 处理完的输出 {因子大类(measurement): {因子名称(field): 方向调整后且正交化后的dataframe}}
    def process_factor(self):
        m_res = []
        for m in self.factor_weight_dict.keys():
            f_res = []
            for f in self.factor_weight_dict[m].keys():
                raw_df = self.influx.getDataMultiprocess(self.factor_db, m, self.start, self.end, ['code', f])
                raw_df.index.names = ['date']
                raw_df.reset_index(inplace=True)
                raw_in_range = pd.merge(raw_df, self.code_range, how='right', on=['date', 'code'])
                # 缺失的因子用行业中位数代替
                raw_in_range[f] = raw_in_range.groupby(['date', self.industry])[f].apply(lambda x: x.fillna(x.median()))
                raw_in_range.set_index('date', inplace=True)
                # 进行remove outlier, z score和中性化
                neutralized_df = DataProcess.neutralize(raw_in_range, f, self.industry_dummies, self.size_data,
                                                        self.size_field, self.n_jobs)
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
        merge = pd.concat(m_res, join='inner', axis=1)
        merge['overall'] = merge.sum(axis=1)
        merge.reset_index(inplace=True)
        return merge

    def run(self, start, end, range, factor_weight_dict, industry='improved_lv1', size_field='ln_market_cap'):
        self.n_jobs = 4

        self.start = start
        self.end = end
        self.range = range
        self.industry = industry
        self.size_field = size_field
        self.factor_weight_dict = factor_weight_dict
        self.data_prepare()
        self.process_factor()


if __name__ == '__main__':
    s = StrategyBase()
    s.run(20150101, 20160101, None, {'Value': {'BP': 0.5, 'EP_TTM': 0.5},
                                     'Turnover': {'bias_std_free_turn_1m': -0.5, 'std_free_turn_1m': -0.5}})