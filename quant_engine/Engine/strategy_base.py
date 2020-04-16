import global_constant
import pandas as pd
import numpy as np
from rdf_data import rdf_data
from influxdb_data import influxdbData
from data_process import DataProcess
from joblib import Parallel, delayed, parallel_backend
import logging
import os.path


class StrategyBase:
    def __init__(self, strategy_name):
        self.influx = influxdbData()
        self.rdf = rdf_data()
        self.strategy_name = strategy_name
        self.mkt_db = 'DailyMarket_Gus'
        self.mkt_measure = 'market'
        self.idx_wgt_measure = 'index_weight'
        self.st_measure = 'isST'
        self.industry_measure = 'industry'
        self.factor_db = 'DailyFactors_Gus'
        self.risk_exp_measure = 'RiskExposure'
        self.risk_cov_measure = 'RiskCov'
        self.spec_risk_measure = 'SpecificRisk'

    def init_log(self):
        # 配置log
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)
        file_name = 'Strategy_Report.log'
        handler = logging.FileHandler(self.folder_dir + file_name)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    # 获得行情信息以及选股的范围
    # 行情信息为全市场，以免吸收合并出现没有行情的情况
    # data_prepare 所生成的所有数据，date都是index
    def data_prepare(self):
        # =============================================================================
        # --------------------------------load data------------------------------------
        # market data
        self.mkt_data = self.influx.getDataMultiprocess(self.mkt_db, self.mkt_measure, self.start, self.end)
        self.mkt_data.index.names = ['date']
        # calendar
        self.calendar = self.mkt_data.index.unique().strftime('%Y%m%d')
        # index weight
        self.idx_wgt_data = self.influx.getDataMultiprocess(self.mkt_db, self.idx_wgt_measure, self.start, self.end)
        self.idx_wgt_data.index.names = ['date']
        # isST
        self.st_data = self.influx.getDataMultiprocess(self.mkt_db, self.st_measure, self.start, self.end)
        self.st_data.index.names = ['date']
        # industry
        self.industry_data = self.influx.getDataMultiprocess(self.mkt_db, self.industry_measure,
                                                             self.start, self.end, ['code', self.industry])
        self.industry_data.rename(columns={self.industry: 'industry'}, inplace=True)
        self.industry_data.index.names = ['date']
        # size
        self.size_data = \
            self.influx.getDataMultiprocess(self.factor_db, 'Size', self.start, self.end, ['code', self.size_field])
        self.size_data.rename(columns={self.size_field: 'size'}, inplace=True)
        self.size_data.index.names = ['date']
        # risk_exposure
        self.risk_exp = self.influx.getDataMultiprocess(self.factor_db, self.risk_exp_measure, self.start, self.end)
        self.risk_exp.index.names = ['date']
        # risk cov
        self.risk_cov = self.influx.getDataMultiprocess(self.factor_db, self.risk_cov_measure, self.start, self.end)
        self.risk_cov.drop('code', axis=1, inplace=True)
        self.risk_cov.index.names = ['date']
        # specific risk
        self.spec_risk = self.influx.getDataMultiprocess(self.factor_db, self.spec_risk_measure, self.start, self.end)
        self.spec_risk.index.names = ['date']
        print('Raw Data loaded...')
        # ========================================================================
        # ----------------------select codes in select range----------------------
        # 过滤 select range内 的票
        if self.select_range == 300:
            self.code_range = self.idx_wgt_data.loc[self.idx_wgt_data['index_code'] == '000300.SH', ['code']].copy()
        elif self.select_range == 500:
            self.code_range = self.idx_wgt_data.loc[self.idx_wgt_data['index_code'] == '000905.SH', ['code']].copy()
        elif self.select_range == 800:
            self.code_range = self.idx_wgt_data.loc[
                (self.idx_wgt_data['index_code'] == '000300.SH') | (self.idx_wgt_data['index_code'] == '000905.SH'),
                ['code']].copy()
        else:
            self.code_range = self.mkt_data.loc[:, ['code']].copy()
        self.code_range.reset_index(inplace=True)
        # 过滤 停牌 的票
        suspend_stk = self.mkt_data.loc[self.mkt_data['status'] != '停牌', ['code']].copy()
        suspend_stk.reset_index(inplace=True)
        self.code_range = pd.merge(self.code_range, suspend_stk, how='inner', on=['date', 'code'])
        # 过滤 st 的票
        st = self.st_data.copy().reset_index()
        self.code_range = pd.merge(self.code_range, st, how='left', on=['date', 'code'])
        self.code_range = self.code_range.loc[pd.isnull(self.code_range['isST']), ['date', 'code']]
        # 过滤 没有行业 的票
        indu = self.industry_data.copy().reset_index()
        self.code_range = pd.merge(self.code_range, indu, how='left', on=['date', 'code'])
        self.code_range = self.code_range.loc[pd.notnull(self.code_range['industry']), :]
        # 过滤 没有风险因子 的票
        self.code_range = pd.merge(self.code_range, self.size_data.reset_index(), how='inner', on=['date', 'code'])
        self.code_range = pd.merge(self.code_range, self.risk_exp.reset_index(), how='inner', on=['date', 'code'])
        self.code_range = pd.merge(self.code_range, self.spec_risk.reset_index(), how='inner', on=['date', 'code'])
        self.code_range = self.code_range.loc[:, ['date', 'code', 'industry']]
        self.code_range.set_index('date', inplace=True)
        # ========================================================================
        # -------------------indu dummies in select range-------------------------
        self.industry_dummies = DataProcess.get_industry_dummies(self.code_range, 'industry')
        self.industry_dummies.index.names = ['date']
        # ----------------------z size in select range----------------------------
        size_in_range = pd.merge(self.code_range.reset_index(), self.size_data.reset_index(),
                                 how='inner', on=['date', 'code'])
        dates = self.size_data['date'].unique()
        split_dates = np.array_split(dates, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            parallel_res = Parallel()(delayed(DataProcess.JOB_cross_section_Z_score)
                                      (size_in_range, 'size', dates) for dates in split_dates)
        self.z_size = pd.concat(parallel_res)
        # ----------------------benchmark_stock_weight----------------------------
        benchmark_code_dict = {50: '000016.SH', 300: '000300.SH', 500: '000905.SH'}
        self.benchmark_code = benchmark_code_dict[self.benchmark]
        self.bm_stk_wgt = self.idx_wgt_data.loc[self.idx_wgt_data['index_code'] == self.benchmark_code,
                                                ['code', 'weight']].copy()

    def process_factor(self, measure, factor, direction, if_fillna=True):
        factor_df = self.influx.getDataMultiprocess(self.factor_db, measure, self.start, self.end, ['code', factor])
        factor_df.index.names = ['date']
        factor_df.reset_index(inplace=True)
        if direction == -1:
            factor_df[factor] = factor_df[factor] * -1
        # 缺失的因子用行业中位数代
        if if_fillna:
            factor_df = pd.merge(factor_df, self.code_range.reset_index(), how='right', on=['date', 'code'])
            factor_df[factor] = factor_df.groupby(['date', 'industry'])[factor].apply(lambda x: x.fillna(x.median()))
        else:
            factor_df = pd.merge(factor_df, self.code_range.reset_index(), how='inner', on=['date', 'code'])
        factor_df.set_index('date', inplace=True)
        industry_dummies = self.industry_dummies.copy()
        size_data = self.size_data.copy()
        # 进行remove outlier, z score和中性化
        factor_df = \
            DataProcess.neutralize(factor_df, factor, industry_dummies, size_data, self.size_field, self.n_jobs)
        return factor_df

    def initialize_strategy(self, start, end, benchmark, select_range, industry, size_field):
        self.n_jobs = global_constant.N_STRATEGY
        self.start = start
        self.end = end
        self.benchmark = benchmark
        self.select_range = select_range
        self.industry = industry
        self.size_field = size_field
        self.folder_dir = global_constant.ROOT_DIR + 'Strategy/{0}/'.format(self.strategy_name)
        if os.path.exists(self.folder_dir):
            pass
        else:
            os.makedirs(self.folder_dir.rstrip('/'))
        self.data_prepare()
        self.init_log()


if __name__ == '__main__':
    sb = StrategyBase('test')
    sb.initialize_strategy(20150101, 20160101, 300, 500, 'improved_lv1', 'ln_market_cap')