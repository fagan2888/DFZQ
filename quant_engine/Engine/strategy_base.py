import global_constant
import pandas as pd
import numpy as np
from rdf_data import rdf_data
from influxdb_data import influxdbData
from data_process import DataProcess
from dateutil.relativedelta import relativedelta
import logging
import os.path


class StrategyBase:
    def __init__(self, strategy_name):
        self.influx = influxdbData()
        self.rdf = rdf_data()
        self.strategy_name = strategy_name
        self.mkt_db = global_constant.MARKET_DB
        self.mkt_measure = 'market'
        self.idx_wgt_measure = 'index_weight'
        self.st_measure = 'isST'
        self.industry_measure = 'industry'
        self.info_measure = 'stk_info'
        self.factor_db = global_constant.FACTOR_DB
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
        console = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.addHandler(console)

    # 获得行情信息以及选股的范围
    # 行情信息为全市场，以免吸收合并出现没有行情的情况
    # data_prepare 所生成的所有数据，date都是index
    # factor test 所用的 start end 会跟 做策略有所不同
    def data_prepare(self, start, end):
        # =============================================================================
        # --------------------------------load data------------------------------------
        # market data
        self.mkt_data = self.influx.getDataMultiprocess(self.mkt_db, self.mkt_measure, start, end)
        self.mkt_data.index.names = ['date']
        # calendar
        self.calendar = self.mkt_data.index.unique().strftime('%Y%m%d')
        # index weight
        self.idx_wgt_data = self.influx.getDataMultiprocess(self.mkt_db, self.idx_wgt_measure, start, end)
        self.idx_wgt_data.index.names = ['date']
        # benchmark_weight
        benchmark_code_dict = {50: '000016.SH', 300: '000300.SH', 500: '000905.SH'}
        self.benchmark_code = benchmark_code_dict[self.benchmark]
        self.bm_stk_wgt = self.idx_wgt_data.loc[
            self.idx_wgt_data['index_code'] == self.benchmark_code, ['code', 'weight']].copy()
        # isST
        self.st_data = self.influx.getDataMultiprocess(self.mkt_db, self.st_measure, start, end)
        self.st_data.index.names = ['date']
        # industry
        self.industry_data = self.influx.getDataMultiprocess(self.mkt_db, self.industry_measure, start, end,
                                                             ['code', self.industry])
        self.industry_data.rename(columns={self.industry: 'industry'}, inplace=True)
        self.industry_data.index.names = ['date']
        # stock info
        self.stk_info = self.influx.getDataMultiprocess(self.mkt_db, self.info_measure, 20200615, 20200615,
                                                        ['code', 'list_date', 'delist_date'])
        self.stk_info.reset_index(inplace=True)
        self.stk_info = self.stk_info.drop('index', axis=1)
        self.stk_info['list_date'] = pd.to_datetime(self.stk_info['list_date'])
        self.stk_info['delist_date'] = pd.to_datetime(self.stk_info['delist_date'])
        self.stk_info['range_date'] = self.stk_info['list_date'].apply(lambda x: x + relativedelta(years=2))
        # industry dummies
        self.industry_dummies = DataProcess.get_industry_dummies(self.industry_data, 'industry')
        self.industry_dummies.index.names = ['date']
        print('-market related data loaded...')
        # risk_exposure
        self.risk_exp = self.influx.getDataMultiprocess(self.factor_db, self.risk_exp_measure, self.start, self.end)
        self.risk_exp.index.names = ['date']
        # risk cov
        self.risk_cov = self.influx.getDataMultiprocess(self.factor_db, self.risk_cov_measure, self.start, self.end)
        cols = self.risk_cov.columns.difference(['code'])
        self.risk_cov[cols] = self.risk_cov[cols] * self.adj_interval * 0.0001
        self.risk_cov.index.names = ['date']
        # specific risk
        self.spec_risk = self.influx.getDataMultiprocess(self.factor_db, self.spec_risk_measure, self.start, self.end)
        self.spec_risk['specific_risk'] = self.spec_risk['specific_risk'] * np.sqrt(self.adj_interval) * 0.01
        self.spec_risk.index.names = ['date']
        print('-risk data loaded...')
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
        suspend_stk = self.mkt_data.loc[:, ['code', 'status']].copy()
        self.code_range = pd.merge(self.code_range, suspend_stk.reset_index(), how='left', on=['date', 'code'])
        # 过滤 st 的票
        self.code_range = pd.merge(self.code_range, self.st_data.reset_index(), how='left', on=['date', 'code'])
        # 过滤 新股 和 退市 的票
        self.code_range = pd.merge(self.code_range, self.stk_info, how='inner', on=['code'])
        self.code_range = self.code_range.loc[(self.code_range['date'] >= self.code_range['range_date']) &
                                              ((self.code_range['date'] < self.code_range['delist_date']) |
                                               pd.isnull(self.code_range['delist_date'])), ['date', 'code']]
        # 组合 industry
        self.code_range = pd.merge(self.code_range, self.industry_data.reset_index(), how='left', on=['date', 'code'])
        self.code_range.set_index('date', inplace=True)
        print('-code range loaded...')
        self.size_data = self.influx.getDataMultiprocess(self.factor_db, 'Size', self.start, self.end,
                                                         ['code', 'ln_market_cap'])
        self.size_data.rename(columns={'ln_market_cap': 'z_mv'}, inplace=True)
        self.size_data.index.names = ['date']
        self.size_data = pd.merge(self.size_data.reset_index(), self.code_range.reset_index(), how='right',
                                  on=['date', 'code'])
        self.size_data['z_mv'] = \
            self.size_data.groupby(['date', 'industry'])['z_mv'].apply(lambda x: x.fillna(x.median()))
        self.size_data = self.size_data.loc[:, ['date', 'code', 'z_mv']]
        self.size_data = DataProcess.standardize(self.size_data, 'z_mv', False, self.n_jobs)
        self.size_data.set_index('date', inplace=True)
        print('-size data loaded...')

    def process_factor(self, db, measure, factor, direction, fillna, check_rp, neutralize):
        if check_rp:
            factor_df = \
                self.influx.getDataMultiprocess(db, measure, self.start, self.end, ['code', factor, 'report_period'])
            factor_df = DataProcess.check_report_period(factor_df)
        else:
            factor_df = self.influx.getDataMultiprocess(db, measure, self.start, self.end, ['code', factor])
        factor_df.index.names = ['date']
        factor_df.reset_index(inplace=True)
        if direction == -1:
            factor_df[factor] = factor_df[factor] * -1
        # 缺失的因子用行业中位数填充
        if fillna == 'median':
            factor_df = pd.merge(factor_df, self.code_range.reset_index(), how='right', on=['date', 'code'])
            factor_df[factor] = factor_df.groupby(['date', 'industry'])[factor].apply(lambda x: x.fillna(x.median()))
        # 缺失的因子用0填充
        elif fillna == 'zero':
            factor_df = pd.merge(factor_df, self.code_range.reset_index(), how='right', on=['date', 'code'])
            factor_df[factor] = factor_df[factor].fillna(0)
        else:
            factor_df = pd.merge(factor_df, self.code_range.reset_index(), how='inner', on=['date', 'code'])
            factor_df = factor_df.dropna(subset=[factor])
        factor_df.set_index('date', inplace=True)
        # 进行 中性化 / 标准化
        if neutralize:
            factor_df = DataProcess.neutralize(factor_df, factor, self.industry_dummies, self.size_data, self.n_jobs)
        else:
            factor_df = factor_df.loc[:, ['date', 'code', 'factor']]
            factor_df = DataProcess.standardize(factor_df, factor, False, self.n_jobs)
        return factor_df

    def initialize_strategy(self, start, end, benchmark, select_range, industry, adj_interval):
        self.n_jobs = global_constant.N_STRATEGY
        self.start = start
        self.end = end
        self.benchmark = benchmark
        self.select_range = select_range
        self.industry = industry
        self.adj_interval = adj_interval
        self.folder_dir = global_constant.ROOT_DIR + 'Backtest_Result/{0}/'.format(self.strategy_name)
        if os.path.exists(self.folder_dir):
            pass
        else:
            os.makedirs(self.folder_dir.rstrip('/'))
        self.data_prepare(self.start, self.end)
        self.init_log()


if __name__ == '__main__':
    sb = StrategyBase('test')
    sb.initialize_strategy(20150101, 20160101, 300, 500, 'improved_lv1', 5)