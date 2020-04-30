import global_constant
import pandas as pd
import numpy as np
import os.path
from data_process import DataProcess
from joblib import Parallel, delayed, parallel_backend
from rdf_data import rdf_data
from influxdb_data import influxdbData
from dateutil.relativedelta import relativedelta
import warnings
import statsmodels.api as sm
from backtest_engine import BacktestEngine
import logging
import datetime


class FactorTest:
    @staticmethod
    def JOB_T_test(processed_factor, factor_field, dates):
        F_alpha = []
        T_alpha = []
        for date in dates:
            day_factor = processed_factor.loc[processed_factor['date'] == date, factor_field].values
            day_alpha = processed_factor.loc[processed_factor['date'] == date, 'alpha'].values
            RLM_est = sm.RLM(day_alpha, day_factor, M=sm.robust.norms.HuberT()).fit()
            day_RLM_para = RLM_est.params
            day_Tvalue = RLM_est.tvalues
            F_alpha.append(day_RLM_para[0])
            T_alpha.append(day_Tvalue[0])
        return np.array([F_alpha, T_alpha])

    @staticmethod
    def JOB_IC(processed_factor, factor_field, dates):
        day_IC = []
        IC_date = []
        for date in dates:
            day_factor = processed_factor.loc[processed_factor['date'] == date, factor_field]
            day_return = processed_factor.loc[processed_factor['date'] == date, 'alpha']
            day_IC.append(day_factor.corr(day_return, method='spearman'))
            IC_date.append(date)
        return pd.Series(day_IC, index=IC_date)

    # 等权所用的分组job
    @staticmethod
    def JOB_group_equal_weight(dates, groups, factor, factor_field):
        labels = []
        for i in range(1, groups + 1):
            labels.append('group_' + str(i))
        res = []
        for date in dates:
            day_factor = factor.loc[factor['date'] == date, :].copy()
            industries = day_factor['industry'].unique()
            for ind in industries:
                day_industry_factor = day_factor.loc[day_factor['industry'] == ind, :].copy()
                # 行业成分不足10支票时，所有group配置一样
                if day_industry_factor.shape[0] < 10:
                    day_industry_factor['group'] = 'same group'
                    day_industry_factor['weight_in_industry'] = 100 / day_industry_factor.shape[0]
                else:
                    day_industry_factor['group'] = pd.qcut(day_industry_factor[factor_field], 5, labels=labels)
                    group_counts = day_industry_factor['group'].value_counts()
                    day_industry_factor['weight_in_industry'] = day_industry_factor.apply(
                        lambda row: 100 / group_counts[row['group']], axis=1)
                res.append(day_industry_factor)
        res_df = pd.concat(res)
        res_df.set_index('next_1_day', inplace=True)
        return res_df

    # 回测所用的工具函数
    @staticmethod
    def JOB_backtest(group_name, grouped_weight, benchmark, adj_interval, cash_reserve, price_field, indu_field,
                     data_input, stock_capital, stk_slippage, stk_fee, logger_lvl=logging.INFO):
        weight = grouped_weight.loc[
            (grouped_weight['group'] == group_name) | (grouped_weight['group'] == 'same_group'),
            ['code', 'industry', 'weight_in_industry']].copy()
        BE = IndustryNeutralEngine(stock_capital, stk_slippage, stk_fee, save_name=group_name, logger_lvl=logger_lvl)
        start = weight.index[0].strftime('%Y%m%d')
        end = weight.index[-1].strftime('%Y%m%d')
        portfolio_value = BE.run(weight, start, end, benchmark, adj_interval, cash_reserve, price_field, indu_field,
                                 data_input)
        portfolio_value.rename(columns={'TotalValue': group_name + '_TotalValue'}, inplace=True)
        portfolio_value = pd.DataFrame(portfolio_value[group_name + '_TotalValue'])
        print('%s backtest finish!' % group_name)
        return portfolio_value

    def __init__(self, measure, factor):
        self.rdf = rdf_data()
        self.influx = influxdbData()
        self.n_jobs = global_constant.N_JOBS
        self.mkt_db = 'DailyMarket_Gus'
        self.mkt_measure = 'market'
        self.idx_wgt_measure = 'index_weight'
        self.st_measure = 'isST'
        self.industry_measure = 'industry'
        self.factor_db = 'DailyFactors_Gus'
        self.factor_measure = measure
        self.factor = factor

    def init_log(self):
        # 配置log
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=self.logger_lvl)
        folder_dir = global_constant.ROOT_DIR + 'Factor_Test/{0}/'.format(self.factor)
        if os.path.exists(folder_dir):
            pass
        else:
            os.makedirs(folder_dir.rstrip('/'))
        handler = logging.FileHandler(folder_dir + 'Test_result.log')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def df_prepare(self):
        # --------------------------------load data------------------------------------
        # market data
        mkt_end = (pd.to_datetime(str(self.end)) + relativedelta(months=1)).strftime('%Y%m%d')
        self.mkt_data = self.influx.getDataMultiprocess(self.mkt_db, self.mkt_measure, self.start, mkt_end)
        self.mkt_data.index.names = ['date']
        # calendar
        self.calendar = self.rdf.get_trading_calendar()
        # index weight
        self.idx_wgt_data = self.influx.getDataMultiprocess(self.mkt_db, self.idx_wgt_measure, self.start, mkt_end)
        self.idx_wgt_data.index.names = ['date']
        # benchmark_weight
        benchmark_code_dict = {50: '000016.SH', 300: '000300.SH', 500: '000905.SH'}
        self.benchmark_code = benchmark_code_dict[self.benchmark]
        self.bm_stk_wgt = self.idx_wgt_data.loc[
            self.idx_wgt_data['index_code'] == self.benchmark_code, ['code', 'weight']].copy()
        self.bm_mkt = self.mkt_data.loc[self.mkt_data['code']==self.benchmark_code, :].copy()
        # isST
        self.st_data = self.influx.getDataMultiprocess(self.mkt_db, self.st_measure, self.start, mkt_end)
        self.st_data.index.names = ['date']
        # industry
        self.industry_data = self.influx.getDataMultiprocess(self.mkt_db, self.industry_measure,
                                                             self.start, mkt_end, ['code', self.industry])
        self.industry_data.rename(columns={self.industry: 'industry'}, inplace=True)
        self.industry_data.index.names = ['date']
        # industry dummies
        self.industry_dummies = DataProcess.get_industry_dummies(self.industry_data, 'industry')
        self.industry_dummies.index.names = ['date']
        # benchmark industry weight
        self.bm_indu_weight = pd.merge(self.idx_wgt_data.reset_index(), self.industry_data.reset_index(),
                                       on=['date', 'code'])
        self.bm_indu_weight = self.bm_indu_weight.group(['date', 'industry'])['weight'].sum()
        self.bm_indu_weight = self.bm_indu_weight.reset_index().set_index('date', inplace=True)
        print('-market related data loaded...')
        # size
        self.size_data = \
            self.influx.getDataMultiprocess(self.factor_db, 'Size', self.start, mkt_end, ['code', self.size_field])
        self.size_data.rename(columns={self.size_field: 'size'}, inplace=True)
        self.size_data.index.names = ['date']
        print('-size data loaded...')
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
        # 组合 industry
        self.code_range = pd.merge(self.code_range, self.industry_data.reset_index(), how='inner', on=['date', 'code'])
        self.code_range.set_index('date')

    def get_factor_df(self):
        factor_df = self.influx.getDataMultiprocess(self.factor_db, self.factor_measure, self.start, self.end,
                                                    ['code', self.factor])
        factor_df.index.names = ['date']
        factor_df.reset_index(inplace=True)
        if direction == -1:
            factor_df[self.factor] = factor_df[self.factor] * -1
        # 缺失的因子用行业中位数代
        if if_fillna:
            factor_df = pd.merge(factor_df, self.code_range.reset_index(), how='right', on=['date', 'code'])
            factor_df[self.factor] = factor_df.groupby(['date', 'industry'])[self.factor].apply(
                lambda x: x.fillna(x.median()))
        else:
            factor_df = pd.merge(factor_df, self.code_range.reset_index(), how='inner', on=['date', 'code'])
        factor_df.set_index('date', inplace=True)
        industry_dummies = self.industry_dummies.copy()
        size_data = self.size_data.copy()
        # 进行remove outlier, z score和中性化
        factor_df = \
            DataProcess.neutralize(factor_df, self.factor, industry_dummies, size_data, self.n_jobs)
        print('-factor loaded...')
        return factor_df

    def get_alpha_df(self):
        next_date_dict = {}
        for date in self.mkt_data.index.unique():
            next_date_dict.update(DataProcess.get_next_date(self.calendar, date, self.adj_interval))
        # get stk return
        stk_ret_df = self.mkt_data.copy()
        stk_ret_df['next_period_date'] = stk_ret_df.index
        stk_ret_df['next_period_date'] = stk_ret_df['next_period_date'].map(next_date_dict)
        stk_ret_df = stk_ret_df.dropna(subset=['next_period_date'])
        stk_ret_df['fq_close'] = stk_ret_df['adj_factor'] * stk_ret_df['close']
        stk_ret_df.reset_index(inplace=True)
        stk_ret_df = stk_ret_df.loc[:, ['date', 'code', 'status', 'fq_close', 'next_period_date']]
        next_mkt_data = stk_ret_df.copy()
        next_mkt_data = next_mkt_data.loc[:, ['date', 'code', 'fq_close']]
        next_mkt_data.rename(columns={'date': 'next_period_date', 'fq_close': 'next_fq_close'}, inplace=True)
        stk_ret_df = pd.merge(stk_ret_df, next_mkt_data, on=['next_period_date', 'code'])
        stk_ret_df['return'] = stk_ret_df['next_fq_close'] / stk_ret_df['fq_close'] - 1
        stk_ret_df = stk_ret_df.loc[(stk_ret_df['return'] < 0.1 * self.adj_interval) &
                                    (stk_ret_df['return'] > -0.1 * self.adj_interval), :]
        stk_ret_df = stk_ret_df.loc[:, ['date', 'code', 'status', 'return']]
        # get benchmark return
        bm_ret_df = self.bm_mkt.copy()
        bm_ret_df['next_period_date'] = bm_ret_df.index
        bm_ret_df['next_period_date'] = bm_ret_df['next_period_date'].map(next_date_dict)
        bm_ret_df = bm_ret_df.dropna(subset=['next_period_date'])
        bm_ret_df['fq_close'] = bm_ret_df['adj_factor'] * bm_ret_df['close']
        bm_ret_df.reset_index(inplace=True)
        bm_ret_df = bm_ret_df.loc[:, ['date', 'fq_close', 'next_period_date']]
        next_bm_data = bm_ret_df.copy()
        next_bm_data = next_bm_data.loc[:, ['date', 'fq_close']]
        next_bm_data.rename(columns={'date': 'next_period_date', 'fq_close': 'next_fq_close'}, inplace=True)
        bm_ret_df = pd.merge(bm_ret_df, next_bm_data, on=['next_period_date'])
        bm_ret_df['idx_return'] = bm_ret_df['next_fq_close'] / bm_ret_df['fq_close'] - 1
        bm_ret_df = bm_ret_df.loc[:, ['date', 'idx_return']]
        alpha_df = pd.merge(stk_ret_df, bm_ret_df, on=['date'])
        alpha_df['alpha'] = alpha_df['return'] - alpha_df['idx_return']
        alpha_df.set_index('date', inplace=True)
        alpha_df = alpha_df.loc[:, ['code', 'status', 'alpha']]
        return alpha_df

    # days 决定next_period_return 的周期
    def validity_check(self, T_test=True):
        factor_alpha = pd.merge(self.factor_data, self.alpha_df.reset_index(), on=['date', 'code'])
        factor_alpha = factor_alpha.loc[factor_alpha['status'] != '停牌', :]
        dates = factor_alpha['date'].unique()
        split_dates = np.array_split(dates, self.n_jobs)
        if T_test:
            # T检验
            with parallel_backend('multiprocessing', n_jobs=4):
                parallel_res = Parallel()(delayed(FactorTest.JOB_T_test)
                                          (factor_alpha, self.factor, dates) for dates in split_dates)
            # 第一行F，第二行T
            RLM_result = np.concatenate(parallel_res, axis=1)
            F_alpha_values = RLM_result[0]
            T_alpha_values = RLM_result[1]
            Falp_over_0_pct = F_alpha_values[F_alpha_values > 0].shape[0] / F_alpha_values.shape[0]
            avg_abs_Talp = abs(T_alpha_values).mean()
            abs_Talp_over_2_pct = abs(T_alpha_values)[abs(T_alpha_values) >= 2].shape[0] / T_alpha_values.shape[0]
            self.summary_dict['Falp_over_0_pct'] = Falp_over_0_pct
            self.summary_dict['avg_abs_Talp'] = avg_abs_Talp
            self.summary_dict['abs_Talp_over_2_pct'] = abs_Talp_over_2_pct
            print('-' * 30)
            print('REGRESSION RESULT: \n -Falp_over_0_pct: %f \n -avg_abs_Talp: %f \n -abs_Talp_over_2_pct: %f \n'
                  % (Falp_over_0_pct, avg_abs_Talp, abs_Talp_over_2_pct))
            # 计算IC
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            parallel_res = Parallel()(delayed(FactorTest.JOB_IC)
                                      (factor_alpha, self.factor, dates) for dates in split_dates)
        IC = pd.concat(parallel_res)
        IC_over_0_pct = IC[IC > 0].shape[0] / IC.shape[0]
        abs_IC_over_20pct_pct = abs(IC)[abs(IC) > 0.02].shape[0] / IC.shape[0]
        IR = IC.mean() / IC.std()
        self.summary_dict['IC_mean'] = IC.mean()
        self.summary_dict['IC_std'] = IC.std()
        self.summary_dict['IC_over_0_pct'] = IC_over_0_pct
        self.summary_dict['abs_IC_over_20pct_pct'] = abs_IC_over_20pct_pct
        self.summary_dict['IR'] = IR
        self.summary_dict['ICIR'] = IC.mean() / IR
        print('-' * 30)
        print('ICIR RESULT: \n   IC mean: %f \n   IC std: %f \n   IC_over_0_pct: %f \n   '
              'abs_IC_over_20pct_pct: %f \n   IR: %f \n   ICIR: %f \n' %
              (IC.mean(), IC.std(), IC_over_0_pct, abs_IC_over_20pct_pct, IR, IC.mean() / IR))
        IC.name = 'IC'
        return IC

    # 此处weight_field为行业内权重分配的field
    def group_factor(self):
        mkt_data = pd.merge(self.mkt_data.reset_index(), self.code_range.reset_index(), on=['date', 'code'])
        mkt_data = mkt_data.loc[mkt_data['status'] != '停牌', :]
        idxs = mkt_data.index.unique()
        next_date_dict = {}
        for date in idxs:
            next_date_dict.update(DataProcess.get_next_date(self.calendar, date, 1))
        mkt_data['next_1_day'] = mkt_data.index
        mkt_data['next_1_day'] = mkt_data['next_1_day'].map(next_date_dict)
        mkt_data.reset_index(inplace=True)
        # 组合得到当天未停牌因子的code，中性后的因子值，行业
        merge_df = pd.merge(self.factor_data, mkt_data, on=['date', 'code'])
        # 按等权测试
        dates = merge_df['date'].unique()
        split_dates = np.array_split(dates, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            result_list = Parallel()(delayed(FactorTest.JOB_group_equal_weight)
                                     (dates, self.groups, merge_df, self.factor) for dates in split_dates)
        grouped_weight = pd.concat(result_list)
        grouped_weight = grouped_weight.loc[grouped_weight['weight_in_industry'] > 0, :].sort_index()
        str_start = grouped_weight.index[0].strftime('%Y%m%d')
        str_end = grouped_weight.index[-1].strftime('%Y%m%d')
        folder_dir = global_constant.ROOT_DIR + '/Backtest_Result/Factor_Group_Weight/{0}/'.format(self.factor)
        if os.path.exists(folder_dir):
            pass
        else:
            os.makedirs(folder_dir.rstrip('/'))
        grouped_weight.to_csv(folder_dir + 'EquW{0}_{1}to{2}.csv'.format(str(self.groups), str_start, str_end),
                              encoding='gbk')
        return grouped_weight

    def group_backtest(self, grouped_weight):
        group = []
        for i in range(1, self.groups + 1):
            group.append('group_' + str(i))
        with parallel_backend('multiprocessing', n_jobs=self.groups):
            parallel_res = Parallel()(delayed(FactorTest.JOB_backtest)
                                      (g, grouped_weight, self.benchmark, self.adj_interval, self.cash_reserve,
                                       self.price_field, self.industry, self.mkt_data, self.capital,
                                       self.stk_slippage, self.stk_fee, self.logger_lvl) for g in group)
        tot_res = pd.concat(parallel_res, axis=1)
        # 合并指数value
        comparing_index = ['000300.SH', '000905.SH']
        index_value = self.mkt_data.loc[self.mkt_data['code'].isin(comparing_index), ['code', 'close']]
        index_value.set_index([index_value.index, 'code'], inplace=True)
        index_value = index_value.unstack()['close']
        index_value['ZZ800'] = 0.5 * index_value['000300.SH'] + 0.5 * index_value['000905.SH']
        for col in index_value.columns:
            index_value[col] = index_value[col] / index_value[col].iloc[0] * self.capital
        tot_res = pd.merge(tot_res, index_value, left_index=True, right_index=True)
        tot_res['accum_300_alpha'] = (tot_res['group_5_TotalValue'] - tot_res['000300.SH']) / self.capital
        tot_res['accum_500_alpha'] = (tot_res['group_5_TotalValue'] - tot_res['000905.SH']) / self.capital
        tot_res['accum_800_alpha'] = (tot_res['group_5_TotalValue'] - tot_res['ZZ800']) / self.capital
        str_start = tot_res.index[0].strftime('%Y%m%d')
        str_end = tot_res.index[-1].strftime('%Y%m%d')

        folder_dir = global_constant.ROOT_DIR + '/Backtest_Result/Group_Value/{0}/'.format(self.factor)
        if os.path.exists(folder_dir):
            pass
        else:
            os.makedirs(folder_dir.rstrip('/'))
        tot_res.to_csv(folder_dir + 'Value_{0}to{1}.csv'.format(str_start, str_end), encoding='gbk')
        # -------------------------------------
        self.summary_dict['AnnRet'] = \
            DataProcess.calc_ann_return(tot_res[group[-1] + '_TotalValue'])
        self.summary_dict['alpha_300'] = \
            DataProcess.calc_alpha_ann_return(tot_res[group[-1] + '_TotalValue'], tot_res['000300.SH'])
        self.summary_dict['alpha_500'] = \
            DataProcess.calc_alpha_ann_return(tot_res[group[-1] + '_TotalValue'], tot_res['000905.SH'])
        self.summary_dict['alpha_800'] = \
            DataProcess.calc_alpha_ann_return(tot_res[group[-1] + '_TotalValue'], tot_res['ZZ800'])
        self.summary_dict['MDD'] = \
            DataProcess.calc_max_draw_down(tot_res[group[-1] + '_TotalValue'])
        self.summary_dict['sharpe_ratio'] = \
            DataProcess.calc_sharpe(tot_res[group[-1] + '_TotalValue'])
        self.summary_dict['sharpe_alpha_300'] = \
            DataProcess.calc_alpha_sharpe(tot_res[group[-1] + '_TotalValue'], tot_res['000300.SH'])
        self.summary_dict['sharpe_alpha_500'] = \
            DataProcess.calc_alpha_sharpe(tot_res[group[-1] + '_TotalValue'], tot_res['000905.SH'])
        self.summary_dict['sharpe_alpha_800'] = \
            DataProcess.calc_alpha_sharpe(tot_res[group[-1] + '_TotalValue'], tot_res['ZZ800'])
        print('-' * 30)
        # -------------------------------------
        self.summary_dict['Start_Time'] = tot_res.index[0].strftime('%Y%m%d')
        self.summary_dict['End_Time'] = tot_res.index[-1].strftime('%Y%m%d')
        return self.summary_dict.copy()

    def generate_report(self, summary_dict):
        rep = {self.factor: summary_dict}
        rep = pd.DataFrame(rep)
        rep = rep.reindex(
            index=['Start_Time', 'End_Time', 'IC_mean', 'IC_std', 'IR', 'ICIR', 'AnnRet', 'alpha_300', 'alpha_500',
                   'alpha_800', 'MDD', 'sharpe_ratio', 'sharpe_alpha_300', 'sharpe_alpha_500', 'sharpe_alpha_800',
                   'abs_Talp_over_2_pct', 'IC_over_0_pct', 'abs_IC_over_20pct_pct', 'Fret_over_0_pct',
                   'Falp_over_0_pct', 'avg_abs_Tret', 'avg_abs_Talp', 'abs_Tret_over_2_pct']).T
        str_start = rep['Start_Time'].iloc[0]
        str_end = rep['End_Time'].iloc[0]
        folder_dir = global_constant.ROOT_DIR + '/Backtest_Result/Factor_Report/{0}/'.format(self.factor)
        if os.path.exists(folder_dir):
            pass
        else:
            os.makedirs(folder_dir.rstrip('/'))
        rep.to_csv(folder_dir + '{0}to{1}.csv'.format(str_start, str_end), encoding='gbk')

    def run(
            # 初始化参数
            self, start, end, direction, if_fillna, benchmark, select_range, industry, size_field,
            # 回测参数
            adj_interval=5, groups=5, capital=5000000, cash_reserve=0.03, stk_slippage=0.001,
            stk_fee=0.0001, price_field='vwap', logger_lvl=logging.ERROR):
        self.start = start
        self.end = end
        self.direction = direction
        self.if_fillna = if_fillna
        self.benchmark = benchmark
        self.select_range = select_range
        self.industry = industry
        self.size_field = size_field
        self.groups = groups
        self.capital = capital
        self.cash_reserve = cash_reserve
        self.stk_slippage = stk_slippage
        self.stk_fee = stk_fee
        self.price_field = price_field
        self.adj_interval = adj_interval
        self.logger_lvl = logger_lvl
        self.summary_dict = {}
        # ---------------------------------------------------------------
        # init log
        self.init_log()
        # load data
        self.df_prepare()
        # alpha 数据
        self.alpha_df = self.get_alpha_df()
        # 因子数据
        self.factor_data = self.get_factor_df()
        # 有效性检验
        self.validity_check(T_test=True)
        print('validity checking finish')
        print('-' * 30)
        # 分组
        grouped_weight = self.group_factor(self.factor_data, self.mkt_data)
        print('factor grouping finish')
        print('-' * 30)
        # 回测
        summary_dict = self.group_backtest(grouped_weight)
        print('group backtest finish')
        print('-' * 30)
        # 生成报告
        self.generate_report(summary_dict)
        print('report got')


if __name__ == '__main__':
    dt_start = datetime.datetime.now()
    warnings.filterwarnings("ignore")

    start = 20120101
    end = 20121231
    measurements = ['Momentum']
    factors = ['free_exp_wgt_rtn_m1']
    directions = [-1]
    if_fillnas = [False]
    benchmark = 300
    select_range = 800
    adj_interval = 5
    industry = 'improved_lv1'
    size_field = 'ln_market_cap'

    for i in range(len(factors)):
        factor = factors[i]
        measurement = measurements[i]
        direction = directions[i]
        if_fillna = if_fillnas[i]
        test = FactorTest(measurement, factor)
        test.run(start, end, direction, if_fillna, benchmark, select_range, industry, size_field)
    print('Test finish! Time token: ', datetime.datetime.now()-dt_start)