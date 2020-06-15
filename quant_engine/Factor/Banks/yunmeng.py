from factor_base import FactorBase
import pandas as pd
import numpy as np
import datetime
from influxdb_data import influxdbData
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS

class YunMeng(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'yunmeng'

    def cal_factors(self, start, end, n_jobs):
        adj_start = (pd.to_datetime(str(start)) - relativedelta(years=1)).strftime('%Y%m%d')
        # 银行指标需提前半年，以便向后填充
        # 拨备率，不良贷款金额，关注贷款金额 需要对齐report period
        provision_amount = self.influx.getDataMultiprocess('FinancialReport_Gus', 'provision_amount', adj_start, end,
                                                           ['code', 'report_period', 'provision_amount'])
        provision_amount.index.names = ['date']
        bad_loan = self.influx.getDataMultiprocess('FinancialReport_Gus', 'bad_loan', adj_start, end,
                                                   ['code', 'report_period', 'bad_loan'])
        bad_loan.index.names = ['date']
        core_CA_ratio = self.influx.getDataMultiprocess('FinancialReport_Gus', 'core_CA_ratio', adj_start, end,
                                                        ['code', 'report_period', 'core_CA_ratio'])
        core_CA_ratio.index.names = ['date']
        loan_classification = self.influx.getDataMultiprocess('FinancialReport_Gus', 'loan_classification',
                                                              20080101, end, ['code', 'spec_ment_loan'])
        loan_classification['report_period'] = loan_classification.index.strftime('%Y%m%d')
        merge = pd.merge(
            provision_amount.reset_index(), bad_loan.reset_index(), how='outer', on=['date', 'code', 'report_period'])
        merge = pd.merge(
            merge, core_CA_ratio.reset_index(), how='outer', on=['date', 'code', 'report_period'])
        merge = pd.merge(
            merge, loan_classification, how='left', on=['code', 'report_period'])
        merge = merge.sort_values('report_period')
        # 同一date 同一code 不同report_period 数据向后填充
        merge['provision_amount'] = merge.groupby(['date', 'code'])['provision_amount'].fillna(method='ffill')
        merge['core_CA_ratio'] = merge.groupby(['date', 'code'])['core_CA_ratio'].fillna(method='ffill')
        merge['bad_loan'] = merge.groupby(['date', 'code'])['bad_loan'].fillna(method='ffill')
        merge['spec_ment_loan'] = merge.groupby(['date', 'code'])['spec_ment_loan'].fillna(method='ffill')
        merge = merge.drop_duplicates(['date', 'code'], 'last')
        merge = merge.sort_values('date')
        merge['provision_amount'] = merge.groupby(['code'])['provision_amount'].fillna(method='ffill')
        merge['core_CA_ratio'] = merge.groupby(['code'])['core_CA_ratio'].fillna(method='ffill')
        merge['bad_loan'] = merge.groupby(['code'])['bad_loan'].fillna(method='ffill')
        merge['spec_ment_loan'] = merge.groupby(['code'])['spec_ment_loan'].fillna(method='ffill')
        # 合并 net equity
        net_equity = self.influx.getDataMultiprocess('FinancialReport_Gus', 'net_equity', adj_start, end,
                                                     ['code', 'net_equity'])
        net_equity.index.names = ['date']
        merge = pd.merge(merge, net_equity.reset_index(), on=['date', 'code'])
        # 计算 资本约束及稳健性指标 * 资本回拨系数
        # 资本约束及稳健性指标 = (core_CA_ratio - 8.5) / 100 * 2 + 1
        # 资本回拨系数 = (拨备总额 - 0.8 * 不良总额 - 0.4 * 关注总额) * 0.75 / net_equity + 1
        merge['robust_coef'] = (merge['core_CA_ratio'] - 8.5) / 100 * 2 + 1
        merge['callback_coef'] = (merge['provision_amount'] - 0.8 * merge['bad_loan'] - 0.4 * merge['spec_ment_loan']) \
                                 * 0.75 / merge['net_equity'] + 1

        # 合并 ROE，BP 计算 yunmeng
        ROE = self.influx.getDataMultiprocess('DailyFactors_Gus', 'ROE2', adj_start, end)
        ROE.index.names = ['date']
        BP = self.influx.getDataMultiprocess('DailyFactors_Gus', 'BP', adj_start, end, ['code', 'BP'])
        BP.index.names = ['date']
        # ROE, BP 不需要对齐report_period
        merge = pd.merge(merge, ROE.reset_index(), how='left', on=['date', 'code'])
        merge = pd.merge(merge, BP.reset_index(), how='left', on=['date', 'code'])
        merge[
            ['ROE', 'ROE_last1Q', 'ROE_last2Q', 'ROE_last3Q', 'ROE_last4Q', 'ROE_last5Q',
             'ROE_last6Q', 'ROE_last7Q', 'ROE_last8Q']] = merge[
            ['ROE', 'ROE_last1Q', 'ROE_last2Q', 'ROE_last3Q', 'ROE_last4Q', 'ROE_last5Q',
             'ROE_last6Q', 'ROE_last7Q', 'ROE_last8Q']].fillna(method='bfill', axis=1)
        merge[
            ['ROE', 'ROE_last1Q', 'ROE_last2Q', 'ROE_last3Q', 'ROE_last4Q', 'ROE_last5Q',
             'ROE_last6Q', 'ROE_last7Q', 'ROE_last8Q']] = merge[
            ['ROE', 'ROE_last1Q', 'ROE_last2Q', 'ROE_last3Q', 'ROE_last4Q', 'ROE_last5Q',
             'ROE_last6Q', 'ROE_last7Q', 'ROE_last8Q']].fillna(method='ffill', axis=1)
        merge['wgt_ROE'] = (3 * merge['ROE'] + 2 * merge['ROE_last4Q'] + merge['ROE_last8Q']) / 6
        merge = merge.sort_values('date')
        merge['robust_coef'] = merge.groupby('code')['robust_coef'].fillna(method='ffill')
        merge['wgt_ROE'] = merge.groupby('code')['wgt_ROE'].fillna(method='ffill')
        merge['BP'] = merge.groupby('code')['BP'].fillna(method='ffill')
        # 计算 云蒙 = (加权ROE * 200 + 40) * 资本约束及稳健性指标 * 资本回拨系数 * BP
        merge['yunmeng'] = (merge['wgt_ROE'] * 200 + 40) * merge['callback_coef'] * merge['robust_coef'] * merge['BP']
        merge.set_index('date', inplace=True)
        merge = merge.loc[str(start):str(end), ['code', 'yunmeng']]
        merge = merge.loc[pd.notnull(merge['yunmeng']), :]
        merge = merge.where(pd.notnull(merge), None)
        codes = merge['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (merge, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('yunmeng finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list

if __name__ == '__main__':
    pd.set_option('mode.use_inf_as_na', True)
    start_dt = datetime.datetime.now()
    ym = YunMeng()
    r = ym.cal_factors(20090101, 20200601, N_JOBS)
    print(r)
    print('time token:', datetime.datetime.now()-start_dt)