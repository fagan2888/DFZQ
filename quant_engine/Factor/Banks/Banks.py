from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData
from global_constant import N_JOBS


class Banks(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'FinancialReport_Gus'

    @staticmethod
    def JOB_factors(df, field, codes, calendar, start, save_db):
        columns = df.columns
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            insert_dates = calendar - set(code_df.index)
            content = [[np.nan] * len(columns)] * len(insert_dates)
            insert_df = pd.DataFrame(content, columns=columns, index=list(insert_dates))
            code_df = code_df.append(insert_df, ignore_index=False).sort_index()
            code_df = code_df.fillna(method='ffill')
            code_df = code_df.dropna(subset=['code'])
            code_df = code_df.loc[str(start):, ]
            # 所有report_period 为 columns, 去掉第一列(code)
            rps = np.flipud(code_df.columns[1:]).astype('datetime64[ns]')
            rp_keys = np.flipud(code_df.columns[1:])
            # 选择最新的report_period
            code_df['report_period'] = code_df.apply(lambda row: row.dropna().index[-1], axis=1)
            choices = []
            for rp in rp_keys:
                choices.append(code_df[rp].values)
            # 计算 当期 和 去年同期
            code_df['process_rp'] = code_df['report_period'].apply(lambda x: FactorBase.get_former_report_period(x, 0))
            conditions = []
            for rp in rps:
                conditions.append(code_df['process_rp'].values == rp)
            code_df[field] = np.select(conditions, choices, default=np.nan)
            code_df['process_rp'] = code_df['report_period'].apply(lambda x: FactorBase.get_former_report_period(x, 4))
            conditions = []
            for rp in rps:
                conditions.append(code_df['process_rp'].values == rp)
            code_df[field + '_lastY'] = np.select(conditions, choices, default=np.nan)
            # 计算过去每一季
            res_flds = []
            for i in range(1, 13):
                res_field = field + '_last{0}Q'.format(str(i))
                res_flds.append(res_field)
                code_df['process_rp'] = code_df['report_period'].apply(
                    lambda x: FactorBase.get_former_report_period(x, i))
                conditions = []
                for rp in rps:
                    conditions.append(code_df['process_rp'].values == rp)
                code_df[res_field] = np.select(conditions, choices, default=np.nan)
            # 处理储存数据
            code_df = code_df.loc[:, ['code', 'report_period', field, field + '_lastY'] + res_flds]
            code_df['report_period'] = code_df['report_period'].apply(lambda x: x.strftime('%Y%m%d'))
            code_df = code_df.where(pd.notnull(code_df), None)
            print('code: %s' % code)
            r = influx.saveData(code_df, save_db, field)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('Banks Field: %s  Error: %s' % (field, r))
        return save_res


    def cal_factors(self, start, end, n_jobs):
        # 读取财报
        # type要用408001000，408005000，408004000(合并报表，合并更正前，合并调整后)，同时有408001000和408005000用408005000
        # 有408004000时，根据ann_dt酌情使用
        query = "select ANN_DT, S_INFO_WINDCODE, REPORT_PERIOD, NPL_RATIO, NET_INTEREST_MARGIN, CAPI_ADE_RATIO ," \
                "CAPI_ADE_RATIO_2013, CORE_CAPI_ADE_RATIO, TIER1CAPI_ADE_RATIO, NPL_PROVISION_COVERAGE, " \
                "TOTAL_INTEREST_INCOME, TOTAL_LOAN, INTEREST_BEARING_ASSET, NON_INTEREST_MARGIN, " \
                "NON_INTEREST_INCOME, STATEMENT_TYPE " \
                "from wind_filesync.AShareBankIndicator " \
                "where ANN_DT >= {0} and ANN_DT <= {1} " \
                "and (STATEMENT_TYPE = '408001000' or STATEMENT_TYPE = '408005000' or STATEMENT_TYPE = '408004000')" \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "order by report_period, ann_dt, statement_type " \
            .format((dtparser.parse(str(start)) - relativedelta(years=4)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        bank = \
            pd.DataFrame(self.rdf.curs.fetchall(),
                         columns=['date', 'code', 'report_period', 'NPL', 'net_interest_margin', 'CA1', 'CA2',
                                  'core_CA1', 'core_CA2', 'provision_cov', 'interest_income', 'tot_loan',
                                  'interest_bearing_asset', 'non_interest_pct', 'non_interest_income', 'type'])
        bank['CA_ratio'] = np.where(pd.notnull(bank['CA2']).values, bank['CA2'].values, bank['CA1'].values)
        bank['core_CA_ratio'] = np.where(pd.notnull(bank['core_CA2']).values, bank['core_CA2'].values,
                                         bank['core_CA1'].values)
        # 同一code，同一date，同一report_period，同时出现type1，2，3时，取type大的
        bank['type'] = bank['type'].apply(lambda x: '2' if x == '408001000' else ('3' if x == '408005000' else '4'))
        bank = bank.sort_values(by=['code', 'date', 'report_period', 'type'])
        bank['date'] = pd.to_datetime(bank['date'])
        bank['report_period'] = pd.to_datetime(bank['report_period'])
        bank['tot_loan'] = bank.groupby(['code', 'report_period'])['tot_loan'].fillna(method='ffill')
        bank['NPL'] = bank.groupby(['code', 'report_period'])['NPL'].fillna(method='ffill')
        bank['provision_cov'] = bank.groupby(['code', 'report_period'])['provision_cov'].fillna(method='ffill')
        bank['bad_loan'] = bank['tot_loan'] * bank['NPL'] / 100
        bank['provision_amount'] = bank['bad_loan'] * bank['provision_cov'] / 100
        # ***************************************************************************
        # 需要的field
        fields = ['NPL', 'CA_ratio', 'core_CA_ratio', 'net_interest_margin', 'provision_cov', 'interest_income',
                  'tot_loan', 'bad_loan', 'provision_amount', 'interest_bearing_asset', 'non_interest_pct',
                  'non_interest_income']
        # 处理数据
        calendar = self.rdf.get_trading_calendar()
        calendar = \
            set(calendar.loc[(calendar >= (dtparser.parse(str(start)) - relativedelta(years=2)).strftime('%Y%m%d')) &
                             (calendar <= str(end))])
        # 存放的db
        fail_list = []
        for f in fields:
            print('BANKS \n field: %s begins processing...' % f)
            df = pd.DataFrame(bank.dropna(subset=[f]).groupby(['code', 'date', 'report_period'])[f].last()) \
                .reset_index()
            df = df.sort_values(by=['report_period', 'date'])
            df.set_index(['code', 'date', 'report_period'], inplace=True)
            df = df.unstack(level=2)
            df = df.loc[:, f]
            df = df.reset_index().set_index('date')
            codes = df['code'].unique()
            split_codes = np.array_split(codes, n_jobs)
            with parallel_backend('multiprocessing', n_jobs=n_jobs):
                res = Parallel()(delayed(Banks.JOB_factors)
                                 (df, f, codes, calendar, start, self.db) for codes in split_codes)
            print('%s finish' % f)
            print('-' * 30)
            for r in res:
                fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    npl = Banks()
    r = npl.cal_factors(20090101, 20200604, N_JOBS)