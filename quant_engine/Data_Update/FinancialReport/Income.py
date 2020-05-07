from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData
from global_constant import N_JOBS

class IncomeUpdate(FactorBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_former_data(series, n_Qs):
        report_period = FactorBase.get_former_report_period(series['report_period'], n_Qs)
        if report_period not in series.index:
            return np.nan
        else:
            return series[report_period]

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
                save_res.append('Income  Field: %s  Error: %s' % (field, r))
        return save_res


    def cal_factors(self, start, end, n_jobs):
        # type要用408001000，408005000，408004000(合并报表，合并更正前，合并调整后)，同时有408001000和408005000用408005000
        # 有408004000时，根据ann_dt酌情使用
        # 目前包含字段: 净利润(net_profit)，扣非净利润(net_profit_ddt)，营收(oper_rev)，总营收(tot_oper_rev)，
        #              营业利润(oper_profit)，摊薄eps(EPS_diluted)，经营利润(oper_income)
        query = "select ANN_DT, S_INFO_WINDCODE, REPORT_PERIOD, NET_PROFIT_EXCL_MIN_INT_INC, " \
                "NET_PROFIT_AFTER_DED_NR_LP, OPER_REV, TOT_OPER_REV, TOT_OPER_COST, OPER_PROFIT, TOT_PROFIT, " \
                "S_FA_EPS_DILUTED, MINORITY_INT_INC, LESS_FIN_EXP, NET_INT_INC, STATEMENT_TYPE " \
                "from wind_filesync.AShareIncome " \
                "where ANN_DT >= {0} and ANN_DT <= {1} " \
                "and (STATEMENT_TYPE = '408001000' or STATEMENT_TYPE = '408005000' or STATEMENT_TYPE = '408004000') " \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "order by report_period, ann_dt, statement_type " \
            .format((dtparser.parse(str(start)) - relativedelta(years=4)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        income = \
            pd.DataFrame(self.rdf.curs.fetchall(),
                         columns=['date', 'code', 'report_period', 'net_profit', 'net_profit_ddt', 'oper_rev',
                                  'tot_oper_rev', 'tot_oper_cost', 'oper_profit', 'tot_profit', 'EPS_diluted',
                                  'minority_interest_income', 'less_fin_exp', 'net_interest_income', 'type'])
        income[['minority_interest_income', 'less_fin_exp', 'net_interest_income']] = \
            income[['minority_interest_income', 'less_fin_exp', 'net_interest_income']].fillna(0)
        # 同一code，同一date，同一report_period，同时出现type1，2，3时，取type大的
        income['type'] = income['type'].apply(lambda x: '2' if x == '408001000' else ('3' if x == '408005000' else '4'))
        income = income.sort_values(by=['code', 'date', 'report_period', 'type'])
        income['date'] = pd.to_datetime(income['date'])
        income['report_period'] = pd.to_datetime((income['report_period']))
        # ***************************************************************************
        # 读取业绩快报
        query = "select ANN_DT, S_INFO_WINDCODE, REPORT_PERIOD, OPER_REV, OPER_PROFIT, NET_PROFIT_EXCL_MIN_INT_INC, " \
                "TOT_PROFIT, EPS_DILUTED " \
                "from wind_filesync.AShareProfitExpress " \
                "where ANN_DT >= {0} and ANN_DT <= {1} " \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "order by report_period, ann_dt" \
            .format((dtparser.parse(str(start)) - relativedelta(years=4)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        express = pd.DataFrame(self.rdf.curs.fetchall(),
                               columns=['date', 'code', 'report_period', 'oper_rev', 'oper_profit', 'net_profit',
                                        'tot_profit', 'EPS_diluted'])
        express['date'] = pd.to_datetime(express['date'])
        express['report_period'] = pd.to_datetime(express['report_period'])
        express['type'] = '1'
        # ***************************************************************************
        # 读取业绩预告
        query = "select S_PROFITNOTICE_DATE, S_INFO_WINDCODE, S_PROFITNOTICE_PERIOD, S_PROFITNOTICE_NETPROFITMIN, " \
                "S_PROFITNOTICE_NETPROFITMAX " \
                "from wind_filesync.AShareProfitNotice " \
                "where S_PROFITNOTICE_DATE >= {0} and S_PROFITNOTICE_DATE <= {1} " \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "order by S_PROFITNOTICE_PERIOD, S_PROFITNOTICE_DATE" \
            .format((dtparser.parse(str(start)) - relativedelta(years=4)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        notice = pd.DataFrame(self.rdf.curs.fetchall(),
                              columns=['date', 'code', 'report_period', 'net_profit_min', 'net_profit_max'])
        notice['date'] = pd.to_datetime(notice['date'])
        notice['report_period'] = pd.to_datetime(notice['report_period'])
        notice['type'] = '0'
        notice[['net_profit_min', 'net_profit_max']] = \
            notice[['net_profit_min', 'net_profit_max']].fillna(method='bfill', axis=1)
        notice[['net_profit_min', 'net_profit_max']] = \
            notice[['net_profit_min', 'net_profit_max']].fillna(method='ffill', axis=1)
        # 业绩预告的单位为： 万元
        notice['net_profit'] = (0.5 * notice['net_profit_min'] + 0.5 * notice['net_profit_max']) * 10000
        notice.drop(['net_profit_min', 'net_profit_max'], axis=1, inplace=True)
        # ***************************************************************************
        income = pd.concat([income, express, notice], ignore_index=True)
        income = income.sort_values(by=['code', 'date', 'report_period', 'type'])
        # 经营利润 = 净利润（含少数股东损益） - 非经常性损益 + 财务费用 * (1-0.25) - 利息净收入 * (1-0.25)
        #         = 扣非净利润（扣除少数股东损益） + 少数股东损益 + 财务费用 * (1-0.25) - 利息净收入 * (1-0.25)
        income['oper_income'] = income['net_profit_ddt'] + income['minority_interest_income'] + \
                                income['less_fin_exp'] * (1 - 0.25) - income['net_interest_income'] * (1 - 0.25)
        income['gross_margin'] = income['tot_oper_rev'] - income['tot_oper_cost']
        # 需要的field
        fields = ['net_profit', 'net_profit_ddt', 'oper_rev', 'tot_oper_rev', 'tot_oper_cost', 'oper_profit',
                  'tot_profit', 'gross_margin', 'EPS_diluted', 'oper_income']
        # 处理数据
        calendar = self.rdf.get_trading_calendar()
        calendar = \
            set(calendar.loc[(calendar >= (dtparser.parse(str(start)) - relativedelta(years=2)).strftime('%Y%m%d')) &
                             (calendar <= str(end))])
        # 存放的db
        save_db = 'FinancialReport_Gus'
        fail_list = []
        for f in fields:
            print('ALL ANNOUNCEMENT \n field: %s begins processing...' % f)
            df = pd.DataFrame(income.dropna(subset=[f]).groupby(['code', 'date', 'report_period'])[f].last()) \
                .reset_index()
            df = df.sort_values(by=['report_period', 'date'])
            df.set_index(['code', 'date', 'report_period'], inplace=True)
            df = df.unstack(level=2)
            df = df.loc[:, f]
            df = df.reset_index().set_index('date')
            codes = df['code'].unique()
            split_codes = np.array_split(codes, n_jobs)
            with parallel_backend('multiprocessing', n_jobs=n_jobs):
                res = Parallel()(delayed(IncomeUpdate.JOB_factors)
                                 (df, f, codes, calendar, start, save_db) for codes in split_codes)
            print('%s finish' % f)
            print('-' * 30)
            for r in res:
                fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    IU = IncomeUpdate()
    r = IU.cal_factors(20100101, 20200501, N_JOBS)