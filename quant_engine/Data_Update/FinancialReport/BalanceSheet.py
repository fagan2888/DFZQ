from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData
from global_constant import N_JOBS


class BalanceSheetUpdate(FactorBase):
    def __init__(self):
        super().__init__()

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
                save_res.append('BalanceSheet  Field: %s  Error: %s' % (field, r))
        return save_res

    def cal_factors(self, start, end, n_jobs):
        # 读取财报
        # type要用408001000，408005000，408004000(合并报表，合并更正前，合并调整后)，同时有408001000和408005000用408005000
        # 有408004000时，根据ann_dt酌情使用
        query = "select ANN_DT, S_INFO_WINDCODE, REPORT_PERIOD, TOT_ASSETS, TOT_LIAB, " \
                "TOT_SHRHLDR_EQY_EXCL_MIN_INT, TOT_SHRHLDR_EQY_INCL_MIN_INT, ST_BORROW, TRADABLE_FIN_LIAB, " \
                "NOTES_PAYABLE, NON_CUR_LIAB_DUE_WITHIN_1Y, LT_BORROW, BONDS_PAYABLE, " \
                "MONETARY_CAP, TRADABLE_FIN_ASSETS, FIN_ASSETS_AVAIL_FOR_SALE, HELD_TO_MTY_INVEST, " \
                "INVEST_REAL_ESTATE, TIME_DEPOSITS, OTH_ASSETS, LONG_TERM_REC, STATEMENT_TYPE " \
                "from wind_filesync.AShareBalanceSheet " \
                "where ANN_DT >= {0} and ANN_DT <= {1} " \
                "and (STATEMENT_TYPE = '408001000' or STATEMENT_TYPE = '408005000' or STATEMENT_TYPE = '408004000') " \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "order by report_period, ann_dt, statement_type " \
            .format((dtparser.parse(str(start)) - relativedelta(years=4)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        balance_sheet = \
            pd.DataFrame(self.rdf.curs.fetchall(),
                         columns=['date', 'code', 'report_period', 'tot_assets', 'tot_liab', 'net_equity',
                                  'net_equity_incl_min', 'shortterm_borrow', 'tradable_fin_liab', 'notes_payable',
                                  'non_cur_liab_within1Y', 'longterm_borrow', 'bonds_payable', 'monetary_cap',
                                  'tradable_fin_assets', 'fin_assets_avail_for_sale', 'held_to_maturity_invest',
                                  'invest_real_estate', 'time_deposits', 'other_assets', 'longterm_rec', 'type'])
        # 同一code，同一date，同一report_period，同时出现type1，2，3时，取type大的
        balance_sheet = balance_sheet.fillna(0)
        balance_sheet['type'] = \
            balance_sheet['type'].apply(lambda x: '2' if x == '408001000' else ('3' if x == '408005000' else '4'))
        balance_sheet = balance_sheet.sort_values(by=['code', 'date', 'report_period', 'type'])
        balance_sheet['date'] = pd.to_datetime(balance_sheet['date'])
        balance_sheet['report_period'] = pd.to_datetime(balance_sheet['report_period'])
        # ***************************************************************************
        # 读取业绩快报
        query = "select ANN_DT, S_INFO_WINDCODE, REPORT_PERIOD, TOT_ASSETS, TOT_SHRHLDR_EQY_EXCL_MIN_INT " \
                "from wind_filesync.AShareProfitExpress " \
                "where ANN_DT >= {0} and ANN_DT <= {1} " \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "order by report_period, ann_dt" \
            .format((dtparser.parse(str(start)) - relativedelta(years=4)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        express = pd.DataFrame(self.rdf.curs.fetchall(),
                               columns=['date', 'code', 'report_period', 'tot_assets', 'net_equity'])
        express['date'] = pd.to_datetime(express['date'])
        express['report_period'] = pd.to_datetime(express['report_period'])
        express['type'] = '1'
        # ***************************************************************************
        balance_sheet = pd.concat([balance_sheet, express], ignore_index=True)
        balance_sheet = balance_sheet.sort_values(by=['code', 'date', 'report_period', 'type'])
        # NOA 为净经营资产
        # NOA = 经营资产 - 经营负债
        #     = 股东权益 + 金融负债 - 金融资产
        #     = 股东权益合计 + 短期借款 + 交易性金融负债 + 应付票据 + 一年内到期的非流动负债 + 长期借款 + 应付债券
        #       - 货币资金 - 交易性金融资产 - 可供出售金融资产 - 持有至到期投资 - 投资性房地产 - 定期存款 - 其他资产 - 长期应收款
        balance_sheet['NOA'] = balance_sheet['net_equity_incl_min'] + balance_sheet['shortterm_borrow'] + \
                               balance_sheet['tradable_fin_liab'] + balance_sheet['notes_payable'] + \
                               balance_sheet['non_cur_liab_within1Y'] + balance_sheet['longterm_borrow'] + \
                               balance_sheet['bonds_payable'] - balance_sheet['monetary_cap'] - \
                               balance_sheet['tradable_fin_assets'] - balance_sheet['fin_assets_avail_for_sale'] - \
                               balance_sheet['held_to_maturity_invest'] - balance_sheet['invest_real_estate'] - \
                               balance_sheet['time_deposits'] - balance_sheet['other_assets'] - \
                               balance_sheet['longterm_rec']
        # 需要的field
        fields = ['NOA', 'tot_assets', 'tot_liab', 'net_equity']
        # fillna
        balance_sheet[fields] = balance_sheet.groupby('code')[fields].fillna(method='ffill')
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
            df = pd.DataFrame(balance_sheet.dropna(subset=[f]).groupby(['code', 'date', 'report_period'])[f].last()) \
                .reset_index()
            df = df.sort_values(by=['report_period', 'date'])
            df.set_index(['code', 'date', 'report_period'], inplace=True)
            df = df.unstack(level=2)
            df = df.loc[:, f]
            df = df.reset_index().set_index('date')
            codes = df['code'].unique()
            split_codes = np.array_split(codes, n_jobs)
            with parallel_backend('multiprocessing', n_jobs=n_jobs):
                res = Parallel()(delayed(BalanceSheetUpdate.JOB_factors)
                                 (df, f, codes, calendar, start, save_db) for codes in split_codes)
            print('%s finish' % f)
            print('-' * 30)
            for r in res:
                fail_list.extend(r)
        # 业绩快报值包含净利润信息，所以此处不需读取业绩预告
        return fail_list


if __name__ == '__main__':
    BU = BalanceSheetUpdate()
    r = BU.cal_factors(20100101, 20200315, N_JOBS)