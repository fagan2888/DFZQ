from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData
from global_constant import N_JOBS


class LoanClassification(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'FinancialReport_Gus'

    @staticmethod
    def JOB_factors(df, codes, calendar, start, save_field, save_db, save_msr):
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
            rp_keys = np.flipud(code_df.columns[1:])
            # 选择最新的report_period
            code_df['report_period'] = code_df.apply(lambda row: row.dropna().index[-1], axis=1)
            choices = []
            for rp in rp_keys:
                choices.append(code_df[rp].values)
            # 计算 当期
            code_rps = code_df['report_period'].unique()
            process_rps = []
            for code_rp in code_rps:
                process_rps.append(FactorBase.get_former_report_period(pd.to_datetime(code_rp), 0).strftime('%Y%m%d'))
            process_rp_dict = dict(zip(code_rps, process_rps))
            code_df['process_rp'] = code_df['report_period'].map(process_rp_dict)
            conditions = []
            for rp in rp_keys:
                conditions.append(code_df['process_rp'].values == rp)
            code_df[save_field] = np.select(conditions, choices, default=np.nan)
            # 计算过去每一季
            res_flds = [save_field]
            for i in range(1, 13):
                res_field = save_field + '_last{0}Q'.format(str(i))
                res_flds.append(res_field)
                process_rps = []
                for code_rp in code_rps:
                    process_rps.append(
                        FactorBase.get_former_report_period(pd.to_datetime(code_rp), i).strftime('%Y%m%d'))
                process_rp_dict = dict(zip(code_rps, process_rps))
                code_df['process_rp'] = code_df['report_period'].map(process_rp_dict)
                conditions = []
                for rp in rp_keys:
                    conditions.append(code_df['process_rp'].values == rp)
                code_df[res_field] = np.select(conditions, choices, default=np.nan)
            # 处理储存数据
            code_df = code_df.loc[:, ['code', 'report_period'] + res_flds]
            code_df = code_df.where(pd.notnull(code_df), None)
            print('code: %s' % code)
            r = influx.saveData(code_df, save_db, save_msr)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('LoanClassification Field: %s  Error: %s' % (save_field, r))
        return save_res


    def cal_factors(self, start, end, n_jobs):
        fail_list = []
        calendar = self.rdf.get_trading_calendar()
        calendar = set(calendar.loc[(calendar >= (dtparser.parse(str(start)) - relativedelta(years=2)).
                                     strftime('%Y%m%d')) & (calendar <= str(end))])
        query = "select OPDATE, S_INFO_COMPCODE, REPORT_PERIOD, LOAN_TYPE, TOTAL_AMOUNT " \
                "from wind_filesync.BankLoan5LClassification " \
                "order by report_period, OPDATE "
        self.rdf.curs.execute(query)
        loan = pd.DataFrame(self.rdf.curs.fetchall(),
                            columns=['date', 'comp_code', 'report_period', 'loan_type', 'total_amount'])
        query = "select S_INFO_COMPCODE, S_INFO_WINDCODE " \
                "from wind_filesync.WindCustomCode " \
                "where (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "and S_INFO_SECURITIESTYPES = 'A'"
        self.rdf.curs.execute(query)
        code2code = pd.DataFrame(self.rdf.curs.fetchall(), columns=['comp_code', 'code'])
        loan = pd.merge(loan, code2code, on='comp_code')
        loan.set_index('date', inplace=True)
        str_start = (pd.to_datetime(str(start)) - relativedelta(years=2)).strftime('%Y%m%d')
        loan = loan.loc[str_start:str(end), :]
        loan['date'] = pd.to_datetime(loan.index.strftime('%Y%m%d'))
        # 统计不良贷款
        bad_loan = loan.loc[loan['loan_type'].isin(['次级', '可疑', '损失']), :].copy()
        bad_loan_amount = bad_loan.groupby(['code', 'report_period'])['total_amount'].sum()
        bad_loan_date = bad_loan.groupby(['code', 'report_period'])['date'].last()
        bad_loan = pd.concat([bad_loan_amount, bad_loan_date], axis=1)
        bad_loan = bad_loan.reset_index().set_index(['code', 'date', 'report_period'])
        bad_loan = bad_loan.unstack(level=2)
        bad_loan = bad_loan['total_amount']
        bad_loan = bad_loan.reset_index().set_index('date')
        codes = bad_loan['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        save_msr = 'BadLoan'
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(LoanClassification.JOB_factors)
                             (bad_loan, codes, calendar, start, 'bad_loan', self.db, save_msr) for codes in split_codes)
        print('BadLoan finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)
        # 统计关注贷款
        special_mention_loan = \
            loan.loc[loan['loan_type'] == '关注', ['date', 'code', 'report_period', 'total_amount']].copy()
        special_mention_loan = special_mention_loan.set_index(['code', 'date', 'report_period'])
        special_mention_loan = special_mention_loan.unstack(level=2)
        special_mention_loan = special_mention_loan['total_amount']
        special_mention_loan = special_mention_loan.reset_index().set_index('date')
        codes = special_mention_loan['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        save_msr = 'SpecialMentionLoan'
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(LoanClassification.JOB_factors)
                             (special_mention_loan, codes, calendar, start, 'special_mention_loan', self.db, save_msr)
                             for codes in split_codes)
        print('SpecialMentionLoan finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    npl = LoanClassification()
    r = npl.cal_factors(20090101, 20200529, N_JOBS)
    print(r)