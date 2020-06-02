from factor_base import FactorBase
import pandas as pd
import numpy as np
import datetime
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData
from global_constant import N_JOBS


class LoanClassification(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'FinancialReport_Gus'
        self.measure = 'loan_classification'

    def cal_factors(self, n_jobs):
        fail_list = []
        query = "select S_INFO_COMPCODE, REPORT_PERIOD, LOAN_TYPE, TOTAL_AMOUNT " \
                "from wind_filesync.BankLoan5LClassification " \
                "order by report_period "
        self.rdf.curs.execute(query)
        loan = pd.DataFrame(self.rdf.curs.fetchall(),
                            columns=['comp_code', 'report_period', 'loan_type', 'total_amount'])
        query = "select S_INFO_COMPCODE, S_INFO_WINDCODE " \
                "from wind_filesync.WindCustomCode " \
                "where (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "and S_INFO_SECURITIESTYPES = 'A'"
        self.rdf.curs.execute(query)
        code2code = pd.DataFrame(self.rdf.curs.fetchall(), columns=['comp_code', 'code'])
        loan = pd.merge(loan, code2code, on='comp_code')
        # 统计不良贷款
        bad_loan = loan.loc[loan['loan_type'].isin(['次级', '可疑', '损失']), :].copy()
        bad_loan = bad_loan.groupby(['code', 'report_period'])['total_amount'].sum()
        bad_loan.name = 'bad_loan'
        bad_loan = pd.DataFrame(bad_loan)
        bad_loan = bad_loan.reset_index()
        spec_ment_loan = loan.loc[loan['loan_type'] == '关注', ['code', 'report_period', 'total_amount']].copy()
        spec_ment_loan.rename(columns={'total_amount': 'spec_ment_loan'}, inplace=True)
        merge = pd.merge(bad_loan, spec_ment_loan, on=['code', 'report_period'], how='outer')
        merge['report_period'] = pd.to_datetime(merge['report_period'])
        merge.set_index('report_period', inplace=True)
        merge = merge.where(pd.notnull(merge), None)
        codes = bad_loan['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (merge, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('Loan Classification finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    npl = LoanClassification()
    r = npl.cal_factors(N_JOBS)
    print(r)