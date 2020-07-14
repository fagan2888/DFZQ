from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData
from global_constant import N_JOBS


class Audit(FactorBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def JOB_factors(df, codes, calendar, save_db):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            insert_dates = calendar - set(code_df.index)
            content = [[np.nan] * 3] * len(insert_dates)
            insert_df = pd.DataFrame(content, columns=['code', 'report_period', 'audit_opinion'],
                                     index=list(insert_dates))
            code_df = code_df.append(insert_df, ignore_index=False).sort_index()
            code_df = code_df.fillna(method='ffill')
            code_df = code_df.dropna()
            print('code: %s' % code)
            r = influx.saveData(code_df, save_db, 'Audit')
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('Audit  Error: %s' % r)
        return save_res

    def cal_factors(self, start, end, n_jobs):
        query = "select ANN_DT, S_INFO_WINDCODE, REPORT_PERIOD, S_STMNOTE_AUDIT_CATEGORY " \
                "from wind_filesync.AShareAuditOpinion " \
                "where ANN_DT >= {0} and ANN_DT <= {1} " \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '6%') "\
            .format((dtparser.parse(str(start)) - relativedelta(years=4)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        audit = pd.DataFrame(self.rdf.curs.fetchall(), columns=['date', 'code', 'report_period', 'audit_opinion'])
        # 数值越大评级越低
        audit = audit.replace(405001000, 1)
        audit = audit.replace(405002000, 2)
        audit = audit.replace(405010000, 3)
        audit = audit.replace(405003000, 4)
        audit = audit.replace(405004000, 5)
        audit = audit.replace(405005000, 6)
        calendar = self.rdf.get_trading_calendar()
        calendar = \
            set(calendar.loc[(calendar >= (dtparser.parse(str(start)) - relativedelta(years=1)).strftime('%Y%m%d')) &
                             (calendar <= str(end))])
        audit = audit.sort_values(['date', 'code'])
        audit = audit.groupby(['date', 'code']).last().reset_index()
        audit['date'] = pd.to_datetime(audit['date'])
        audit = audit.set_index('date')
        codes = audit['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        save_db = 'DailyFactors_Gus'
        fail_list = []
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(Audit.JOB_factors)
                             (audit, codes, calendar, save_db) for codes in split_codes)
        print('Audit finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)

        return fail_list


if __name__ == '__main__':
    ao = Audit()
    r = ao.cal_factors(20090101, 20200712, N_JOBS)