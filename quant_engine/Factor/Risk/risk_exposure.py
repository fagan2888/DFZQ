from DFQ_risk import dfq_risk
from influxdb_data import influxdbData
from rdf_data import rdf_data
import pandas as pd
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS, ROOT_DIR


class RiskFactorsExposure:
    def __init__(self):
        self.rdf = rdf_data()
        self.n_jobs = N_JOBS
        self.db = 'DailyFactors_Gus'
        self.measure = 'RiskExposure'

    @staticmethod
    def JOB_factors(dates, factor_dict, db, measure):
        DFQr = dfq_risk()
        influx = influxdbData()
        table = 'factorexposure'
        field = 'fvjson'
        uni = '000000'
        save_res = []
        for date in dates:
            query = "SELECT {0} FROM dfrisk.{1} WHERE tradingdate ='{2}' and universe='{3}'" \
                .format(field, table, date, uni)
            DFQr.cur.execute(query)
            try:
                day_df = pd.read_json(DFQr.cur.fetchone()[0], orient='split', convert_axes=False)
                day_df['date'] = pd.to_datetime(date)
                day_df.index.names = ['code']
                day_df = day_df.reset_index().set_index('date')
                day_df['code'] = \
                    np.where(day_df['code'].str[0] == '6', day_df['code'] + '.SH', day_df['code'] + '.SZ')
                day_df = day_df.rename(columns=factor_dict)
                day_df = day_df.where(pd.notnull(day_df), None)
                # save
                print('date: %s' % date)
                r = influx.saveData(day_df, db, measure)
                if r == 'No error occurred...':
                    pass
                else:
                    save_res.append('%s Error: %s' % ('RiskExposure', r))
            except TypeError:
                save_res.append('%s Error from DB! Date: %s' % ('RiskExposure', date))
        return save_res


    def cal_factors(self, start, end):
        calendar = self.rdf.get_trading_calendar()
        calendar = calendar[(calendar >= str(start)) & (calendar <= str(end))]
        calendar = pd.DatetimeIndex(calendar).strftime('%Y%m%d')
        factor_dict = pd.read_excel(ROOT_DIR + '/Data_Resource/风险因子说明.xlsx', header=None)
        factor_dict = dict(zip(factor_dict[0].values, factor_dict[1].values))
        save_res = RiskFactorsExposure.JOB_factors(calendar, factor_dict, self.db, self.measure)
        print('RiskExposure finish')
        print('-' * 30)
        return save_res

if __name__ == '__main__':
    rfe = RiskFactorsExposure()
    res = rfe.cal_factors(20200518, 20200522)
    print(res)