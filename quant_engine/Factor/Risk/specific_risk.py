from DFQ_risk import dfq_risk
from influxdb_data import influxdbData
from rdf_data import rdf_data
import pandas as pd
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS, ROOT_DIR


class SpecificRisk:
    def __init__(self):
        self.rdf = rdf_data()
        self.n_jobs = N_JOBS
        self.db = 'DailyFactors_Gus'
        self.measure = 'SpecificRisk'

    @staticmethod
    def JOB_factors(dates, db, measure):
        DFQr = dfq_risk()
        influx = influxdbData()
        table = 'specificriskhuber'
        field = 'sriskjson'
        uni = '000000'
        save_res = []
        for date in dates:
            query = "SELECT {0} FROM dfrisk.{1} WHERE tradingdate ='{2}' and universe='{3}'" \
                .format(field, table, date, uni)
            DFQr.cur.execute(query)
            try:
                day_df = pd.read_json(DFQr.cur.fetchone()[0], orient='split', convert_axes=False, typ='series')
                day_df.name = 'specific_risk'
                day_df = pd.DataFrame(day_df)
                day_df['date'] = pd.to_datetime(date)
                day_df.index.names = ['code']
                day_df = day_df.reset_index().set_index('date')
                day_df['code'] = \
                    np.where(day_df['code'].str[0] == '6', day_df['code'] + '.SH', day_df['code'] + '.SZ')
                day_df = day_df.dropna(subset=['specific_risk'])
                # save
                print('date: %s' % date)
                r = influx.saveData(day_df, db, measure)
                if r == 'No error occurred...':
                    pass
                else:
                    save_res.append('%s Error: %s' % ('SpecificRisk', r))
            except TypeError:
                save_res.append('%s Error from DB! Date: %s' % ('SpecificRisk', date))
        return save_res


    def cal_factors(self, start, end):
        calendar = self.rdf.get_trading_calendar()
        calendar = calendar[(calendar >= str(start)) & (calendar <= str(end))]
        calendar = pd.DatetimeIndex(calendar).strftime('%Y%m%d')
        save_res = SpecificRisk.JOB_factors(calendar, self.db, self.measure)
        print('SpecificRisk finish')
        print('-' * 30)
        return save_res


if __name__ == '__main__':
    rfe = SpecificRisk()
    rfe.cal_factors(20200518, 20200522)