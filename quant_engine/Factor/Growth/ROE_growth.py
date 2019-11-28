from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
import datetime
from influxdb_data import influxdbData
from joblib import Parallel,delayed,parallel_backend

class ROE_growth(FactorBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_former_date(date,calendar,months):
        calendar = calendar[calendar<=date-relativedelta(months=months)]
        if calendar.empty:
            return np.nan
        else:
            return calendar.iloc[-1]

    @staticmethod
    def job_factors(code,ROE):
        code_ROE = ROE.loc[ROE['code']==code,:].sort_index().copy()
        code_ROE.index.names = ['date']
        code_ROE.reset_index(inplace=True)
        init_ROE = code_ROE.loc[:,['date','ROE','ROE_Q','ROE_ddt','ROE_ddt_Q']].copy()
        dates = code_ROE['date']
        code_ROE['date_lastQ'] = code_ROE.apply(lambda row:ROE_growth.get_former_date(row['date'],dates,3),axis=1)
        code_ROE['date_lastY'] = code_ROE.apply(lambda row:ROE_growth.get_former_date(row['date'],dates,12),axis=1)

        # 只要date_lastQ不全为nan，则计算growth
        if pd.isnull(code_ROE['date_lastQ']).all():
            code_ROE['ROE_growthQ'] = np.nan
            code_ROE['ROE_Q_growthQ'] = np.nan
            code_ROE['ROE_ddt_growthQ'] = np.nan
            code_ROE['ROE_ddt_Q_growthQ'] = np.nan
        else:
            init_ROE.columns = ['date_lastQ','ROE_lastQ','ROE_Q_lastQ','ROE_ddt_lastQ','ROE_ddt_Q_lastQ']
            code_ROE = pd.merge(code_ROE,init_ROE,on='date_lastQ',how='left')
            code_ROE['ROE_growthQ'] = \
                code_ROE.apply(lambda row: ROE_growth.cal_growth(row['ROE_lastQ'], row['ROE']),
                              axis=1).fillna(method='ffill')
            code_ROE['ROE_Q_growthQ'] = \
                code_ROE.apply(lambda row: ROE_growth.cal_growth(row['ROE_Q_lastQ'], row['ROE_Q']),
                              axis=1).fillna(method='ffill')
            code_ROE['ROE_ddt_growthQ'] = \
                code_ROE.apply(lambda row: ROE_growth.cal_growth(row['ROE_ddt_lastQ'], row['ROE_ddt']),
                              axis=1).fillna(method='ffill')
            code_ROE['ROE_ddt_Q_growthQ'] = \
                code_ROE.apply(lambda row: ROE_growth.cal_growth(row['ROE_ddt_Q_lastQ'], row['ROE_ddt_Q']),
                              axis=1).fillna(method='ffill')

        # 只要date_lastQ不全为nan，则计算growth
        if pd.isnull(code_ROE['date_lastY']).all():
            code_ROE['ROE_growthY'] = np.nan
            code_ROE['ROE_Q_growthY'] = np.nan
            code_ROE['ROE_ddt_growthY'] = np.nan
            code_ROE['ROE_ddt_Q_growthY'] = np.nan
        else:
            init_ROE.columns = ['date_lastY', 'ROE_lastY', 'ROE_Q_lastY', 'ROE_ddt_lastY', 'ROE_ddt_Q_lastY']
            code_ROE = pd.merge(code_ROE, init_ROE, on='date_lastY', how='left')
            code_ROE['ROE_growthY'] = \
                code_ROE.apply(lambda row: ROE_growth.cal_growth(row['ROE_lastY'], row['ROE']),
                               axis=1).fillna(method='ffill')
            code_ROE['ROE_Q_growthY'] = \
                code_ROE.apply(lambda row: ROE_growth.cal_growth(row['ROE_Q_lastY'], row['ROE_Q']),
                               axis=1).fillna(method='ffill')
            code_ROE['ROE_ddt_growthY'] = \
                code_ROE.apply(lambda row: ROE_growth.cal_growth(row['ROE_ddt_lastY'], row['ROE_ddt']),
                               axis=1).fillna(method='ffill')
            code_ROE['ROE_ddt_Q_growthY'] = \
                code_ROE.apply(lambda row: ROE_growth.cal_growth(row['ROE_ddt_Q_lastY'], row['ROE_ddt_Q']),
                               axis=1).fillna(method='ffill')

        code_ROE = code_ROE.loc[pd.notnull(code_ROE['ROE_growthQ'])      |pd.notnull(code_ROE['ROE_growthY'])    |
                                pd.notnull(code_ROE['ROE_Q_growthQ'])    |pd.notnull(code_ROE['ROE_Q_growthY'])  |
                                pd.notnull(code_ROE['ROE_ddt_growthQ'])  |pd.notnull(code_ROE['ROE_ddt_growthY'])|
                                pd.notnull(code_ROE['ROE_ddt_Q_growthQ'])|pd.notnull(code_ROE['ROE_ddt_Q_growthY']),
                                ['date','code','ROE_growthQ','ROE_growthY','ROE_Q_growthQ','ROE_Q_growthY',
                                 'ROE_ddt_growthQ','ROE_ddt_growthY','ROE_ddt_Q_growthQ','ROE_ddt_Q_growthY']]
        code_ROE.set_index('date',inplace=True)
        code_ROE = code_ROE.where(pd.notnull(code_ROE), None)
        print(code)
        influx = influxdbData()
        influx.saveData(code_ROE,'DailyFactor_Gus','Growth')


    def cal_factors(self,start,end):
        print('task start!')
        start = (dtparser.parse(str(start))-relativedelta(years=1)).strftime('%Y%m%d')
        end = str(end)
        start_time = datetime.datetime.now()
        print('Start time: ',start_time)
        ROE = self.influx.getDataMultiprocess('DailyFactor_Gus','FinancialQuality',start,end,
                                              ['code','ROE','ROE_Q','ROE_ddt','ROE_ddt_Q'])
        print('ROE data got')
        codes = ROE['code'].unique()
        Parallel(n_jobs=-1,verbose=0)(delayed(ROE_growth.job_factors)(code,ROE) for code in codes)
        print('task finish!')
        print('Time token: ',datetime.datetime.now()-start_time)



if __name__ == '__main__':
    pd.set_option('mode.use_inf_as_na', True)
    ROEg = ROE_growth()
    ROEg.cal_factors(20100101,20190901)