from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
import datetime
from influxdb_data import influxdbData
from joblib import Parallel,delayed,parallel_backend

class EP_growth(FactorBase):
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
    def job_factors(code,EP):
        code_EP = EP.loc[EP['code']==code,:].sort_index().copy()
        code_EP.index.names = ['date']
        code_EP.reset_index(inplace=True)
        init_EP = code_EP.loc[:,['date','EP_TTM','EPcut_TTM']].copy()
        dates = code_EP['date']
        code_EP['date_lastQ'] = code_EP.apply(lambda row:EP_growth.get_former_date(row['date'],dates,3),axis=1)
        code_EP['date_lastY'] = code_EP.apply(lambda row:EP_growth.get_former_date(row['date'],dates,12),axis=1)

        if pd.isnull(code_EP['date_lastQ']).all():
            code_EP['EP_TTM_growthQ'] = np.nan
            code_EP['EPcut_TTM_growthQ'] = np.nan
        else:
            init_EP.columns = ['date_lastQ','EP_TTM_lastQ','EPcut_TTM_lastQ']
            code_EP = pd.merge(code_EP,init_EP,on='date_lastQ',how='left')
            code_EP['EP_TTM_growthQ'] = \
                code_EP.apply(lambda row: EP_growth.cal_growth(row['EP_TTM_lastQ'], row['EP_TTM']),
                              axis=1).fillna(method='ffill')
            code_EP['EPcut_TTM_growthQ'] = \
                code_EP.apply(lambda row: EP_growth.cal_growth(row['EPcut_TTM_lastQ'], row['EPcut_TTM']),
                              axis=1).fillna(method='ffill')

        if pd.isnull(code_EP['date_lastY']).all():
            code_EP['EP_TTM_growthY'] = np.nan
            code_EP['EPcut_TTM_growthY'] = np.nan
        else:
            init_EP.columns = ['date_lastY', 'EP_TTM_lastY', 'EPcut_TTM_lastY']
            code_EP = pd.merge(code_EP, init_EP, on='date_lastY', how='left')
            code_EP['EP_TTM_growthY'] = \
                code_EP.apply(lambda row: EP_growth.cal_growth(row['EP_TTM_lastY'], row['EP_TTM']),
                              axis=1).fillna(method='ffill')
            code_EP['EPcut_TTM_growthY'] = \
                code_EP.apply(lambda row:EP_growth.cal_growth(row['EPcut_TTM_lastY'], row['EPcut_TTM']),
                              axis=1).fillna(method='ffill')

        code_EP = code_EP.loc[pd.notnull(code_EP['EP_TTM_growthQ'])|pd.notnull(code_EP['EP_TTM_growthY'])|
                              pd.notnull(code_EP['EPcut_TTM_growthQ'])|pd.notnull(code_EP['EPcut_TTM_growthY']),
                              ['date','code','EP_TTM_growthQ','EP_TTM_growthY','EPcut_TTM_growthQ','EPcut_TTM_growthY']]
        code_EP.set_index('date',inplace=True)
        code_EP = code_EP.where(pd.notnull(code_EP), None)
        influx = influxdbData()
        influx.saveData(code_EP,'DailyFactor_Gus','Growth')
        print('%s finish' %code)


    def cal_factors(self,start,end):
        print('task start!')
        start = (dtparser.parse(str(start))-relativedelta(years=1)).strftime('%Y%m%d')
        end = str(end)
        start_time = datetime.datetime.now()
        print('Start time: ',start_time)
        EP = self.influx.getDataMultiprocess('DailyFactor_Gus','Value',start,end,['code','EP_TTM','EPcut_TTM'])
        print('EP data got')
        codes = EP['code'].unique()
        with parallel_backend('multiprocessing', n_jobs=-1):
            Parallel()(delayed(EP_growth.job_factors)(code,EP) for code in codes)
        print('task finish!')
        print('Time token: ',datetime.datetime.now()-start_time)



if __name__ == '__main__':
    pd.set_option('mode.use_inf_as_na', True)
    epg = EP_growth()
    epg.cal_factors(20100101,20190901)