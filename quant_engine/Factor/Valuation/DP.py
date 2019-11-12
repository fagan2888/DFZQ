from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
import datetime
import joblib

# DP_TTM = 近12个月现金红利(按除息日计)/总市值

class DP(FactorBase):
    def __init__(self):
        super().__init__()

    def job_factors(self,code,mv,dvd,start,end):
        code_dvd = dvd.loc[dvd['code']==code,:].copy()
        code_mv = mv.loc[mv['code']==code,:].copy()
        code_mv = pd.merge(code_mv, code_dvd, how='outer', left_on=['date', 'code'], right_on=['date', 'code'])
        code_mv['dvd_amount'] = code_mv['dvd_per_share']*code_mv['shares']
        code_mv.set_index('date',inplace=True)
        code_mv = code_mv.sort_index()

        dvd_12m = []
        for idx, row in code_mv.iterrows():
            date_12m_ago = idx - relativedelta(years=1)
            dvd_12m.append(code_mv.loc[date_12m_ago:idx, 'dvd_amount'].sum())
        code_mv['dvd_12m'] = pd.Series(dvd_12m, index=code_mv.index)
        code_mv['DP_TTM'] = code_mv['dvd_12m']/code_mv['mv']
        code_mv = code_mv.dropna(subset=['mv'])
        code_mv = code_mv.loc[str(start):str(end),['code','DP_TTM']]
        print(code)
        self.influx.saveData(code_mv, 'DailyFactor_Gus', 'Value')
        return code_mv


    def cal_factors(self,start,end):
        query = "select S_INFO_WINDCODE, CASH_DVD_PER_SH_PRE_TAX, EX_DT, REPORT_PERIOD " \
                "from wind_filesync.AShareDividend " \
                "where S_DIV_PROGRESS = 3 " \
                "and REPORT_PERIOD >= {0} and REPORT_PERIOD <= {1}" \
                "and (s_info_windcode like '0%' " \
                "or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "order by report_period" \
            .format((dtparser.parse(str(start)) - relativedelta(years=1)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        dvd = pd.DataFrame(self.rdf.curs.fetchall(),columns=['code','dvd_per_share','date','report_period'])
        dvd = dvd.loc[dvd['dvd_per_share']>0,:]
        dvd['report_period'] = pd.to_datetime(dvd['report_period'])
        # 填充错误数据
        dvd.loc[pd.isnull(dvd['date']),'date'] = dvd.loc[pd.isnull(dvd['date']),'report_period']
        dvd['report_period'] = dvd['report_period'].apply(
            lambda x: datetime.datetime(year=x.year - 1, month=12, day=31) if x.day < 30 else x)
        dvd['date'] = pd.to_datetime(dvd['date'])

        query = "select TRADE_DT, S_INFO_WINDCODE, S_VAL_MV, TOT_SHR_TODAY " \
                "from wind_filesync.AShareEODDerivativeIndicator " \
                "where TRADE_DT >= {0} and TRADE_DT <= {1} " \
                "and (s_info_windcode like '0%' " \
                "or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "order by TRADE_DT" \
            .format((dtparser.parse(str(start)) - relativedelta(years=1)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        mv = pd.DataFrame(self.rdf.curs.fetchall(), columns=['date', 'code', 'mv', 'shares'])
        mv['date'] = pd.to_datetime(mv['date'])
        print('Data loaded!')

        joblib.Parallel()(joblib.delayed(self.job_factors)(code, mv, dvd, start, end)
                          for code in mv['code'].unique())


if __name__ == '__main__':
    print(datetime.datetime.now())
    dp = DP()
    dp.cal_factors(20100101,20190901)
    print('task finish')
    print(datetime.datetime.now())