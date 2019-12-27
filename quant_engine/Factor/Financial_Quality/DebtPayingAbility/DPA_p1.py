# 偿债能力因子 current_ratio 的计算

from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
import datetime
import joblib


class DPA_p1(FactorBase):
    def __init__(self):
        super().__init__()

    def cal_factors(self,start,end):
        q_start = (dtparser.parse(str(start))-relativedelta(years=1)).strftime('%Y%m%d')
        query = "select ANN_DT, S_INFO_WINDCODE, S_FA_CURRENT " \
                "from wind_filesync.AShareFinancialIndicator " \
                "where ANN_DT >= {0} and ANN_DT <= {1} " \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '60%') " \
                "order by ANN_DT, S_INFO_WINDCODE " \
            .format(q_start, str(end))
        self.rdf.curs.execute(query)
        dpa = pd.DataFrame(self.rdf.curs.fetchall(),columns=['date', 'code', 'current_ratio'])
        print('raw data got!')
        dpa['date'] = pd.to_datetime(dpa['date'])
        mkt_data = self.influx.getDataMultiprocess('DailyData_Gus','marketData',q_start,end,['code','close'])
        mkt_data.index.names = ['date']
        mkt_data.reset_index(inplace=True)
        merge_df = pd.merge(mkt_data,dpa,how='outer',on=['date','code'])
        merge_df = merge_df.sort_values('date')
        merge_df['current_ratio'] = \
            merge_df.groupby('code')['current_ratio'].apply(lambda x: x.fillna(method='ffill'))
        merge_df = merge_df.groupby(['date','code']).last()
        merge_df = merge_df.reset_index().set_index('date')
        merge_df = merge_df.loc[str(start):str(end),['code','current_ratio']]
        merge_df = merge_df.dropna(subset=['current_ratio'])
        merge_df = merge_df.where(pd.notnull(merge_df),None)
        print('data processing finish')
        self.influx.saveData(merge_df, 'DailyFactor_Gus', 'FinancialQuality')



if __name__ == '__main__':
    i = DPA_p1()
    i.cal_factors(20100101,20190901)