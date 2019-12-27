# 资本结构因子 cur_dbt2dbt 的计算

from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
import datetime
import joblib


class CC_p1(FactorBase):
    def __init__(self):
        super().__init__()

    def cal_factors(self,start,end):
        q_start = (dtparser.parse(str(start))-relativedelta(years=1)).strftime('%Y%m%d')
        query = "select ANN_DT, S_INFO_WINDCODE, S_FA_CURRENTDEBTTODEBT " \
                "from wind_filesync.AShareFinancialIndicator " \
                "where ANN_DT >= {0} and ANN_DT <= {1} " \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '60%') " \
                "order by ANN_DT, S_INFO_WINDCODE " \
            .format(q_start, str(end))
        self.rdf.curs.execute(query)
        CC = pd.DataFrame(self.rdf.curs.fetchall(),columns=['date', 'code', 'cur_dbt2dbt'])
        print('raw data got!')
        CC['date'] = pd.to_datetime(CC['date'])
        mkt_data = self.influx.getDataMultiprocess('DailyData_Gus','marketData',q_start,end,['code','close'])
        mkt_data.index.names = ['date']
        mkt_data.reset_index(inplace=True)
        merge_df = pd.merge(mkt_data,CC,how='outer',on=['date','code'])
        merge_df = merge_df.sort_values('date')
        merge_df['cur_dbt2dbt'] = \
            merge_df.groupby('code')['cur_dbt2dbt'].apply(lambda x: x.fillna(method='ffill'))
        merge_df = merge_df.groupby(['date','code']).last()
        merge_df = merge_df.reset_index().set_index('date')
        merge_df = merge_df.loc[str(start):str(end),['code','cur_dbt2dbt']]
        merge_df = merge_df.dropna(subset=['cur_dbt2dbt'])
        merge_df = merge_df.where(pd.notnull(merge_df),None)
        print('data processing finish')
        self.influx.saveData(merge_df, 'DailyFactor_Gus', 'FinancialQuality')



if __name__ == '__main__':
    i = CC_p1()
    i.cal_factors(20100101,20191201)