# 收益质量因子 OperateIncome2EBT_Q, profit_ddt2profit_Q 的计算

from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
import datetime
import joblib


class EQ_p1(FactorBase):
    def __init__(self):
        super().__init__()

    def cal_factors(self,start,end):
        q_start = (dtparser.parse(str(start))-relativedelta(years=1)).strftime('%Y%m%d')
        query = "select ANN_DT, S_INFO_WINDCODE, S_QFA_OPERATEINCOMETOEBT, S_QFA_DEDUCTEDPROFITTOPROFIT " \
                "from wind_filesync.AShareFinancialIndicator " \
                "where ANN_DT >= {0} and ANN_DT <= {1} " \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "order by ANN_DT, S_INFO_WINDCODE " \
            .format(q_start, str(end))
        self.rdf.curs.execute(query)
        EQ = pd.DataFrame(self.rdf.curs.fetchall(),
                          columns=['date', 'code', 'OperateIncome2EBT_Q', 'profit_ddt2profit_Q'])
        print('raw data got!')
        EQ['date'] = pd.to_datetime(EQ['date'])
        mkt_data = self.influx.getDataMultiprocess('DailyData_Gus','marketData',q_start,end,['code','close'])
        mkt_data.index.names = ['date']
        mkt_data.reset_index(inplace=True)
        merge_df = pd.merge(mkt_data,EQ,how='outer',on=['date','code'])
        merge_df = merge_df.sort_values('date')
        merge_df['OperateIncome2EBT_Q'] = \
            merge_df.groupby('code')['OperateIncome2EBT_Q'].apply(lambda x: x.fillna(method='ffill'))
        merge_df['profit_ddt2profit_Q'] = \
            merge_df.groupby('code')['profit_ddt2profit_Q'].apply(lambda x: x.fillna(method='ffill'))
        merge_df = merge_df.groupby(['date','code']).last()
        merge_df = merge_df.reset_index().set_index('date')
        merge_df = merge_df.loc[str(start):str(end),['code','OperateIncome2EBT_Q','profit_ddt2profit_Q']]
        merge_df = merge_df.loc[pd.notnull(merge_df['OperateIncome2EBT_Q']) |
                                pd.notnull(merge_df['profit_ddt2profit_Q']),:]
        merge_df = merge_df.where(pd.notnull(merge_df),None)
        print('data processing finish')
        self.influx.saveData(merge_df, 'DailyFactor_Gus', 'FinancialQuality')



if __name__ == '__main__':
    eq = EQ_p1()
    eq.cal_factors(20100101,20191201)