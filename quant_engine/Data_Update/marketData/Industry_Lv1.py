# 把保险和券商从非银中单独拆分出来
from influxdb_data import influxdbData
import pandas as pd

class IndustryLv1:
    def __init__(self):
        self.influx = influxdbData()

    def process_data(self,start,end):
        mkt_data = self.influx.getDataMultiprocess('DailyData_Gus', 'marketData', start, end,
                                                   ['code','citics_lv1_name','citics_lv2_name'])
        mkt_data['improved_lv1'] = mkt_data.apply(lambda row: row['citics_lv2_name']
                                            if row['citics_lv2_name'] in ['保险Ⅱ(中信)','证券Ⅱ(中信)']
                                            else row['citics_lv1_name'], axis=1)
        mkt_data = mkt_data.loc[:,['code','improved_lv1']].copy()
        mkt_data = mkt_data.where(pd.notnull(mkt_data), None)
        self.influx.saveData(mkt_data,'DailyData_Gus', 'marketData')


if __name__ == '__main__':
    ind = IndustryLv1()
    ind.process_data(20100101,20190901)