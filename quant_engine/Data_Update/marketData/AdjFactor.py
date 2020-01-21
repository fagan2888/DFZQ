from rdf_data import rdf_data
from influxdb_data import influxdbData
import pandas as pd


class AdjFactor:
    def __init__(self):
        self.rdf = rdf_data()
        self.influx = influxdbData()

    def process_data(self, start, end):
        query = "select TRADE_DT, S_INFO_WINDCODE, S_DQ_ADJFACTOR " \
                "from wind_filesync.AShareEODPrices " \
                "where (S_INFO_WINDCODE like '0%' or S_INFO_WINDCODE like '3%' or S_INFO_WINDCODE like '6%') " \
                "and TRADE_DT >= {0} and TRADE_DT <= {1}".format(str(start), str(end))
        self.rdf.curs.execute(query)
        adj_factor = pd.DataFrame(self.rdf.curs.fetchall(), columns=['date', 'code', 'adj_factor'])
        adj_factor['date'] = pd.to_datetime(adj_factor['date'])
        adj_factor = adj_factor.groupby(['date', 'code']).last()
        adj_factor = adj_factor.reset_index().set_index('date')
        adj_factor = adj_factor.sort_index()
        self.influx.saveData(adj_factor, 'DailyData_Gus', 'marketData')
        print('Adj factor finish!')
        print('-'*30)


if __name__ == '__main__':
    adj = AdjFactor()
    adj.process_data(20190801, 20200201)