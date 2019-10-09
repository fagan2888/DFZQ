import influxdb
from rdf_data import rdf_data
import dateutil.parser as dtparser
import datetime
import pandas as pd
from joblib import Parallel,delayed,parallel_backend
from dateutil.relativedelta import relativedelta

class influxdbData:
    def __init__(self, db_input=None):
        self.client = influxdb.DataFrameClient\
            (host='192.168.58.71', port=8086, username='root', password='root', database=db_input)
        # 服务器:192.168.58.71 阿毛:192.168.38.176

    def getDBs(self):
        self.dbs = self.client.get_list_database()
        return self.dbs

    def dropDB(self,database):
        self.client.drop_database(database)
        return

    def getTables(self):
        tables = self.client.get_list_measurements()
        return tables

    def getData(self,database,measure,startdate=None,enddate=None,fields=None):
        if fields:
            fields = ','.join(str(fld) for fld in fields)
        else:
            fields = '*'
        if startdate and (not enddate):
            begin = dtparser.parse(startdate)
            b = begin - datetime.timedelta(hours=8)
            q_postfix = f''' where time >= {int(b.timestamp() * 1000 * 1000 * 1000)}'''
        elif (not startdate) and enddate:
            end = dtparser.parse(enddate)
            e = end - datetime.timedelta(hours=8)
            q_postfix = f''' where time < {int(e.timestamp() * 1000 * 1000* 1000)}'''
        elif startdate and enddate:
            begin = dtparser.parse(startdate)
            b = begin - datetime.timedelta(hours=8)
            end = dtparser.parse(enddate)
            e = end - datetime.timedelta(hours=8)
            q_postfix = f''' where time >= {int(b.timestamp() * 1000 * 1000 * 1000)} and time < {int(e.timestamp() * 1000 * 1000* 1000)}'''

        q = f'''select {fields} from "{database}"."autogen"."{measure}"''' + q_postfix
        result = self.client.query(q)
        if not result:
            return
        data = pd.DataFrame(result[measure])
        return data


    def saveData(self,data,database,measure):
        dbs = self.getDBs()
        if not {'name':database} in dbs:
            self.client.create_database(database)
        success_flag = False
        for i in range(10):
            try:
                self.client.write_points(dataframe=data,database=database,measurement=measure,tag_columns=['code'],
                                         protocol='json',batch_size=1000)
                success_flag = True
                break
            except Exception as excp:
                error_msg = excp
        if not success_flag:
            print(error_msg)
        else:
            print('data saved!')



    # 需要对空的情况特殊处理
    def getDataMultiprocess(self,database,measure,startdate,enddate,fields=None):
        dt_start = datetime.datetime.strptime(str(startdate),'%Y%m%d')
        dt_end = datetime.datetime.strptime(str(enddate),'%Y%m%d')
        parameter_list = []

        if isinstance(measure,list):
            while dt_start <= dt_end:
                if dt_start + relativedelta(months=1) < dt_end:
                    period_end = dt_start+relativedelta(months=1)
                else:
                    period_end = dt_end
                for code in measure:
                    parameter_list.append((dt_start.strftime('%Y%m%d'),period_end.strftime('%Y%m%d'),code))
                dt_start += relativedelta(months=1)
        else:
            while dt_start <= dt_end:
                if dt_start + relativedelta(months=1) < dt_end:
                    period_end = dt_start + relativedelta(months=1)
                else:
                    period_end = dt_end
                parameter_list.append((dt_start.strftime('%Y%m%d'),period_end.strftime('%Y%m%d'),measure))
                dt_start += relativedelta(months=1)

        with parallel_backend('multiprocessing', n_jobs=-1):
            result_list =  Parallel()(delayed(self.getData)(database,cd,start_date,end_date,fields)
                                      for start_date,end_date,cd in parameter_list)
        return result_list


if __name__ == '__main__':
    influx = influxdbData()
    print(influx.getDBs())
    print(datetime.datetime.now())
    a = influx.getDataMultiprocess('DailyData_backtest','marketData','20100101','20190901',None)
    a = pd.concat(a)
    a = a.loc[pd.notnull(a['split_ratio']),:]
    print('finish!')
    print(influx.getDBs())
