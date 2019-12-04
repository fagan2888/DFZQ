# 开始日期： 2019-11-7

from strategy_base import StrategyBase
import pandas as pd
from data_process import DataProcess
from influxdb_data import influxdbData

class strategy_exercise:
    def __init__(self):
        self.influx = influxdbData()

    def run(self):
        a = self.influx.getDataMultiprocess('DailyData_Gus','marketData',20180101,20190501,None)
        return a







if __name__ == '__main__':
    exe = strategy_exercise()
    a = exe.run()
    print('.')