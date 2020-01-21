import sys
sys.path.extend(['D:\github\quant_engine', 'D:\github\quant_engine\Data_Update\marketData',
                 'D:\github\quant_engine\Data_Resource'])

from influxdb_data import influxdbData
from rdf_data import rdf_data
import datetime
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
from BacktestDayData import BacktestDayData
from AdjFactor import AdjFactor
from Industry_Lv1 import IndustryLv1
from StkSwap import UpdateSwapData
from SwapDataProcess import FillSwapData

class DailyUpdate:
    def __init__(self):
        self.rdf = rdf_data()
        self.calendar = self.rdf.get_trading_calendar()

    def run(self, n_jobs):
        dt_today = dtparser.parse(datetime.datetime.now().strftime('%Y%m%d'))
        if self.calendar[self.calendar == dt_today].empty:
            print('Not Trade Day...')
        else:
            dt_last_trade_day = self.calendar[self.calendar < dt_today].iloc[-1]
            dt_last_week = dt_last_trade_day - relativedelta(weeks=1)
            dt_last_1yr = dt_last_trade_day - relativedelta(years=1)
            dt_last_2yr = dt_last_trade_day - relativedelta(years=2)
            last_trade_day = dt_last_trade_day.strftime('%Y%m%d')
            last_week = dt_last_week.strftime('%Y%m%d')
            last_1yr = dt_last_1yr.strftime('%Y%m%d')
            last_2yr = dt_last_2yr.strftime('%Y%m%d')
            # ---------------------------------------------
            btd = BacktestDayData()
            btd.process_data(last_week, last_trade_day, n_jobs)
            adj = AdjFactor()
            adj.process_data(last_week, last_trade_day)
            idsty = IndustryLv1()
            idsty.process_data(last_week, last_trade_day)
            usd = UpdateSwapData()
            usd.process_data(last_1yr, last_trade_day)
            fsd = FillSwapData()
            fsd.process_data()


if __name__ == '__main__':
    du = DailyUpdate()
    du.run(4)