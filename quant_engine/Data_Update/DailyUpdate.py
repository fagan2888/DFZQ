import sys
root_dir = 'D:\\github\\quant_engine'
sys.path.extend([root_dir, root_dir + '\\Data_Resource',
                 root_dir + '\\Data_Update\\marketData', root_dir + '\\Data_Update\\Indicators'])

from rdf_data import rdf_data
import logging
import datetime
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
from BacktestDayData import BacktestDayData
from AdjFactor import AdjFactor
from Industry_Lv1 import IndustryLv1
from StkSwap import UpdateSwapData
from SwapDataProcess import FillSwapData
from shares_and_turnover import shares_and_turnover

class DailyUpdate:
    def __init__(self):
        self.rdf = rdf_data()
        self.calendar = self.rdf.get_trading_calendar()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)
        self.log_file = root_dir + '\\Data_Update\\DailyUpdate.log'
        handler = logging.FileHandler(self.log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

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
            # 更新每日行情
            self.logger.info('Date:  %s' % (dt_today.strftime('%Y%m%d')))
            self.logger.info('*'*30)
            btd = BacktestDayData()
            res = btd.process_data(last_week, last_trade_day, n_jobs)
            for r in res:
                self.logger.info(r)
            self.logger.info('------------------market data finish------------------')
            adj = AdjFactor()
            res = adj.process_data(last_week, last_trade_day)
            self.logger.info(res)
            self.logger.info('------------------adj factor finish------------------')
            idsty = IndustryLv1()
            res = idsty.process_data(last_week, last_trade_day)
            self.logger.info(res)
            self.logger.info('------------------improved Lv1 finish------------------')
            usd = UpdateSwapData()
            res = usd.process_data(last_1yr, last_trade_day)
            self.logger.info(res)
            self.logger.info('------------------swap data finish------------------')
            fsd = FillSwapData()
            res = fsd.process_data()
            self.logger.info(res)
            self.logger.info('------------------fill swap data finish------------------')
            sh = shares_and_turnover()
            res = sh.process_data(last_week, last_trade_day, n_jobs)
            for r in res:
                self.logger.info(r)
            self.logger.info('------------------market data finish------------------')


if __name__ == '__main__':
    du = DailyUpdate()
    du.run(4)