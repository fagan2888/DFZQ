import pandas as pd
from portfolio import stock_portfolio,futures_portfolio
from influxdb_data import influxdbData
import dateutil.parser as dtparser

class BacktestEngine:
    def __init__(self,stock_capital=1000000,futures_capital=None):
        self.stk_portfolio = stock_portfolio(stock_capital)
        if not futures_capital:
            pass
        else:
            self.ftrs_portfolio = futures_portfolio(futures_capital)

    def run(self,stk_weight,start,end,cash_reserve_rate=0.05):
        influx = influxdbData()
        DB = 'DailyData_backtest'
        measure = 'marketData'
        daily_data = pd.concat(influx.getDataMultiprocess(DB,measure,str(start),str(end)))
        # 日线数据中的preclose已是相对前一天的复权价格
        daily_data = daily_data.loc[daily_data['code'].isin(stk_weight['code']),
                                    ['code','open','high','low','close','preclose','status','bonus_share_ratio',
                                     'cash_dvd_ratio','conversed_ratio','rightissue_price','rightissue_ratio']]
        daily_data[['bonus_share_ratio', 'cash_dvd_ratio', 'conversed_ratio', 'rightissue_price', 'rightissue_ratio']] = \
            daily_data[['bonus_share_ratio', 'cash_dvd_ratio', 'conversed_ratio', 'rightissue_price', 'rightissue_ratio']].fillna(0)
        daily_data.set_index([daily_data.index,'code'],inplace=True)
        stk_weight.set_index([stk_weight.index,'code'],inplace=True)
        daily_data.sort_index(inplace=True)
        i1 = stk_weight.index[0]
        q = daily_data.loc[i1,:]

        trade_days = daily_data.index.unique()
        for trade_day in trade_days:
            one_day_data = daily_data.loc[trade_day,:]
            print(',')


if __name__ == '__main__':
    d = pd.DataFrame({'code':['600000.SH','600016.SH','600000.SH','600016.SH','600000.SH','600016.SH'],
                      'weight':[0.3,0.7,0.4,0.6,0.5,0.5]},index=['20190108','20190108','20190109','20190109','20190110','20190110'])
    d.index = pd.to_datetime(d.index)
    QE = BacktestEngine()
    QE.run(d,20190101,20190501)
    print('.')