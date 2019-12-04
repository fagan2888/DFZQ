import math
import logging
import datetime
import futures_constant
import pandas as pd

class stock_portfolio:
    def __init__(self,capital_input=1000000,slippage_input=0.001,transaction_fee_input=0.0001):
        self.capital = capital_input
        self.balance = capital_input
        self.slippage = slippage_input
        self.transaction_fee_ratio = transaction_fee_input
        self.tax_ratio = 0.001
        # 股票仓位模板{'600000.SH':{'price':2.22,'volume':300}}
        self.stk_positions = {}

        self.transactions_list = []
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)
        dir = 'C:/Users/trader_9/PycharmProjects/DFZQ/quant_engine/Transaction_Log/'
        file_name = 'StkTransactions_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M") +'.log'
        handler = logging.FileHandler(dir+file_name)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def buy_stks_by_volume(self,time,stock_code,price,volume):
        actual_price = price*(1+self.slippage)
        amount = round(actual_price * volume,2)
        transaction_fee = round(self.transaction_fee_ratio * amount,2)

        if stock_code in self.stk_positions:
            weighted_volume = self.stk_positions[stock_code]['volume'] + volume
            weighted_price = (self.stk_positions[stock_code]['price'] * self.stk_positions[stock_code]['volume'] \
                             + amount) / weighted_volume
            self.stk_positions[stock_code]['volume'] = weighted_volume
            self.stk_positions[stock_code]['price']  = weighted_price
        else:
            self.stk_positions[stock_code] = {'volume':volume,'price':actual_price}
        self.balance = self.balance - amount - transaction_fee
        self.logger.info("TransactionTime: %s, Type: BuyStock, Detail: %s - %f * %i, Balance: %f"
                         % (time, stock_code, price, volume, self.balance))
        self.transactions_list.append(pd.Series([time,'BUY',stock_code,price,actual_price,volume,self.balance,transaction_fee],
                                                index=['Time','Type','Code','RawPrice','ActualPrice','Volume','Balance','Fee']))


    def buy_stks_by_amount(self,time,stock_code,price,goal_amount):
        goal_volume = round(goal_amount/price,-2)
        if goal_volume <= 0:
            self.logger.warning("Error: goal volume <= 0, won't execute transaction!")
        else:
            self.logger.info('Estimate Trade Volume: %i' % goal_volume)
            self.buy_stks_by_volume(time,stock_code,price,goal_volume)


    def sell_stks_by_volume(self,time,stock_code,price,volume):
        if stock_code in self.stk_positions:
            if volume > self.stk_positions[stock_code]['volume']:
                self.logger.warning("Error: volume to sell > volume in positions, adjust to volume in positions!")
                volume = self.stk_positions[stock_code]['volume']
            else:
                pass
            actual_price = price * (1 - self.slippage)
            amount = round(actual_price * volume,2)
            transaction_fee = round(self.transaction_fee_ratio * amount,2)
            tax = round(self.tax_ratio * amount,2)
            weighted_volume = self.stk_positions[stock_code]['volume'] - volume
            if weighted_volume == 0:
                self.stk_positions.pop(stock_code)
            else:
                self.stk_positions[stock_code]['volume'] = weighted_volume
            self.balance = self.balance + amount - transaction_fee - tax
            self.logger.info("TransactionTime: %s, Type: SellStock, Detail: %s - %f * %i, Balance: %f"
                             % (time, stock_code, price, volume, self.balance))
            self.transactions_list.append(pd.Series([time, 'SELL', stock_code, price, actual_price, volume, self.balance, transaction_fee],
                                                    index=['Time', 'Type', 'Code', 'RawPrice', 'ActualPrice', 'Volume', 'Balance', 'Fee']))
        else:
            self.logger.warning("Error: this stk not in portfolio!")


    def sell_stks_by_amount(self,time,stock_code,price,goal_amount):
        goal_volume = round(goal_amount/price,-2)
        if goal_volume <= 0:
            self.logger.warning("Error: goal volume <= 0, won't execute transaction!")
        else:
            self.logger.info('Estimate Trade Volume: %i' % goal_volume)
            self.sell_stks_by_volume(time,stock_code,price,goal_volume)


    def trade_stks_to_target_volume(self,time,stock_code,price,target_volume):
        if stock_code not in self.stk_positions:
            volume_held = 0
        else:
            volume_held = self.stk_positions[stock_code]['volume']
        volume_to_trade = target_volume - volume_held
        if volume_to_trade > 0:
            volume_to_trade = round(volume_to_trade,-2)
            self.buy_stks_by_volume(time,stock_code,price,volume_to_trade)
        elif volume_to_trade < 0 and volume_to_trade*(-1) < volume_held :
            volume_to_trade = round(volume_to_trade,-2) * -1
            self.volume = self.sell_stks_by_volume(time, stock_code, price, volume_to_trade)
        elif volume_to_trade < 0 and volume_to_trade*(-1) >= volume_held :
            self.sell_stks_by_volume(time,stock_code,price,volume_held)
        else:
            pass


    def process_ex_right(self,ex_right:pd.DataFrame):
        self.logger.info('****************************************')
        self.logger.info('Ex Right Info:')
        shr_ex_right = ex_right.loc[(ex_right['cash_dvd_ratio']!=0) | (ex_right['bonus_share_ratio']!=0) |
                                    (ex_right['conversed_ratio']!=0) |(ex_right['rightissue_price']!=0) |
                                    (ex_right['rightissue_ratio']!=0),
                                    ['cash_dvd_ratio','bonus_share_ratio','conversed_ratio',
                                     'rightissue_price','rightissue_ratio']]
        # 默认参加配股
        for code,row in shr_ex_right.iterrows():
            if code in self.stk_positions:
                self.logger.info('------------------------------------')
                self.logger.info('Code: %s, Cash DVD: %f, Bonus Share: %f, Conversed Ratio: %f, '
                                 'RI Price: %f, RI Ratio: %f'
                                 %(code, row['cash_dvd_ratio'], row['bonus_share_ratio'], row['conversed_ratio'],
                                   row['rightissue_price'], row['rightissue_ratio']))
                self.logger.info('Price Before: %f, Volume Before: %f, Latest Close Before: %f'
                                 %(self.stk_positions[code]['price'],self.stk_positions[code]['volume'],
                                   self.stk_positions[code]['latest_close']))
                self.logger.info('Balance Before: %f' %self.balance)
                self.stk_positions[code]['price'] = \
                    (self.stk_positions[code]['price'] - row['cash_dvd_ratio']
                     + row['rightissue_price'] * row['rightissue_ratio']) / \
                    (1 + row['bonus_share_ratio'] + row['conversed_ratio'] + row['rightissue_ratio'])
                # 复权后的收盘价需要保留两位小数
                self.stk_positions[code]['latest_close'] = \
                    round((self.stk_positions[code]['latest_close'] - row['cash_dvd_ratio']
                           + row['rightissue_price'] * row['rightissue_ratio']) /
                          (1 + row['bonus_share_ratio'] + row['conversed_ratio'] + row['rightissue_ratio']),2)
                self.balance = self.balance + self.stk_positions[code]['volume'] * row['cash_dvd_ratio'] - \
                               row['rightissue_price'] * self.stk_positions[code]['volume'] * row['rightissue_ratio']
                self.stk_positions[code]['volume'] = round(self.stk_positions[code]['volume'] * \
                                (1 + row['bonus_share_ratio'] + row['conversed_ratio'] + row['rightissue_ratio']))
                self.logger.info('Price After: %f, Volume After: %f, Latest Close After: %f'
                                 % (self.stk_positions[code]['price'], self.stk_positions[code]['volume'],
                                    self.stk_positions[code]['latest_close']))
                self.logger.info('Balance After: %f' %self.balance)
            else:
                pass
        self.logger.info('****************************************')


    def get_portfolio_value(self,price_input:pd.Series):
        self.logger.info('****************************************')
        self.logger.info('Balance: %f' %self.balance)

        stk_value = 0
        for stk in self.stk_positions:
            # lastest_close 最新的close, 以防数据缺失
            if stk in price_input:
                self.stk_positions[stk]['latest_close'] = price_input[stk]
            else:
                pass
            stk_value += self.stk_positions[stk]['latest_close'] * self.stk_positions[stk]['volume']
        self.logger.info('Stock Value: %f' %stk_value)
        total_value = self.balance + stk_value
        self.logger.info('Total Value: %f' %total_value)
        self.logger.info('***************************************')
        return total_value


class futures_portfolio:
    def __init__(self,capital_input=1000000):
        self.account_right = capital_input
        self.slippage_multi = 1
        # 期货仓位模板{'IH1901.CFE':{'price':2536,'volume':-25}}
        self.ftrs_positions = {}

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)
        dir = 'C:/Users/trader_9/PycharmProjects/DFZQ/quant_engine/Log/'
        file_name = 'ftrs_log_' + datetime.datetime.now().strftime("%Y%m%d") + '.log'
        handler = logging.FileHandler(dir + file_name)
        handler.setLevel(logging.INFO)
        console = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.addHandler(console)

    def buy_ftrs_by_volume(self,time,symbol,price,volume):
        slippage = self.slippage_multi * futures_constant.FuturesTools.get_ftrs_tick(symbol)
        multi = futures_constant.FuturesTools.get_ftrs_multi(symbol)
        fee_rate = futures_constant.FuturesTools.get_ftrs_fee(symbol)
        margin_rate = futures_constant.FuturesTools.get_ftrs_margin(symbol)

        actual_price = price + slippage
        fee = round(actual_price * multi * volume * fee_rate, 2)

        if symbol not in self.ftrs_positions:
            self.ftrs_positions[symbol] = {'price':actual_price,'volume':volume,'margin':margin_rate}
            profit = 0
        else:
            weighted_volume = self.ftrs_positions[symbol]['volume'] + volume
            if (self.ftrs_positions[symbol]['volume'] < 0) and (abs(self.ftrs_positions[symbol]['volume']) >= volume):
                profit = round((self.ftrs_positions[symbol]['price'] - actual_price) * multi * volume,2)
                weighted_price = self.ftrs_positions[symbol]['price']
            elif (self.ftrs_positions[symbol]['volume'] < 0) and (abs(self.ftrs_positions[symbol]['volume']) < volume):
                profit = round((self.ftrs_positions[symbol]['price'] - actual_price) * multi * abs(self.ftrs_positions[symbol]['volume']),2)
                weighted_price = actual_price
            else:
                profit = 0
                weighted_price = round((self.ftrs_positions[symbol]['price'] * self.ftrs_positions[symbol]['volume'] \
                                 + actual_price * volume) / weighted_volume,3)

            if weighted_volume == 0:
                self.ftrs_positions.pop(symbol)
            else:
                self.ftrs_positions[symbol]['price'] = weighted_price
                self.ftrs_positions[symbol]['volume'] = weighted_volume

        self.account_right = self.account_right + profit - fee
        self.logger.info("TransactionTime: %s, Type: BuyFutures, Detail: %s - %f * %i"
                         % (time.strftime('%Y%m%d-%H:%M:%S'), symbol, price, volume))

    def sell_ftrs_by_volume(self,time,symbol,price,volume):
        slippage = self.slippage_multi * futures_constant.FuturesTools.get_ftrs_tick(symbol)
        multi = futures_constant.FuturesTools.get_ftrs_multi(symbol)
        fee_rate = futures_constant.FuturesTools.get_ftrs_fee(symbol)
        margin_rate = futures_constant.FuturesTools.get_ftrs_margin(symbol)

        actual_price = price - slippage
        fee = round(actual_price * multi * volume * fee_rate, 2)

        if symbol not in self.ftrs_positions:
            self.ftrs_positions[symbol] = {'price':actual_price,'volume':volume*(-1),'margin':margin_rate}
            profit = 0
        else:
            weighted_volume = self.ftrs_positions[symbol]['volume'] - volume
            fee = round(actual_price * multi * volume * fee_rate,2)
            if (self.ftrs_positions[symbol]['volume'] >= volume):
                profit = round((actual_price - self.ftrs_positions[symbol]['price']) * multi * volume,2)
                weighted_price = self.ftrs_positions[symbol]['price']
            elif (self.ftrs_positions[symbol]['volume'] > 0) and (self.ftrs_positions[symbol]['volume'] < volume):
                profit = round((actual_price - self.ftrs_positions[symbol]['price']) * multi * self.ftrs_positions[symbol]['volume'],2)
                weighted_price = actual_price
            else:
                profit = 0
                weighted_price = (self.ftrs_positions[symbol]['price'] * abs(self.ftrs_positions[symbol]['volume']) \
                                 + actual_price * volume) / abs(weighted_volume)

            if weighted_volume == 0:
                self.ftrs_positions.pop(symbol)
            else:
                self.ftrs_positions[symbol]['price'] = weighted_price
                self.ftrs_positions[symbol]['volume'] = weighted_volume

        self.account_right = self.account_right + profit - fee
        self.logger.info("TransactionTime: %s, Type: SellFutures, Detail: %s - %f * %i"
                         % (time.strftime('%Y%m%d-%H:%M:%S'), symbol, price, volume))

    def get_portfolio_margin(self,price_input:pd.Series):
        margin = 0
        for symbol in self.ftrs_positions:
            multi = futures_constant.FuturesTools.get_ftrs_multi(symbol)
            margin += price_input[symbol] * multi * self.ftrs_positions[symbol]['volume'] * self.ftrs_positions[symbol]['margin']
        return margin



if __name__ == '__main__':
     futures_portfolio = futures_portfolio()
     dt1 = datetime.datetime(1990,9,4)
     dt2 = datetime.datetime(1990,11,26)
     futures_portfolio.buy_ftrs_by_volume(dt1,'IH01.CFE',3000,2)
     futures_portfolio.buy_ftrs_by_volume(dt1,'IH01.CFE',2000,3)
     futures_portfolio.sell_ftrs_by_volume(dt2,'IH01.CFE',1000,1)
     print('ok')