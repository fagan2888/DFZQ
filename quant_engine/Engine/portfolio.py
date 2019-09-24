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
        # 期货仓位模板{'IH1901.CFE':{'price':2536,'volume':-25}}
        self.stk_positions = {}
        self.ftr_positions = {}

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)
        dir = 'D:/github/quant_engine/Log/'
        file_name = 'stk_log_'+ datetime.datetime.now().strftime("%Y%m%d") +'.log'
        handler = logging.FileHandler(dir+file_name)
        handler.setLevel(logging.INFO)
        console = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.addHandler(console)

    def buy_stks_by_volume(self,time,stock_code,price,volume):
        actual_price = round(price*(1+self.slippage),3)
        amount = actual_price * volume
        transaction_fee = round(self.transaction_fee_ratio * amount,2)

        if stock_code in self.stk_positions:
            weighted_volume = self.stk_positions[stock_code]['volume'] + volume
            weighted_price = round((self.stk_positions[stock_code]['price'] * self.stk_positions[stock_code]['volume'] \
                             + amount) / weighted_volume,3)
            self.stk_positions[stock_code]['volume'] = weighted_volume
            self.stk_positions[stock_code]['price']  = weighted_price
        else:
            self.stk_positions[stock_code] = {'volume':volume,'price':actual_price}
        self.balance = self.balance - amount - transaction_fee

        self.logger.info("TransactionTime: %s, Type: BuyStock, Detail: %s - %f * %i, Balance: %f"
                         % (time.strftime('%Y%m%d-%H:%M:%S'), stock_code, price, volume, self.balance))


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
            actual_price = round(price * (1 - self.slippage), 3)
            amount = actual_price * volume
            transaction_fee = round(self.transaction_fee_ratio * amount,2)
            tax = round(self.tax_ratio * amount,2)
            weighted_volume = self.stk_positions[stock_code]['volume'] - volume
            if weighted_volume == 0:
                self.stk_positions.pop(stock_code)
            else:
                self.stk_positions[stock_code]['volume'] = weighted_volume
            self.balance = self.balance + amount - transaction_fee - tax

            self.logger.info("TransactionTime: %s, Type: SellStock, Detail: %s - %f * %i, Balance: %f"
                             % (time.strftime('%Y%m%d-%H:%M:%S'), stock_code, price, volume, self.balance))
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
        volume_held = self.stk_positions[stock_code]['volume']
        volume_diff = target_volume - volume_held
        if volume_diff > 0:
            self.buy_stks_by_volume(time,stock_code,price,volume_diff)
        elif volume_diff < 0:
            self.sell_stks_by_volume(time,stock_code,price,volume_diff*(-1))


    def get_portfolio_value(self,price_input:pd.Series):
        total_value = self.balance
        for stk in self.stk_positions:
            total_value += price_input[stk] * self.stk_positions[stk]['volume']
        return total_value


class futures_portfolio:
    def __init__(self,capital_input=1000000):
        self.account_right = capital_input
        self.slippage_multi = 1
        self.ftrs_positions = {}

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)
        dir = 'D:/github/quant_engine/Log/'
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
            margin += price_input[symbol] * multi * self.ftrs_positions[stk]['volume'] * self.ftrs_positions[stk]['margin']
        return margin



if __name__ == '__main__':
     futures_portfolio = futures_portfolio()
     dt1 = datetime.datetime(1990,9,4)
     dt2 = datetime.datetime(1990,11,26)
     futures_portfolio.buy_ftrs_by_volume(dt1,'IH01.CFE',3000,2)
     futures_portfolio.buy_ftrs_by_volume(dt1,'IH01.CFE',2000,3)
     futures_portfolio.sell_ftrs_by_volume(dt2,'IH01.CFE',1000,1)
     print('ok')