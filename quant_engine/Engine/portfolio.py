import math
import logging

class portfolio:
    def __init__(self,capital_input=1000000,slippage_input=0.002,transaction_fee_input=0.0001):
        self.capital = capital_input
        self.balance = capital_input
        self.slippage = slippage_input
        self.transaction_fee_ratio = transaction_fee_input
        self.tax_ratio = 0.001
        # exp: long_side = {stock_code:[open_position_price,volume]}
        self.long_side = {}
        self.short_side = {}

        logging.basicConfig(level=logging.INFO,filename='transaction.log',filemode='w',
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

    def buy_on_volume(self,stock_code,price,volume,transaction_time):
        stock_code = str(stock_code)
        self.multi = self.multi_dict[stock_code]
        self.transaction_cost_per_hand = self.transaction_cost_per_hand_dict[stock_code]
        amount = price * self.multi * volume / self.leverage
        transaction_cost = self.transaction_cost_per_hand * volume

        if stock_code in self.short_side:
            if volume > self.short_side[stock_code][1]:
                self.long_side[stock_code] = [price, volume - self.short_side[stock_code][1]]
                self.balance = self.balance \
                            + self.short_side[stock_code][0] * self.short_side[stock_code][1] * self.multi /self.leverage\
                            + (self.short_side[stock_code][0] - price) * self.short_side[stock_code][1] * self.multi \
                            - price * (volume - self.short_side[stock_code][1]) * self.multi / self.leverage\
                            - transaction_cost
                self.short_side.pop(stock_code)
            elif volume == self.short_side[stock_code][1]:
                self.balance = self.balance \
                            + self.short_side[stock_code][0] * self.short_side[stock_code][1] * self.multi /self.leverage \
                            + (self.short_side[stock_code][0] - price) * self.short_side[stock_code][1] * self.multi \
                            - transaction_cost
                self.short_side.pop(stock_code)
            else:
                self.short_side[stock_code][1] = self.short_side[stock_code][1] - volume
                self.balance = self.balance \
                            + self.short_side[stock_code][0] * volume * self.multi /self.leverage \
                            + (self.short_side[stock_code][0] - price) * volume * self.multi \
                            - transaction_cost
        else:
            if stock_code in self.long_side:
                weighted_price = (self.long_side[stock_code][0] * self.long_side[stock_code][1] + price * volume)\
                            / (self.long_side[stock_code][1] + volume)
                self.balance = self.balance - amount - transaction_cost
                positions = volume + self.long_side[stock_code][1]
                self.long_side[stock_code] = [weighted_price, positions]
            else:
                self.long_side[stock_code] = [price, volume]
                self.balance = self.balance - amount - transaction_cost

        self.logger.info("transaction_time: %s, Buy - %s @%f, volume: %i"
                         % (str(transaction_time), stock_code, price, volume))
        self.logger.info("portfolio balance: %f" % self.balance)


    def sell_on_volume(self,stock_code,price,volume,transaction_time):
        stock_code = str(stock_code)
        self.multi = self.multi_dict[stock_code]
        self.transaction_cost_per_hand = self.transaction_cost_per_hand_dict[stock_code]
        amount = price * self.multi * volume / self.leverage
        transaction_cost = self.transaction_cost_per_hand * volume

        if stock_code in self.long_side:
            if volume > self.long_side[stock_code][1]:
                self.short_side[stock_code] = [price, volume - self.long_side[stock_code][1]]
                self.balance = self.balance \
                            + self.long_side[stock_code][0] * self.long_side[stock_code][1] * self.multi /self.leverage\
                            + (price - self.long_side[stock_code][0]) * self.long_side[stock_code][1] * self.multi \
                            - price * (volume - self.long_side[stock_code][1]) * self.multi / self.leverage \
                            - transaction_cost
                self.long_side.pop(stock_code)
            elif volume == self.long_side[stock_code][1]:
                self.balance = self.balance \
                            + self.long_side[stock_code][0] * self.long_side[stock_code][1] * self.multi /self.leverage\
                            + (price - self.long_side[stock_code][0]) * self.long_side[stock_code][1] * self.multi \
                            - transaction_cost
                self.long_side.pop(stock_code)
            else:
                self.long_side[stock_code][1] = self.long_side[stock_code][1] - volume
                self.balance = self.balance \
                            + self.long_side[stock_code][0] * volume * self.multi /self.leverage \
                            + (price - self.long_side[stock_code][0]) * volume * self.multi \
                            - transaction_cost
        else:
            if stock_code in self.short_side:
                weighted_price = (self.short_side[stock_code][0] * self.short_side[stock_code][1] + price * volume) /\
                                 (self.short_side[stock_code][1] + volume)
                self.balance = self.balance - amount - transaction_cost
                positions = volume + self.short_side[stock_code][1]
                self.short_side[stock_code] = [weighted_price, positions]
            else:
                self.short_side[stock_code] = [price, volume]
                self.balance = self.balance - amount - transaction_cost

        self.logger.info("transaction_time: %s, Sell - %s @%f, volume: %i"
                         % (str(transaction_time), stock_code, price, volume))
        self.logger.info("portfolio balance: %f" % self.balance)

    def get_portfolio_value(self,price_dict):
        # exp: price_dict={stock_code:price}
        long_side_value = 0
        short_side_value = 0
        for stk in self.long_side:
            self.multi = self.multi_dict[stk]
            long_side_value += self.long_side[stk][0] * self.long_side[stk][1] * self.multi / self.leverage \
                            + (price_dict[stk] - self.long_side[stk][0]) * self.long_side[stk][1] * self.multi
        for stk in self.short_side:
            self.multi = self.multi_dict[stk]
            short_side_value += self.short_side[stk][0] * self.short_side[stk][1] * self.multi / self.leverage \
                            + (self.short_side[stk][0] - price_dict[stk]) * self.short_side[stk][1] * self.multi
        portfolio_value = self.balance + long_side_value + short_side_value
        return portfolio_value
