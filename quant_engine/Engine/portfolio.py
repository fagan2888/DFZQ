import logging
import datetime
import pandas as pd
import numpy as np
import global_constant
import os.path


class stock_portfolio:
    def __init__(self, save_name, capital_input=1000000, slippage_input=0.001, transaction_fee_input=0.0001):
        self.capital = capital_input
        self.balance = capital_input
        self.slippage = slippage_input
        self.transaction_fee_ratio = transaction_fee_input
        self.tax_ratio = 0.001
        # 股票仓位模板{'600000.SH':{'price':2.22,'volume':300}}
        self.stk_positions = {}

        self.transactions = np.array([])
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)
        dir = global_constant.ROOT_DIR + 'Transaction_Log/{0}/'.format(save_name)
        if os.path.exists(dir):
            pass
        else:
            os.makedirs(dir.rstrip('/'))
        file_name = 'StkTransactions_{0}.log'.format(save_name)
        handler = logging.FileHandler(dir + file_name)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    # 函数可以接受非100倍数
    def buy_stks_by_volume(self, time, stock_code, price, volume):
        actual_price = price * (1 + self.slippage)
        amount = round(actual_price * volume, 2)
        transaction_fee = round(self.transaction_fee_ratio * amount, 2)
        if stock_code in self.stk_positions:
            weighted_volume = self.stk_positions[stock_code]['volume'] + volume
            weighted_price = \
                (self.stk_positions[stock_code]['price'] * self.stk_positions[stock_code]['volume'] + amount) \
                / weighted_volume
            self.stk_positions[stock_code]['volume'] = weighted_volume
            self.stk_positions[stock_code]['price'] = weighted_price
        else:
            self.stk_positions[stock_code] = {'volume': volume, 'price': actual_price}
        self.balance = self.balance - amount - transaction_fee
        trade = np.array([time, 'BUY', stock_code, price, actual_price, volume, self.balance, transaction_fee])
        self.transactions = np.append(self.transactions, trade)

    def buy_stks_by_amount(self, time, stock_code, price, goal_amount):
        goal_volume = round(goal_amount / price, -2)
        if goal_volume <= 0:
            self.logger.warning("Error: goal volume <= 0, won't execute transaction!")
        else:
            self.logger.info('Estimate Trade Volume: %i' % goal_volume)
            self.buy_stks_by_volume(time, stock_code, price, goal_volume)

    # 函数可以接受非100倍数
    def sell_stks_by_volume(self, time, stock_code, price, volume):
        # 零股可卖不可买
        if stock_code in self.stk_positions:
            actual_price = price * (1 - self.slippage)
            # volume 调整
            if volume > self.stk_positions[stock_code]['volume']:
                self.logger.warning("Code: %s Error: volume to sell > volume in positions, "
                                    "adjust to volume in positions!" % stock_code)
                volume = self.stk_positions[stock_code]['volume']
            amount = round(actual_price * volume, 2)
            transaction_fee = round(self.transaction_fee_ratio * amount, 2)
            tax = round(self.tax_ratio * amount, 2)
            weighted_volume = self.stk_positions[stock_code]['volume'] - volume
            if weighted_volume == 0:
                self.stk_positions.pop(stock_code)
            else:
                self.stk_positions[stock_code]['volume'] = weighted_volume
            self.balance = self.balance + amount - transaction_fee - tax
            trade = np.array([time, 'SELL', stock_code, price, actual_price, volume, self.balance, transaction_fee])
            self.transactions = np.append(self.transactions, trade)
        else:
            print("Code: %s Error: this stk not in portfolio!" % stock_code)
            raise NameError

    def sell_stks_by_amount(self, time, stock_code, price, goal_amount):
        goal_volume = round(goal_amount / price, -2)
        if goal_volume <= 0:
            self.logger.warning("Code: %s Error: goal volume <= 0, won't execute transaction!" % stock_code)
            raise NameError
        else:
            self.logger.info('Estimate Trade Volume: %i' % goal_volume)
            self.sell_stks_by_volume(time, stock_code, price, goal_volume)

    def trade_stks_to_target_volume(self, time, stock_code, price, target_volume, price_limit='no_limit'):
        if stock_code in self.stk_positions:
            volume_held = self.stk_positions[stock_code]['volume']
        else:
            volume_held = 0
        vol_diff = target_volume - volume_held
        round_vol = round(vol_diff, -2)
        # 考虑了涨跌停一字板的情况
        # 仅在 买入 >=100股 | 卖出 <=100股 且目标股数不为 0 | 目标股数为0 时发生交易
        # 卖出情况
        if vol_diff < 0:
            if price_limit == 'low':
                return 'Trade Fail'
            else:
                if target_volume == 0:
                    self.sell_stks_by_volume(time, stock_code, price, volume_held)
                    return price * volume_held
                elif round_vol < 0:
                    self.sell_stks_by_volume(time, stock_code, price, round_vol * -1)
                    return price * round_vol * -1
                else:
                    return 0
        # 买入情况
        elif vol_diff > 0:
            if price_limit == 'high':
                return 'Trade Fail'
            else:
                if round_vol > 0:
                    self.buy_stks_by_volume(time, stock_code, price, round_vol)
                    return price * round_vol
                else:
                    return 0
        # 不交易情况
        else:
            return 0

    def process_ex_right(self, ex_right: pd.DataFrame):
        # date 为 index
        self.logger.info('****************************************')
        self.logger.info('Ex Right Info:')
        shr_ex_right = ex_right.loc[ex_right['code'].isin(self.stk_positions.keys()) & (
                (ex_right['cash_dvd_ratio'] > 0) | (ex_right['bonus_share_ratio'] > 0) |
                (ex_right['conversed_ratio'] > 0) | (ex_right['rightissue_price'] > 0) |
                (ex_right['rightissue_ratio'] > 0)), :]
        # 默认参加配股
        for date, row in shr_ex_right.iterrows():
            self.logger.info('------------------------------------')
            self.logger.info('Code: %s, Cash DVD: %f, Bonus Share: %f, Conversed Ratio: %f, '
                             'RI Price: %f, RI Ratio: %f'
                             % (row['code'], row['cash_dvd_ratio'], row['bonus_share_ratio'], row['conversed_ratio'],
                                row['rightissue_price'], row['rightissue_ratio']))
            self.logger.info('Price Before: %f, Volume Before: %f, Latest Close Before: %f'
                             % (self.stk_positions[row['code']]['price'], self.stk_positions[row['code']]['volume'],
                                self.stk_positions[row['code']]['latest_close']))
            self.logger.info('Balance Before: %f' % self.balance)
            self.stk_positions[row['code']]['price'] = \
                (self.stk_positions[row['code']]['price'] - row['cash_dvd_ratio']
                 + row['rightissue_price'] * row['rightissue_ratio']) / \
                (1 + row['bonus_share_ratio'] + row['conversed_ratio'] + row['rightissue_ratio'])
            # 复权后的收盘价需要保留两位小数
            self.stk_positions[row['code']]['latest_close'] = \
                round((self.stk_positions[row['code']]['latest_close'] - row['cash_dvd_ratio']
                       + row['rightissue_price'] * row['rightissue_ratio']) /
                      (1 + row['bonus_share_ratio'] + row['conversed_ratio'] + row['rightissue_ratio']), 2)
            self.balance = \
                self.balance + self.stk_positions[row['code']]['volume'] * row['cash_dvd_ratio'] - \
                row['rightissue_price'] * self.stk_positions[row['code']]['volume'] * row['rightissue_ratio']
            self.stk_positions[row['code']]['volume'] = round(
                self.stk_positions[row['code']]['volume'] *
                (1 + row['bonus_share_ratio'] + row['conversed_ratio'] + row['rightissue_ratio']))
            self.logger.info('Price After: %f, Volume After: %f, Latest Close After: %f'
                             % (self.stk_positions[row['code']]['price'], self.stk_positions[row['code']]['volume'],
                                self.stk_positions[row['code']]['latest_close']))
            self.logger.info('Balance After: %f' % self.balance)
        self.logger.info('****************************************')

    def get_portfolio_value(self, price_input: pd.Series):
        self.logger.info('****************************************')
        self.logger.info('Balance: %f' % self.balance)
        stk_value = 0
        for stk in self.stk_positions:
            # lastest_close 最新的close, 以防数据缺失
            if stk in price_input.index:
                self.stk_positions[stk]['latest_close'] = price_input[stk]
            else:
                pass
            stk_value += self.stk_positions[stk]['latest_close'] * self.stk_positions[stk]['volume']
        self.logger.info('Stock Value: %f' % stk_value)
        total_value = self.balance + stk_value
        self.logger.info('Total Value: %f' % total_value)
        self.logger.info('***************************************')
        return [self.balance, stk_value, total_value]


if __name__ == '__main__':
    portfolio = stock_portfolio('test')
    dt1 = datetime.datetime(1990, 9, 4)
    dt2 = datetime.datetime(1990, 11, 26)
    portfolio.trade_stks_to_target_volume(dt1, '000001.SZ', 5, 60)
    portfolio.buy_stks_by_volume(dt1, '000001.SZ', 6, 20)
    portfolio.trade_stks_to_target_volume(dt2, '000001.SZ', 66, 50)
    portfolio.trade_stks_to_target_volume(dt2, '000001.SZ', 66, 0)
    print('ok')
