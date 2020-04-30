import pandas as pd
from portfolio import stock_portfolio
from influxdb_data import influxdbData
import logging
import datetime
import copy
import global_constant
import os.path
from data_process import DataProcess


class BacktestEngine:
    def __init__(self, save_name, stock_capital=1000000, stk_slippage=0.001, stk_fee=0.0001, logger_lvl=logging.INFO):
        # 配置钱包
        self.stk_portfolio = \
            stock_portfolio(save_name, capital_input=stock_capital, slippage_input=stk_slippage,
                            transaction_fee_input=stk_fee)
        # 配置交易logger
        self.logger = logging.getLogger('transactions_log')
        self.logger.setLevel(level=logger_lvl)
        self.save_name = save_name
        self.dir = global_constant.ROOT_DIR + 'Backtest_Result/{0}/'.format(self.save_name)
        if os.path.exists(self.dir):
            pass
        else:
            os.makedirs(self.dir.rstrip('/'))
        file_name = 'Backtest_{0}.log'.format(self.save_name)
        handler = logging.FileHandler(self.dir + file_name)
        handler.setLevel(logging.INFO)
        console = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.addHandler(console)
        # 配置result logger
        self.res_logger = logging.getLogger('result_log')
        self.res_logger.setLevel(level=logger_lvl)
        file_name = 'Result_{0}.log'.format(self.save_name)
        handler = logging.FileHandler(self.dir + file_name)
        handler.setLevel(logging.INFO)
        console = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.res_logger.addHandler(handler)
        self.res_logger.addHandler(console)

    def initialize_engine(self, start, end, benchmark, price_field='vwap', indu_field='improved_lv1'):
        bm_dict = {50: '000016.SH', 300: '000300.SH', 500: '000905.SH'}
        self.benchmark_code = bm_dict[benchmark]
        influx = influxdbData()
        DB = 'DailyMarket_Gus'
        # 读取数据
        measure = 'market'
        self.market = influx.getDataMultiprocess(
            DB, measure, start, end, ['code', 'status', 'preclose', 'high', 'low', 'close', price_field])
        self.market.index.names = ['date']
        measure = 'exright'
        self.exright = influx.getDataMultiprocess(DB, measure, start, end)
        self.exright.index.names = ['date']
        self.exright = self.exright.fillna(0)
        measure = 'swap'
        self.swap = influx.getDataMultiprocess(DB, measure, start, end)
        self.swap.index.names = ['date']
        self.swap['str_date'] = self.swap.index.strftime('%Y%m%d')
        self.swap = self.swap.loc[self.swap['swap_date'] == self.swap['str_date'], :]
        # industry 为统计用数据
        measure = 'industry'
        self.industry = influx.getDataMultiprocess(DB, measure, start, end, ['code', indu_field])
        self.industry.index.names = ['date']
        self.calendar = self.market.index.unique()
        self.calendar = self.calendar.strftime('%Y%m%d')
        # benchmark 的报价 Series
        self.benchmark_quote = self.market.loc[self.market['code'] == self.benchmark_code, 'close'].copy()
        print('All Data Needed is ready...')


    # 默认输入的权重 以日期为index
    def run(self, stk_weight, start, end, adj_interval, benchmark, cash_reserve_rate=0.05,
            price_field='vwap', indu_field='improved_lv1'):
        backtest_starttime = datetime.datetime.now()
        # initialize engine
        self.initialize_engine(start, end, benchmark, price_field, indu_field)
        # 默认输入的权重 以日期为index
        stk_weight.index.names = ['date']
        mkt_with_weight = pd.merge(self.market.reset_index(), stk_weight.reset_index(),
                                   on=['date', 'code'], how='left')
        mkt_with_weight['weight'] = mkt_with_weight['weight'].fillna(0)
        mkt_with_weight.set_index('date', inplace=True)
        mkt_with_weight.sort_index(inplace=True)
        # backtest begins
        day_counter = 0
        positions_dict = {}
        portfolio_value_dict = {}
        balance, stk_value, total_value = self.stk_portfolio.get_portfolio_value(price_input=pd.Series([]))
        portfolio_start_value = total_value
        # 记录 benchmark 净值
        benchmark_networth = self.benchmark_quote[self.calendar[0]]
        benchmark_start_value = total_value
        for trade_day in self.calendar:
            self.logger.info('Trade Day: %s' % trade_day)
            day_mkt_with_weight = mkt_with_weight.loc[mkt_with_weight.index == trade_day, :].copy()
            day_mkt_with_weight.set_index('code', inplace=True)
            day_ex_right = self.exright.loc[self.exright.index == trade_day, :].copy()
            trade_amount = 0
            # 开盘前处理除权除息分红送股
            self.stk_portfolio.process_ex_right(day_ex_right)
            # 没有行情且在position里的stk记为退市，并统计退市金额
            delist_amount = 0
            no_quote_stks = set(self.stk_portfolio.stk_positions.keys()) - set(day_mkt_with_weight.index)
            for stk in no_quote_stks:
                delist_amount += self.stk_portfolio.stk_positions[stk]['volume'] * \
                                 self.stk_portfolio.stk_positions[stk]['latest_close']
            # 退市股票的金额需在stock_value中剔除，以免资金占用
            target_capital = (1 - cash_reserve_rate) * (total_value + delist_amount)
            # 计算目标成交量
            day_mkt_with_weight['target_volume'] = \
                target_capital * day_mkt_with_weight['weight'] / 100 / day_mkt_with_weight['preclose']
            # -----------------------------------------------------------------------------------------------
            # 每隔x天调仓
            if day_counter % adj_interval == 0:
                # 记录没法交易的股票
                # 记录 (weight不为0 或者 已在position中) 且 状态停牌 的票
                codes = day_mkt_with_weight.loc[
                    ((day_mkt_with_weight['target_volume'] > 0) |
                     (day_mkt_with_weight.index.isin(self.stk_portfolio.stk_positions.keys()))) &
                    (day_mkt_with_weight['status'] == '停牌'), :].index.values
                weights = day_mkt_with_weight.loc[
                    ((day_mkt_with_weight['target_volume'] > 0) |
                     (day_mkt_with_weight.index.isin(self.stk_portfolio.stk_positions.keys()))) &
                    (day_mkt_with_weight['status'] == '停牌'), 'weight'].values
                untradeable = dict(zip(codes, weights))
                # (weight不为0 或者 已在position中) 且 状态不停牌 的票
                tradeable_df = day_mkt_with_weight.loc[
                    ((day_mkt_with_weight['target_volume'] > 0) |
                     (day_mkt_with_weight.index.isin(self.stk_portfolio.stk_positions.keys()))) &
                    (day_mkt_with_weight['status'] != '停牌'), :].copy()
                for code, row in tradeable_df.iterrows():
                    if (row['low'] == row['high']) and (row['high'] >= round(row['preclose'] * 1.1, 2)):
                        price_limit = 'high'
                    elif (row['low'] == row['high']) and (row['low'] <= round(row['preclose'] * 0.9, 2)):
                        price_limit = 'low'
                    else:
                        price_limit = 'no_limit'
                    # 因涨跌停没有交易成功的股票代码会返回
                    trade_res = self.stk_portfolio.trade_stks_to_target_volume(
                        trade_day, code, row[price_field], row['target_volume'], price_limit)
                    # 记录可以交易，但是因为涨跌停无法交易 的票
                    if trade_res == 'Trade Fail':
                        untradeable[code] = row['weight']
                    # 记录双边的交易金额
                    else:
                        trade_amount += trade_res
            # 不是调仓日时，之前因涨跌停没买到的stks也要补
            else:
                # 没有行情的认为已退市，从失败列表中剔除
                no_quote_in_untradeable = set(untradeable.keys()) - set(day_mkt_with_weight.index)
                for stk in no_quote_in_untradeable:
                    untradeable.pop(stk)
                # untradeable 不为空时，补买卖stk
                if untradeable:
                    tradeable_df = day_mkt_with_weight.loc[
                                   (day_mkt_with_weight['status'] != '停牌') &
                                   day_mkt_with_weight.index.isin(untradeable.keys()), :].copy()
                    if not tradeable_df.empty:
                        tradeable_df['weight'] = tradeable_df.index
                        tradeable_df['weight'] = tradeable_df['weight'].map(untradeable)
                        tradeable_df['target_volume'] = \
                            target_capital * tradeable_df['weight'] / 100 / tradeable_df['preclose']
                        for code, row in tradeable_df.iterrows():
                            if (row['low'] == row['high']) and (row['high'] >= round(row['preclose'] * 1.1, 2)):
                                price_limit = 'high'
                            elif (row['low'] == row['high']) and (row['low'] <= round(row['preclose'] * 0.9, 2)):
                                price_limit = 'low'
                            else:
                                price_limit = 'no_limit'
                            trade_res = self.stk_portfolio.trade_stks_to_target_volume(
                                trade_day, code, row[price_field], row['target_volume'], price_limit)
                            if trade_res == 'Trade Fail':
                                pass
                            else:
                                trade_amount += trade_res
                                untradeable.pop(code)
            # -------------------------------------------------------------------------------
            # 处理 吸收合并
            day_swap = self.swap.loc[self.swap['code'].isin(self.stk_portfolio.stk_positions.keys()) &
                                     (self.swap.index == trade_day), :].copy()
            if not day_swap.empty:
                for date, row in day_swap.iterrows():
                    swap_price = self.stk_portfolio.stk_positions[row['code']]['price'] / row['swap_ratio']
                    swap_volume = round(self.stk_portfolio.stk_positions[row['code']]['volume'] * row['swap_ratio'])
                    if row['swap_code'] in self.stk_portfolio.stk_positions:
                        merged_volume = self.stk_portfolio.stk_positions[row['swap_code']]['volume'] + swap_volume
                        merged_price = (self.stk_portfolio.stk_positions[row['swap_code']]['volume'] *
                                        self.stk_portfolio.stk_positions[row['swap_code']]['price'] +
                                        swap_volume * swap_price) / merged_volume
                        self.stk_portfolio.stk_positions[row['swap_code']]['volume'] = merged_volume
                        self.stk_portfolio.stk_positions[row['swap_code']]['price'] = merged_price
                    else:
                        self.stk_portfolio.stk_positions[row['swap_code']] = {}
                        self.stk_portfolio.stk_positions[row['swap_code']]['volume'] = swap_volume
                        self.stk_portfolio.stk_positions[row['swap_code']]['price'] = swap_price
                        self.stk_portfolio.stk_positions[row['swap_code']]['latest_close'] = \
                            self.stk_portfolio.stk_positions[row['code']]['latest_close'] / row['swap_ratio']
                    self.stk_portfolio.stk_positions.pop(row['code'])
            # 记录 已有持仓中停牌的stk
            suspend_stks_in_pos = \
                day_mkt_with_weight.loc[
                day_mkt_with_weight.index.isin(self.stk_portfolio.stk_positions.keys()) &
                (day_mkt_with_weight['status'] == '停牌'), :].index.values
            # 记录 benchmark value
            # 记录 portfolio value
            # 记录 accum_alpha
            # 记录双边换手率 turnover
            benchmark_value = benchmark_start_value / benchmark_networth * self.benchmark_quote[trade_day]
            balance, stk_value, total_value = self.stk_portfolio.get_portfolio_value(day_mkt_with_weight['close'])
            accum_alpha = total_value / portfolio_start_value - benchmark_value / benchmark_start_value
            if stk_value == 0:
                turnover = 0
            else:
                turnover = trade_amount / stk_value
            portfolio_value_dict[trade_day] = \
                {'Balance': balance, 'StockValue': stk_value, 'TotalValue': total_value,
                 'DelistAmount': delist_amount, 'SuspendStk': len(suspend_stks_in_pos),
                 'BenchmarkValue': benchmark_value, 'AccumAlpha': accum_alpha, 'Turnover': turnover}
            self.logger.info(' \n -Balance: %f \n -StockValue: %f \n -TotalValue %f \n -DelistAmount: %f \n'
                             ' -SuspendStk: %i \n -BenchmarkValue: %f \n -AccumAlpha: %f \n -Turnover: %f'
                             % (balance, stk_value, total_value, delist_amount, len(suspend_stks_in_pos),
                                benchmark_value, accum_alpha, turnover))
            # 记录 持仓
            positions_dict[trade_day] = copy.deepcopy(self.stk_portfolio.stk_positions)
            day_counter += 1
        # 输出交易记录
        transactions = pd.DataFrame(self.stk_portfolio.transactions.
                                    reshape(int(self.stk_portfolio.transactions.shape[0]/8), 8),
                                    columns=['Time', 'Direction', 'Code', 'RawPrice', 'ActualPrice', 'Volume',
                                             'Balance', 'TransFee'])
        transactions.set_index('Time', inplace=True)
        filename = self.dir + 'Transactions_' + self.save_name + '.csv'
        transactions.to_csv(filename, encoding='gbk')
        # 输出持仓记录
        positions_dfs = []
        for time in positions_dict:
            trade_day_position = pd.DataFrame(positions_dict[time]).T
            trade_day_position['Time'] = time
            trade_day_position.index.name = 'Code'
            positions_dfs.append(trade_day_position.reset_index())
        positions = pd.concat(positions_dfs, ignore_index=True)
        positions.set_index('Time', inplace=True)
        filename = self.dir + 'Positions_{0}.csv'.format(self.save_name)
        positions.to_csv(filename, encoding='gbk')
        # 输出净值
        portfolio_value = pd.DataFrame(portfolio_value_dict).T
        filename = self.dir + 'Value_{0}.csv'.format(self.save_name)
        portfolio_value.to_csv(filename, encoding='gbk')
        self.res_logger.info('Backtest finish time: %s' % datetime.datetime.now().strftime('%Y/%m/%d - %H:%M:%S'))
        self.res_logger.info('*' * 50)
        self.res_logger.info('PERFORMANCE:')
        self.res_logger.info('-ANN_Alpha: %f' % DataProcess.calc_alpha_ann_return(
            portfolio_value['TotalValue'], portfolio_value['BenchmarkValue']))
        self.res_logger.info('-Alpha_MDD: %f' % DataProcess.calc_alpha_max_draw_down(
            portfolio_value['TotalValue'], portfolio_value['BenchmarkValue']))
        self.res_logger.info('-Alpha_sharpe: %f' % DataProcess.calc_alpha_sharpe(
            portfolio_value['TotalValue'], portfolio_value['BenchmarkValue']))
        print('Backtest finish! Time used: ', datetime.datetime.now() - backtest_starttime)
        return portfolio_value


if __name__ == '__main__':
    weight = pd.read_csv(global_constant.ROOT_DIR + 'Strategy/Alpha_version_3/TARGET_WEIGHT.csv')
    weight = weight.loc[:, ['date', 'code', 'weight']].copy()
    weight['date'] = pd.to_datetime(weight['date'])
    weight.set_index('date', inplace=True)
    print('Weight_loaded')
    QE = BacktestEngine(stock_capital=100000000, save_name='alpha3.0', logger_lvl=logging.INFO)
    QE.run(weight, 20130101, 20200401, 2, 300)