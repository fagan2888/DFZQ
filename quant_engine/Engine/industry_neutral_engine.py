import pandas as pd
import numpy as np
from industry_neutral_portfolio import industry_neutral_portfolio
from influxdb_data import influxdbData
from rdf_data import rdf_data
import dateutil.parser as dtparser
import logging
import datetime
import copy
import global_constant


class IndustryNeutralEngine:
    def __init__(self, stock_capital=100000000, stk_slippage=0.001, stk_fee=0.0001, save_name=None,
                 logger_lvl=logging.INFO):
        self.stk_portfolio = industry_neutral_portfolio(capital_input=stock_capital, slippage_input=stk_slippage,
                                                        transaction_fee_input=stk_fee)
        self.rdf = rdf_data()
        self.calendar = self.rdf.get_trading_calendar()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logger_lvl)
        self.save_name = save_name
        dir = global_constant.ROOT_DIR + 'Backtest_Result/Portfolio_Value/'
        if not self.save_name:
            file_name = 'Backtest_' + datetime.datetime.now().strftime("%Y%m%d-%H%M") + '.log'
        else:
            file_name = save_name + '.log'
        handler = logging.FileHandler(dir + file_name)
        handler.setLevel(logging.INFO)
        console = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.addHandler(console)

    def get_next_trade_day(self, mkt_data, days):
        trade_day = pd.DataFrame(mkt_data.index.unique().tolist(), columns=['date'])
        field = 'next_' + str(days) + '_day'
        trade_day[field] = trade_day['date'].apply(lambda x: self.calendar[self.calendar > x].iloc[days - 1])
        trade_day.set_index('date', inplace=True)
        return trade_day

    # 传入stk在该行业的权重
    def run(self, stk_indu_weight, start, end, benchmark, adj_interval,
            cash_reserve_rate=0.05, price_field='vwap', indu_field='improved_lv1', data_input=None):
        backtest_starttime = datetime.datetime.now()
        # set benchmark weight field
        benchmark_dict = {50: 'IH_weight', 300: 'IF_weight', 500: 'IC_weight'}
        benchmark_weight = benchmark_dict[benchmark]
        # load data
        if not isinstance(data_input, pd.DataFrame):
            self.logger.info('Start loading Data! %s' % backtest_starttime)
            influx = influxdbData()
            DB = 'DailyData_Gus'
            measure = 'marketData'
            daily_data = influx.getDataMultiprocess(DB, measure, start, end)
        else:
            daily_data = data_input.loc[start:end, :].copy()
        if 'swap_date' not in daily_data.columns:
            daily_data['swap_date'] = None
        self.logger.info('Data loaded! %s' % datetime.datetime.now())
        self.logger.info('****************************************\n')
        # 日线数据中的preclose已是相对前一天的复权价格
        exclude_col = ['IC_weight', 'IF_weight', 'IH_weight', 'improved_lv1', 'citics_lv1_code', 'citics_lv1_name',
                       'citics_lv2_code', 'citics_lv2_name', 'citics_lv3_code', 'citics_lv3_name', 'sw_lv1_code',
                       'sw_lv1_name', 'sw_lv2_code', 'sw_lv2_name', 'isST']
        exclude_col.remove(benchmark_weight)
        exclude_col.remove(indu_field)
        daily_data = daily_data.loc[:, daily_data.columns.difference(exclude_col)]
        daily_data.rename(columns={indu_field: 'industry'}, inplace=True)
        daily_data.dropna(subset=['industry'], inplace=True)
        daily_data['swap_date'] = pd.to_datetime(daily_data['swap_date'])
        ex_right_related_cols = \
            ['bonus_share_ratio', 'cash_dvd_ratio', 'conversed_ratio', 'rightissue_price', 'rightissue_ratio']
        daily_data[ex_right_related_cols] = daily_data[ex_right_related_cols].fillna(0)
        # merge得到下一日的benchmark weight
        nxt_trade_day = self.get_next_trade_day(daily_data, 1)
        nxt_trade_day.index.names = ['date']
        nxt_trade_day.reset_index(inplace=True)
        daily_data.index.names = ['date']
        daily_data.reset_index(inplace=True)
        daily_data = pd.merge(daily_data, nxt_trade_day, how='left', on='date')
        nxt_day_benchmark_weight = daily_data.loc[:, ['date', benchmark_weight, 'industry']].copy()
        nxt_day_industry_weight = pd.DataFrame(
            nxt_day_benchmark_weight.groupby(['date', 'industry'])[benchmark_weight].sum())
        nxt_day_industry_weight.reset_index(inplace=True)
        nxt_day_industry_weight.rename(
            columns={'date': 'next_1_day', benchmark_weight: 'industry_weight'}, inplace=True)
        daily_data = pd.merge(daily_data, nxt_day_industry_weight, on=['next_1_day', 'industry'], how='outer')
        daily_data = daily_data.dropna(subset=['date'])
        # 现将股票行业内权重与行业权重合并
        stk_indu_weight = stk_indu_weight.loc[start:end, :]
        stk_indu_weight.index.names = ['date']
        stk_indu_weight.reset_index(inplace=True)
        stk_indu_weight = stk_indu_weight.drop('industry', axis=1)
        # 把行情数据中的industry删掉，以免跟权重中的industry字段冲突
        # 合并stk indu weight
        daily_data = pd.merge(daily_data, stk_indu_weight, on=['date', 'code'], how='left')
        daily_data.set_index('date', inplace=True)
        daily_data['weight_in_industry'] = daily_data['weight_in_industry'].fillna(0)

        trade_days = daily_data.index.unique()
        positions_dict = {}
        portfolio_value_dict = {}
        day_counter = 0
        total_value = self.stk_portfolio.get_portfolio_value(price_input=None)
        for trade_day in trade_days:
            self.logger.info('Trade Day: %s' % trade_day)
            # 处理除权除息
            one_day_data = daily_data.loc[trade_day, :].copy()
            one_day_data.set_index('code', inplace=True)
            ex_right = one_day_data.loc[:, ex_right_related_cols]
            self.stk_portfolio.process_ex_right(ex_right)
            # 没有行情且在pos里的stks记为退市
            # 统计退市金额
            delist_amount = 0
            locked_amount = {}
            no_quote_stks = set(self.stk_portfolio.stk_positions.keys()) - set(one_day_data.index)
            for stk in no_quote_stks:
                delist_amount += self.stk_portfolio.stk_positions[stk]['volume'] * \
                                 self.stk_portfolio.stk_positions[stk]['latest_close']
            # 退市股票的金额需在stock_value中剔除，以免资金占用
            target_capital = (1 - cash_reserve_rate) * (total_value + delist_amount)
            # 在pos中且状态为停牌的stks
            suspend_stks_in_pos = one_day_data.loc[
                                  ((one_day_data['status'] == '停牌') | pd.isnull(one_day_data['status'])) &
                                  one_day_data.index.isin(self.stk_portfolio.stk_positions.keys()), :].index
            # -----------------------------------------------------------------------------------------------
            # 每隔x天调仓
            if day_counter % adj_interval == 0:
                # 对有目标权重，却停牌或者没有行情的情况，重新分配行业内部权重
                one_day_pause_data = one_day_data.loc[(one_day_data['weight_in_industry'] > 0) &
                                                      ((one_day_data['status'] == '停牌') |
                                                       pd.isnull(one_day_data['status'])), :]
                one_day_indu_pause_weight = pd.DataFrame(
                    one_day_pause_data.groupby('industry')['weight_in_industry'].sum())
                # 将停牌股票权重扣除，将剩余股票权重等比放大
                for industry, row in one_day_indu_pause_weight.iterrows():
                    one_day_data.loc[one_day_data['industry'] == industry, 'weight_in_industry'] = \
                        one_day_data.loc[one_day_data['industry'] == industry, 'weight_in_industry'] / \
                        (100 - row['weight_in_industry']) * 100
                # 权重放大后，将停牌的票的weight_in_industry置为nan
                one_day_data.loc[(one_day_data['status'] == '停牌') | pd.isnull(one_day_data['status']),
                                 'weight_in_industry'] = np.nan
                # 统计行业的停牌金额
                for stk in suspend_stks_in_pos:
                    stk_industry = self.stk_portfolio.stk_positions[stk]['industry']
                    if stk_industry in locked_amount:
                        locked_amount[stk_industry] += self.stk_portfolio.stk_positions[stk]['volume'] * \
                                                       self.stk_portfolio.stk_positions[stk]['latest_close']
                    else:
                        locked_amount[stk_industry] = self.stk_portfolio.stk_positions[stk]['volume'] * \
                                                      self.stk_portfolio.stk_positions[stk]['latest_close']
                # 通过weight计算当天股票的目标volume
                # 如果该行业持仓停牌的金额已超过行业需配的金额，置为0
                one_day_data['industry_amount'] = \
                    one_day_data.apply(
                        lambda row:
                        max(target_capital * row['industry_weight'] / 100 - locked_amount[row['industry']], 0)
                        if row['industry'] in locked_amount
                        else target_capital * row['industry_weight'] / 100, axis=1)
                one_day_data['target_volume'] = \
                    one_day_data['industry_amount'] * one_day_data['weight_in_industry'] / 100 / one_day_data['preclose']

                # 可交易的且有weight的stks + 可交易的且在pos里却没有weight的stks
                trade_data = one_day_data.loc[
                             ((one_day_data['status'] != '停牌') & (pd.notnull(one_day_data['status']))) &
                             (pd.notnull(one_day_data['target_volume']) |
                              (one_day_data.index.isin(self.stk_portfolio.stk_positions.keys()))), :].copy()
                trade_data['target_volume'] = trade_data['target_volume'].fillna(0)
                fail_to_trade_stks = {}
                for idx, row in trade_data.iterrows():
                    if (row['low'] == row['high']) and (row['high'] >= round(row['preclose'] * 1.1, 2)):
                        price_limit = 'high'
                    elif (row['low'] == row['high']) and (row['low'] <= round(row['preclose'] * 0.9, 2)):
                        price_limit = 'low'
                    else:
                        price_limit = 'no_limit'
                    # 因涨跌停没有交易成功的股票代码会返回
                    trade_res = self.stk_portfolio.trade_stks_to_target_volume(trade_day, idx, row['industry'],
                                    row[price_field], row['target_volume'], price_limit)
                    if trade_res == 'Trade Fail':
                        stk_amount = row['preclose'] * row['target_volume']
                        fail_to_trade_stks[idx] = stk_amount
                    else:
                        pass
            else:
                # 不是调仓日时，之前因涨跌停没买到的stks也要补
                # 没有行情的认为已退市，从失败列表中剔除
                no_quote_fail_stks = set(fail_to_trade_stks.keys()) - set(one_day_data.index)
                for stk in no_quote_fail_stks:
                    fail_to_trade_stks.pop(stk)
                if not fail_to_trade_stks:
                    pass
                else:
                    trade_data = one_day_data.loc[
                                 ((one_day_data['status'] != '停牌') & (pd.notnull(one_day_data['status']))) &
                                 one_day_data.index.isin(fail_to_trade_stks.keys()), :].copy()
                    if trade_data.empty:
                        pass
                    else:
                        trade_data['target_volume'] = \
                            trade_data.apply(lambda row: fail_to_trade_stks[row.name] / row['preclose'], axis=1)
                        for idx, row in trade_data.iterrows():
                            if (row['low'] == row['high']) and (row['high'] >= round(row['preclose'] * 1.1, 2)):
                                price_limit = 'high'
                            elif (row['low'] == row['high']) and (row['low'] <= round(row['preclose'] * 0.9, 2)):
                                price_limit = 'low'
                            else:
                                price_limit = 'no_limit'
                            trade_res = self.stk_portfolio.trade_stks_to_target_volume(trade_day, idx, row['industry'],
                                            row[price_field], row['target_volume'], price_limit)
                            if trade_res == 'Trade Succeed':
                                fail_to_trade_stks.pop(idx)
                            else:
                                pass
            # -------------------------------------------------------------------------------
            # 处理 吸收合并
            swap_info = one_day_data.loc[(one_day_data['swap_date'] == trade_day) &
                                         (one_day_data.index.isin(self.stk_portfolio.stk_positions)), :]
            if swap_info.empty:
                pass
            else:
                for idx, row in swap_info.iterrows():
                    swap_price = self.stk_portfolio.stk_positions[idx]['price'] / row['swap_ratio']
                    swap_volume = round(self.stk_portfolio.stk_positions[idx]['volume'] * row['swap_ratio'])
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
                            self.stk_portfolio.stk_positions[idx]['latest_close'] / row['swap_ratio']
                    self.stk_portfolio.stk_positions.pop(idx)
            # 刷新所属行业
            self.stk_portfolio.refresh_portfolio_industry(one_day_data['industry'])
            # 记录 portfolio value 和 locked amount
            total_value = self.stk_portfolio.get_portfolio_value(one_day_data['close'].dropna())
            portfolio_value_dict[trade_day] = \
                {'Balance': self.stk_portfolio.balance, 'StockValue': total_value - self.stk_portfolio.balance,
                 'TotalValue': total_value, 'DelistAmount': delist_amount,
                 'SuspendedStks': suspend_stks_in_pos.shape[0]}
            self.logger.info(' -Balance: %f   -StockValue: %f   -TotalValue %f   -DelistAmount %f   -SuspendedStks %i'
                             % (self.stk_portfolio.balance, total_value - self.stk_portfolio.balance,
                                total_value, delist_amount, suspend_stks_in_pos.shape[0]))
            # 记录持仓
            positions_dict[trade_day] = copy.deepcopy(self.stk_portfolio.stk_positions)
            day_counter += 1

        # 输出交易记录
        transactions = pd.concat(self.stk_portfolio.transactions_list, axis=1, ignore_index=True).T
        transactions.set_index('Time', inplace=True)
        if not self.save_name:
            filename = global_constant.ROOT_DIR + 'Transaction_Log/' + \
                       'Transactions_' + backtest_starttime.strftime("%Y%m%d-%H%M") + '.csv'
        else:
            filename = global_constant.ROOT_DIR + 'Transaction_Log/' + \
                       'Transactions_' + self.save_name + '.csv'
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
        if not self.save_name:
            filename = global_constant.ROOT_DIR + 'Backtest_Result/Positions/' + \
                       'Positions_' + backtest_starttime.strftime("%Y%m%d-%H%M") + '.csv'
        else:
            filename = global_constant.ROOT_DIR + 'Backtest_Result/Positions/' + \
                       'Positions_' + self.save_name + '.csv'
        positions.to_csv(filename, encoding='gbk')
        # 输出净值
        portfolio_value = pd.DataFrame(portfolio_value_dict).T
        if not self.save_name:
            filename = global_constant.ROOT_DIR + 'Backtest_Result/Portfolio_Value/' + \
                       'Backtest_' + backtest_starttime.strftime("%Y%m%d-%H%M") + '.csv'
        else:
            filename = global_constant.ROOT_DIR + 'Backtest_Result/Portfolio_Value/' + \
                       'Backtest_' + self.save_name + '.csv'
        portfolio_value.to_csv(filename, encoding='gbk')
        return portfolio_value


if __name__ == '__main__':
    d = pd.read_csv('D:/github/quant_engine/Backtest_Result/Factor_Group_Weight/g5.csv', encoding='gbk')
    d.drop('Unnamed: 0', axis=1, inplace=True)
    d['next_1_day'] = pd.to_datetime(d['next_1_day'])
    d.set_index('next_1_day', inplace=True)
    start_time = datetime.datetime.now()
    QE = IndustryNeutralEngine(stock_capital=5000000, save_name='g5_5', logger_lvl=logging.INFO)
    portfolio_value_dict = QE.run(d, 20120112, 20160831, adj_interval=5,
                                  benchmark='IC', price_field='vwap', cash_reserve_rate=0.03)
    print('backtest finish')
    print('time used:', datetime.datetime.now() - start_time)
