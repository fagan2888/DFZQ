import pandas as pd
from portfolio import stock_portfolio,futures_portfolio
from influxdb_data import influxdbData
import dateutil.parser as dtparser
import logging
import datetime
import copy
import global_constant

class BacktestEngine:
    def __init__(self,stock_capital=1000000,stk_slippage=0.001,stk_fee=0.0001,save_name=None,logger_lvl=logging.INFO):
        self.stk_portfolio = stock_portfolio(capital_input=stock_capital,slippage_input=stk_slippage,transaction_fee_input=stk_fee)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logger_lvl)
        self.save_name = save_name
        dir = global_constant.ROOT_DIR+'Backtest_Result/Portfolio_Value/'
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

    def run(self,stk_weight,start,end,cash_reserve_rate=0.05,price_field='vwap'):
        backtest_starttime = datetime.datetime.now()
        self.logger.info('Start loading Data! %s' %backtest_starttime)
        influx = influxdbData()
        DB = 'DailyData_Gus'
        measure = 'marketData'
        daily_data = influx.getDataMultiprocess(DB,measure,str(start),str(end))
        self.logger.info('Data loaded! %s' %datetime.datetime.now())
        self.logger.info('****************************************\n')

        # 日线数据中的preclose已是相对前一天的复权价格
        exclude_col = ['IC_weight','IF_weight','IH_weight','citics_lv1_code','citics_lv1_name','citics_lv2_code',
                       'citics_lv2_name','citics_lv3_code','citics_lv3_name','sw_lv1_code','sw_lv1_name','sw_lv2_code',
                       'sw_lv2_name','isST']
        daily_data = daily_data.loc[:,daily_data.columns.difference(exclude_col)]
        daily_data['swap_date'] = pd.to_datetime(daily_data['swap_date'])
        daily_data[['bonus_share_ratio', 'cash_dvd_ratio', 'conversed_ratio', 'rightissue_price', 'rightissue_ratio']] = \
            daily_data[['bonus_share_ratio', 'cash_dvd_ratio', 'conversed_ratio', 'rightissue_price', 'rightissue_ratio']].fillna(0)

        daily_data.index.names = ['index']
        stk_weight.index.names = ['index']
        daily_data.reset_index(inplace=True)
        stk_weight.reset_index(inplace=True)
        daily_data = pd.merge(daily_data,stk_weight,on=['index','code'],how='outer')
        daily_data.set_index('index',inplace=True)
        daily_data['weight'] = daily_data['weight'].fillna(0)
        daily_data.sort_index(inplace=True)

        trade_days = daily_data.index.unique()
        positions_dict = {}
        portfolio_value_dict = {}
        total_value = self.stk_portfolio.get_portfolio_value(price_input=None)
        for trade_day in trade_days:
            self.logger.info('Trade Day: %s' %trade_day)
            one_day_data = daily_data.loc[trade_day,:].copy()
            one_day_data.set_index('code',inplace=True)
            ex_right = \
                one_day_data.loc[:,['bonus_share_ratio','cash_dvd_ratio','conversed_ratio','rightissue_price','rightissue_ratio']]
            self.stk_portfolio.process_ex_right(ex_right)
            target_capital = (1-cash_reserve_rate) * total_value
            one_day_data['target_volume'] = target_capital * one_day_data['weight'] /100 / one_day_data['preclose']
            trade_data = one_day_data.loc[(one_day_data['target_volume']>0)|
                                          (one_day_data.index.isin(self.stk_portfolio.stk_positions.keys())),:]
            for idx,row in trade_data.iterrows():
                if row['status'] == '停牌' or row['status'] != row['status']:
                    pass
                else:
                    self.stk_portfolio.trade_stks_to_target_volume(trade_day,idx,row[price_field],row['target_volume'])
            # 处理 吸收合并
            swap_info = one_day_data.loc[(one_day_data['swap_date']==trade_day) &
                                         (one_day_data.index.isin(self.stk_portfolio.stk_positions)), :]
            if swap_info.empty:
                pass
            else:
                for idx,row in swap_info.iterrows():
                    swap_price = self.stk_portfolio.stk_positions[idx]['price']/row['swap_ratio']
                    swap_volume = round(self.stk_portfolio.stk_positions[idx]['volume']*row['swap_ratio'])
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
                            self.stk_portfolio.stk_positions[idx]['latest_close']/row['swap_ratio']
                    self.stk_portfolio.stk_positions.pop(idx)

            # 记录 portfolio value
            total_value = self.stk_portfolio.get_portfolio_value(one_day_data['close'])
            portfolio_value_dict[trade_day] = \
                {'Balance':self.stk_portfolio.balance, 'StockValue':total_value-self.stk_portfolio.balance, 'TotalValue':total_value}
            self.logger.info(' -Balance: %f   -StockValue: %f   -TotalValue %f'
                             %(self.stk_portfolio.balance, total_value-self.stk_portfolio.balance, total_value))
            # 记录持仓
            positions_dict[trade_day] = copy.deepcopy(self.stk_portfolio.stk_positions)

        # 输出交易记录
        transactions = pd.concat(self.stk_portfolio.transactions_list, axis=1 ,ignore_index=True).T
        transactions.set_index('Time',inplace=True)
        if not self.save_name:
            filename = global_constant.ROOT_DIR + 'Transaction_Log/' + \
                       'Transactions_' + backtest_starttime.strftime("%Y%m%d-%H%M") + '.csv'
        else:
            filename = global_constant.ROOT_DIR + 'Transaction_Log/' + \
                       'Transactions_' + self.save_name + '.csv'
        transactions.to_csv(filename,encoding='gbk')
        # 输出持仓记录
        positions_dfs = []
        for time in positions_dict:
            trade_day_position = pd.DataFrame(positions_dict[time]).T
            trade_day_position['Time'] = time
            trade_day_position.index.name = 'Code'
            positions_dfs.append(trade_day_position.reset_index())
        positions = pd.concat(positions_dfs,ignore_index=True)
        positions.set_index('Time',inplace=True)
        if not self.save_name:
            filename = global_constant.ROOT_DIR + 'Backtest_Result/Positions/' + \
                       'Positions_' + backtest_starttime.strftime("%Y%m%d-%H%M") + '.csv'
        else:
            filename = global_constant.ROOT_DIR + 'Backtest_Result/Positions/' + \
                       'Positions_' + self.save_name + '.csv'
        positions.to_csv(filename,encoding='gbk')
        # 输出净值
        portfolio_value = pd.DataFrame(portfolio_value_dict).T
        if not self.save_name:
            filename = global_constant.ROOT_DIR + 'Backtest_Result/Portfolio_Value/' + \
                       'Backtest_' + backtest_starttime.strftime("%Y%m%d-%H%M") + '.csv'
        else:
            filename = global_constant.ROOT_DIR + 'Backtest_Result/Portfolio_Value/' + \
                       'Backtest_' + self.save_name + '.csv'
        portfolio_value.to_csv(filename,encoding='gbk')
        return portfolio_value


if __name__ == '__main__':
    influx = influxdbData()

    d = influx.getDataMultiprocess('DailyData_Gus','marketData','20120105','20160831')
    d = d.loc[pd.notnull(d['IF_weight']) & (d['volume']>0),['code','IF_weight','vwap']]
    d.columns = ['code','weight','vwap']
    d = d.loc[:,['code','weight']]
    start_time = datetime.datetime.now()
    QE = BacktestEngine(stock_capital=5000000,save_name='test',logger_lvl=logging.INFO)
    portfolio_value_dict = QE.run(d,20120106,20160831,price_field='vwap',cash_reserve_rate=0)
    print('backtest finish')
    print('time used:',datetime.datetime.now()-start_time)
    '''
    weight = pd.read_csv('D:/github/quant_engine/Backtest_Result/Factor_Group_Weight/EPcut_TTM_5groups.csv',encoding='gbk')
    weight.set_index('next_trade_day',inplace=True)
    weight.index = pd.to_datetime(weight.index)
    weight = weight.loc[weight['group']=='group_5',['code','weight']]
    start_time = datetime.datetime.now()
    QE = BacktestEngine(stock_capital=5000000, save_name='g5', logger_lvl=logging.INFO)
    portfolio_value_dict = QE.run(weight, 20120105, 20160831, price_field='vwap', cash_reserve_rate=0)
    print('backtest finish')
    print('time used:', datetime.datetime.now() - start_time)
    '''