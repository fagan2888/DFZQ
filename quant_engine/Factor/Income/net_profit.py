# 净利润数据
# 原数据

from factor_base import FactorBase
import pandas as pd
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
import global_constant

class net_profit(FactorBase):
    def __init__(self):
        super().__init__()

    def cal_factors(self,start,end):
        # type要用408001000，408005000，408004000(合并报表，合并更正前，合并调整后)，同时有408001000和408005000用408005000
        # 有408004000时，根据ann_dt酌情使用
        query = "select ANN_DT, S_INFO_WINDCODE, REPORT_PERIOD, NET_PROFIT_EXCL_MIN_INT_INC, " \
                "NET_PROFIT_AFTER_DED_NR_LP , STATEMENT_TYPE " \
                "from wind_filesync.AShareIncome " \
                "where ANN_DT >= {0} and ANN_DT <= {1} " \
                "and (STATEMENT_TYPE = '408001000' or STATEMENT_TYPE = '408005000' or STATEMENT_TYPE = '408004000') " \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "order by report_period, ann_dt, statement_type " \
            .format((dtparser.parse(str(start)) - relativedelta(years=2)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        income = pd.DataFrame(self.rdf.curs.fetchall(),
                              columns=['date', 'code', 'report_period', 'net_profit', 'net_profit_ddt', 'type'])
        # 同一code，同一date，同一report_period，同时出现type1，2，3时，取type大的
        income['type'] = income['type'].apply(lambda x: '1' if x == '408001000' else ('2' if x == '408005000' else '3'))
        income = income.sort_values(by=['code', 'date', 'report_period', 'type'])
        net_profit = income.dropna(subset=['net_profit']).groupby(['code', 'date', 'report_period'])['net_profit'].last()
        net_profit_ddt = income.dropna(subset=['net_profit_ddt']).groupby(['code', 'date', 'report_period'])['net_profit_ddt'].last()
        net_profit = pd.DataFrame(net_profit).reset_index()
        net_profit_ddt = pd.DataFrame(net_profit_ddt).reset_index()
        return [net_profit,net_profit_ddt]


if __name__ == '__main__':
    netprofit_data = net_profit()
    r = netprofit_data.cal_factors(20100101,20190901)
    print('data got')
    net_profit = r[0]
    net_profit_ddt = r[1]
    h5 = pd.HDFStore(global_constant.ROOT_DIR+'Data_Resource/Income/net_profit.h5', 'w')
    h5['data'] = net_profit
    h5.close()
    print('net profit saved')
    h5 = pd.HDFStore(global_constant.ROOT_DIR+'Data_Resource/Income/net_profit_ddt.h5', 'w')
    h5['data'] = net_profit_ddt
    h5.close()
    print('net profit ddt saved')