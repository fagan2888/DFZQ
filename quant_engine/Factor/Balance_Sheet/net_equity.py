# 股东权益数据

from factor_base import FactorBase
import pandas as pd
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta


class net_equity(FactorBase):
    def __init__(self):
        super().__init__()

    def cal_factors(self,start,end):
        # type要用408001000，408005000，408004000(合并报表，合并更正前，合并调整后)，同时有408001000和408005000用408005000
        # 有408004000时，根据ann_dt酌情使用
        query = "select ANN_DT, S_INFO_WINDCODE, REPORT_PERIOD, TOT_SHRHLDR_EQY_EXCL_MIN_INT, STATEMENT_TYPE " \
                "from wind_filesync.AShareBalanceSheet " \
                "where ANN_DT >= {0} and ANN_DT <= {1} " \
                "and (STATEMENT_TYPE = '408001000' or STATEMENT_TYPE = '408005000' or STATEMENT_TYPE = '408004000') " \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "order by report_period, ann_dt, statement_type " \
            .format((dtparser.parse(str(start)) - relativedelta(years=2)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        balance_sheet = pd.DataFrame(self.rdf.curs.fetchall(),columns=['date', 'code', 'report_period', 'net_equity', 'type'])
        # 同一code，同一date，同一report_period，同时出现type1，2，3时，取type大的
        balance_sheet['type'] = balance_sheet['type'].apply(lambda x: '1' if x == '408001000' else ('2' if x == '408005000' else '3'))
        balance_sheet = balance_sheet.sort_values(by=['code', 'date', 'report_period', 'type'])
        net_equity = balance_sheet.dropna(subset=['net_equity']).groupby(['code', 'date', 'report_period'])['net_equity'].last()
        net_equity = pd.DataFrame(net_equity).reset_index()
        return net_equity


if __name__ == '__main__':
    neteqy_data = net_equity()
    r = neteqy_data.cal_factors(20100101,20190901)
    print('data got')
    h5 = pd.HDFStore('D:/github/quant_engine/Data_Resource/Balance_Sheet/net_equity.h5', 'w')
    h5['data'] = r
    h5.close()