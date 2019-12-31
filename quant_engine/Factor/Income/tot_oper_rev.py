# 净利润数据
# 原数据

from factor_base import FactorBase
import pandas as pd
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
import global_constant

class tot_oper_rev(FactorBase):
    def __init__(self):
        super().__init__()

    def cal_factors(self,start,end):
        # type要用408001000，408005000，408004000(合并报表，合并更正前，合并调整后)，同时有408001000和408005000用408005000
        # 有408004000时，根据ann_dt酌情使用
        query = "select ANN_DT, S_INFO_WINDCODE, REPORT_PERIOD, TOT_OPER_REV, STATEMENT_TYPE " \
                "from wind_filesync.AShareIncome " \
                "where ANN_DT >= {0} and ANN_DT <= {1} " \
                "and (STATEMENT_TYPE = '408001000' or STATEMENT_TYPE = '408005000' or STATEMENT_TYPE = '408004000') " \
                "and (s_info_windcode like '0%' or s_info_windcode like '3%' or s_info_windcode like '60%') " \
                "order by report_period, ann_dt, statement_type " \
            .format((dtparser.parse(str(start)) - relativedelta(years=2)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        income = pd.DataFrame(self.rdf.curs.fetchall(),
                              columns=['date', 'code', 'report_period', 'tot_oper_rev', 'type'])
        # 同一code，同一date，同一report_period，同时出现type1，2，3时，取type大的
        income['type'] = income['type'].apply(lambda x: '1' if x == '408001000' else ('2' if x == '408005000' else '3'))
        income = income.sort_values(by=['code', 'date', 'report_period', 'type'])
        tot_oper_rev = income.dropna(subset=['tot_oper_rev']).groupby(['code', 'date', 'report_period'])['tot_oper_rev'].last()
        tot_oper_rev = pd.DataFrame(tot_oper_rev).reset_index()
        return tot_oper_rev


if __name__ == '__main__':
    tot_oper_rev = tot_oper_rev()
    r = tot_oper_rev.cal_factors(20100101,20190901)
    print('data got')
    h5 = pd.HDFStore(global_constant.ROOT_DIR+'Data_Resource/Income/tot_oper_rev.h5', 'w')
    h5['data'] = r
    h5.close()
    print('data saved')