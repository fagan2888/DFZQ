from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
import datetime
import joblib

class FCFF_FCFF2CS(FactorBase):
    def __init__(self):
        super().__init__()

    def cal_factors(self,start,end):
        # FCFF = 经营活动产生的现金流量净额  NET_CASH_FLOWS_OPER_ACT(@AShareCashFlow)
        #      + 投资活动产生的现金流量净额  NET_CASH_FLOWS_INV_ACT(@AShareCashFlow)
        #      - 处置固定资产、无形资产和其他长期资产收回的现金净额  NET_CASH_RECP_DISP_FIOLTA(@AShareCashFlow)
        #      + 构建固定资产、无形资产和其他长期资产支付的现金净额  CASH_PAY_ACQ_CONST_FIOLTA(@AShareCashFlow)
        #      - 收回投资收到的现金  CASH_RECP_DISP_WITHDRWL_INVEST(@AShareCashFlow)
        #      + 投资支付的现金  CASH_PAID_INVEST(@AShareCashFlow)
        #      - 分配股利、利润或偿付利息支付的现金  CASH_PAY_DIST_DPCP_INT_EXP(@AShareCashFlow)

        query = "select s_info_windcode, ANN_DT, REPORT_PERIOD, NET_CASH_FLOWS_OPER_ACT, NET_CASH_FLOWS_INV_ACT, " \
                "NET_CASH_RECP_DISP_FIOLTA, CASH_PAY_ACQ_CONST_FIOLTA, CASH_RECP_DISP_WITHDRWL_INVEST, " \
                "CASH_PAID_INVEST, CASH_PAY_DIST_DPCP_INT_EXP " \
                "from wind_filesync.AShareCashFlow " \
                "where ANN_DT >= {0} and ANN_DT <= {1} " \
                "and (s_info_windcode like '0%' " \
                "or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "order by ANN_DT" \
            .format((dtparser.parse(str(start)) - relativedelta(years=1)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        CashFlow = pd.DataFrame(self.rdf.curs.fetchall(),columns=['code','date','report_period','NCF_OperAct',
                'NCF_InvAct','Cash_Recp_Fiolta','Cash_Pay_Fiolta','Cash_Recp_Invest','Cash_Pay_Invest','Cash_Pay_DPCP'])
        CashFlow = CashFlow.fillna(0)
        CashFlow['FCFF_own'] = CashFlow['NCF_OperAct'] + CashFlow['NCF_InvAct'] - CashFlow['Cash_Recp_Fiolta'] + \
                               CashFlow['Cash_Pay_Fiolta'] - CashFlow['Cash_Recp_Invest'] + \
                               CashFlow['Cash_Pay_Invest'] - CashFlow['Cash_Pay_DPCP']
        print('.')



if __name__ == '__main__':
    fcff = FCFF_FCFF2CS()
    fcff.cal_factors(20150101,20190101)