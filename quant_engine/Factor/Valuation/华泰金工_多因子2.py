from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
import datetime

# EP = 净利润(TTM)/总市值
# EPcut = 扣非后净利润(TTM)/总市值
# BP = 净资产/总市值
# SP = 营业收入(TTM)/总市值
# NCFP = 净现金流(TTM)/总市值
# OCFP = 经营现金流(TTM)/总市值
# FCFP = 自由现金流(最新年报）/总市值
# DP = 近12个月现金红利(按除息日计)/总市值
# EV2EBITDA = 企业价值(扣除现金)/EBITDA(最新年报)
# PEG = (总市值)/净利润(TTM)) / 净利润(TTM)同比增长率

class Huatai_MF_2(FactorBase):
    def __init__(self):
        super().__init__()

    def cal_factor(self,start,end):
        # 表:AShareFinancialIndicator
        # 字段:S_INFO_WINDCODE, ANN_DT, REPORT_PERIOD, S_QFA_DEDUCTEDPROFIT
        # 起始时间往前推一年防止ttm为nan
        query = "select s_info_windcode, ann_dt, report_period, S_QFA_DEDUCTEDPROFIT " \
                "from wind_filesync.AShareFinancialIndicator " \
                "where ann_dt >= {0} and ann_dt <= {1} and (s_info_windcode like '0%' " \
                "or s_info_windcode like '3%' or s_info_windcode like '6%') "\
                "order by report_period"\
            .format((dtparser.parse(str(start))-relativedelta(years=1)).strftime('%Y%m%d'),str(end))
        self.rdf.curs.execute(query)
        FinIndi = pd.DataFrame(self.rdf.curs.fetchall(),columns=['code','ann_dt','report_period','profit_ddt_q'])
        FinIndi.dropna(inplace=True)
        codes = FinIndi['code'].unique()
        FinIndi.set_index('code',inplace=True)
        for code in codes:
            temp_df = FinIndi.loc[code,:].copy().sort_values(['report_period'])
            print('.')

        FinIndi['profit_ddt_ttm'] = FinIndi.groupby(FinIndi['code'])['profit_ddt_q'].rolling(4,min_periods=4).sum()
        FinIndi['ann_dt'] = pd.to_datetime(FinIndi['ann_dt'])
        FinIndi.set_index(['ann_dt','code'],inplace=True)

        # 表:AShareEODDerivativeIndicator
        # 字段:S_INFO_WINDCODE, TRADE_DT, S_VAL_MV, S_VAL_PE_TTM
        # 起始时间往前推一年防止ttm为nan
        query = "select s_info_windcode, TRADE_DT, S_VAL_PE_TTM, S_VAL_MV, NET_PROFIT_PARENT_COMP_TTM " \
                "from wind_filesync.AShareEODDerivativeIndicator " \
                "where trade_dt >= {0} and trade_dt <= {1} " \
                "order by trade_dt"\
            .format((dtparser.parse(str(start))-relativedelta(years=1)).strftime('%Y%m%d'),str(end))
        self.rdf.curs.execute(query)
        EODDerIndi = pd.DataFrame(self.rdf.curs.fetchall(),columns=['code','date','pe_ttm','mv','profit_ttm'])
        EODDerIndi['date'] = pd.to_datetime(EODDerIndi['date'])
        EODDerIndi.set_index(['date','code'],inplace=True)

        factors = EODDerIndi.join(FinIndi)
        print('.')



if __name__ == '__main__':
    aaa = Huatai_MF_2()
    aaa.cal_factor(20150101,20160101)