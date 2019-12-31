# 盈利能力因子 ROE,ROE_Q,ROE_ddt,ROE_ddt_Q的计算

from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
import datetime
import joblib

class ROE(FactorBase):
    def __init__(self):
        super().__init__()


    def job_ROE(self,code):
        code_ROE = self.ROE.loc[self.ROE['code']==code,:].copy()
        code_ROE['ROE'] = code_ROE['net_profit_TTM']/(code_ROE['net_equity_lastY']+code_ROE['net_equity'])*2
        code_ROE['ROE_Q'] = code_ROE['net_profit_Q']/(code_ROE['net_equity_last1Q']+code_ROE['net_equity'])*2
        code_ROE = code_ROE.loc[:,['code','ROE','ROE_Q']]
        code_ROE = code_ROE.loc[pd.notnull(code_ROE['ROE'])|pd.notnull(code_ROE['ROE_Q']),:]
        code_ROE = code_ROE.where(pd.notnull(code_ROE), None)
        print('ROE task: %s' %code)
        #self.save_factor_to_influx(code_ROE, 'DailyFactor_Gus', 'FinancialQuality')
        return


    def job_ROE_ddt(self,code):
        code_ROE_ddt = self.ROE_ddt.loc[self.ROE_ddt['code']==code,:].copy()
        code_ROE_ddt['ROE_ddt'] = code_ROE_ddt['net_profit_ddt_TTM']/\
                                  (code_ROE_ddt['net_equity_lastY']+code_ROE_ddt['net_equity'])*2
        code_ROE_ddt['ROE_ddt_Q'] = code_ROE_ddt['net_profit_ddt_Q']/\
                                    (code_ROE_ddt['net_equity_last1Q']+code_ROE_ddt['net_equity'])*2
        code_ROE_ddt = code_ROE_ddt.loc[:,['code','ROE_ddt','ROE_ddt_Q']]
        code_ROE_ddt = code_ROE_ddt.loc[pd.notnull(code_ROE_ddt['ROE_ddt'])|pd.notnull(code_ROE_ddt['ROE_ddt_Q']),:]
        code_ROE_ddt = code_ROE_ddt.where(pd.notnull(code_ROE_ddt),None)
        print('ROE ddt task: %s' %code)
        #self.save_factor_to_influx(code_ROE_ddt, 'DailyFactor_Gus', 'FinancialQuality')
        return


    def cal_factors(self,start,end):
        print('task start!')
        self.start = str(start)
        self.end = str(end)
        start_time = datetime.datetime.now()
        print(start_time)

        net_profit = self.influx.getDataMultiprocess('Financial_Report_Gus', 'net_profit',self.start,self.end)
        print('net_profit got')
        net_profit_ddt = self.influx.getDataMultiprocess('Financial_Report_Gus', 'net_profit_ddt',self.start,self.end)
        print('net_profit_ddt got')
        net_equity = self.influx.getDataMultiprocess('Financial_Report_Gus', 'net_equity',self.start,self.end)
        print('net_equity got')

        # 净资产向前填充
        net_equity.loc[:,['net_equity', 'net_equity_last1Q', 'net_equity_last2Q', 'net_equity_last3Q','net_equity_lastY']] \
            = net_equity.loc[:,['net_equity', 'net_equity_last1Q', 'net_equity_last2Q','net_equity_last3Q','net_equity_lastY']]\
            .fillna(method='ffill',axis=1)
        net_profit = net_profit.loc[:,['code','net_profit_Q','net_profit_TTM','report_period']].reset_index()
        net_profit_ddt = net_profit_ddt.loc[:,['code','net_profit_ddt_Q','net_profit_ddt_TTM','report_period']].reset_index()
        net_equity = net_equity.loc[:,['code','net_equity','net_equity_last1Q','net_equity_lastY']].reset_index()

        self.ROE = pd.merge(net_profit,net_equity,how='outer',left_on=['index','code'],right_on=['index','code'])
        self.ROE_ddt = pd.merge(net_profit_ddt,net_equity,how='outer',left_on=['index','code'],right_on=['index','code'])
        self.ROE.set_index('index',inplace=True)
        self.ROE_ddt.set_index('index',inplace=True)

        joblib.Parallel(n_jobs=-1)(joblib.delayed(self.job_ROE)(code)
                          for code in self.ROE['code'].unique())
        joblib.Parallel(n_jobs=-1)(joblib.delayed(self.job_ROE_ddt)(code)
                          for code in self.ROE_ddt['code'].unique())
        print('task finish!')
        print(datetime.datetime.now()-start_time)


if __name__ == '__main__':
    pd.set_option('mode.use_inf_as_na', True)
    roe = ROE()
    roe.cal_factors(20100101,20190901)