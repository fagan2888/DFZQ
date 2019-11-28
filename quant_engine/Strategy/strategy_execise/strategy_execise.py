# 开始日期： 2019-11-7

from strategy_base import StrategyBase
import pandas as pd
from data_process import DataProcess

class strategy_exercise(StrategyBase):
    def __init__(self):
        super().__init__()

    def run(self,start,end):
        stk_info = self.get_basic_info(start,end)
        ep_cut = self.influx.getDataMultiprocess('DailyFactor_Gus','Value',start,end,None)
        ep_cut.index.names = ['date']
        ep_cut.reset_index(inplace=True)
        ep_cut = ep_cut.loc[:,['date','code','EPcut_TTM']]
        ep_cut = ep_cut.loc[pd.notnull(ep_cut['EPcut_TTM']),:]
        ep_cut['EPcut_TTM'] = DataProcess.Z_standardize(ep_cut['EPcut_TTM'])
        stk_info = pd.merge(stk_info,ep_cut,on=['date','code'])



        print('.')







if __name__ == '__main__':
    exe = strategy_exercise()
    exe.run(20150101,20160101)