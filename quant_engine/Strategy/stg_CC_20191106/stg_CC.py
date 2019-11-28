# 开始日期： 2019-11-7

from strategy_base import StrategyBase
import pandas as pd
from dateutil.relativedelta import relativedelta
import dateutil.parser as dtparser
import joblib
import datetime


class stg_CC(StrategyBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def job_filter(code,ROE,ROE_mean_filter,ROE_lastest_filter):
        code_ROE = ROE.loc[ROE['code']==code,:].copy()
        code_ROE['ROE_ddt_mean'] = code_ROE['ROE_ddt'].rolling(700,300).mean()
        code_ROE = code_ROE.loc[(code_ROE['ROE_ddt']>=ROE_mean_filter)&(code_ROE['ROE_ddt_mean']>=ROE_lastest_filter),'code']
        print(code,'process finish')
        return code_ROE


    def run(self,start,end,DP_rate=0.02,ROE_mean_filter=0.15,ROE_latest_filter=0.15):
        # 获取股息率信息
        DP = self.influx.getDataMultiprocess('DailyFactor_Gus', 'Value', start, end, None)
        print('DP got!')
        # 筛选股息率达标的code
        DP = DP.loc[DP['DP_TTM'] >= DP_rate, ['code', 'DP_TTM']]
        # 起始时间往前3年，获取ROE信息
        ROE_start = (dtparser.parse(str(start)) - relativedelta(years=3)).strftime('%Y%m%d')
        ROE = self.influx.getDataMultiprocess('DailyFactor_Gus', 'FinancialQuality', ROE_start, end, None)
        print('ROE_ddt got!')
        ROE = ROE.loc[:, ['code', 'ROE_ddt']]

        df_list = joblib.Parallel(n_jobs=6)(joblib.delayed(stg_CC.job_run)(date,DP,ROE,ROE_mean_filter,ROE_latest_filter)
                                    for date in DP.index.unique())
        result_df = pd.concat(df_list,ignore_index=True)
        result_df = result_df.sort_values(by=['date','code'])
        result_df.to_csv('DP_ROE.csv',encoding='gbk')


    def run_simple(self,start,end,ROE_mean_filter=0.15,ROE_latest_filter=0.15):
        # 起始时间往前3年，获取ROE信息
        ROE_start = (dtparser.parse(str(start)) - relativedelta(years=3)).strftime('%Y%m%d')
        ROE = self.influx.getDataMultiprocess('DailyFactor_Gus', 'FinancialQuality', ROE_start, end, None)
        print('ROE_ddt got!')
        ROE = ROE.loc[:, ['code', 'ROE_ddt']]
        codes = ROE['code'].unique()

        result_list = []
        for code in codes.tolist():
            result_list.append(stg_CC.job_filter(code, ROE, ROE_mean_filter, ROE_latest_filter))
        result_df = pd.concat(result_list)
        result_df = result_df.sort_index()
        result_df = result_df.loc[dtparser.parse(str(start)):dtparser.parse(str(end))]
        result_df.to_csv('result.csv',encoding='gbk')


if __name__ == '__main__':
    exe = stg_CC()
    exe.run_simple(20120101,20190901)