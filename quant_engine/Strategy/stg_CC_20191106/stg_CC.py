# 开始日期： 2019-11-7

from strategy_base import StrategyBase
import pandas as pd
from dateutil.relativedelta import relativedelta
import dateutil.parser as dtparser
import joblib


class stg_CC(StrategyBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def job_run(date,DP,ROE,ROE_mean_filter,ROE_latest_filter):
        day_DP = DP.loc[date, :].copy()
        DP_codes = day_DP['code'].unique()
        date_3yr_ago = date - relativedelta(years=3)
        satisified_code = []
        for code in DP_codes:
            day_code_ROE = ROE.loc[(ROE.index >= date_3yr_ago)&(ROE.index <= date)&(ROE['code'] == code),
                                   ['code', 'ROE_ddt']].copy()
            day_code_ROE = day_code_ROE.loc[pd.notnull(day_code_ROE['ROE_ddt']), :]
            if day_code_ROE.empty:
                continue
            ROE_mean = day_code_ROE['ROE_ddt'].unique().mean()
            if ROE_mean >= ROE_mean_filter and day_code_ROE['ROE_ddt'].iloc[-1] > ROE_latest_filter:
                satisified_code.append(code)
            else:
                continue
        result_df = pd.DataFrame({'code': satisified_code, 'date': [date] * len(satisified_code)})
        print(date,'process finish')
        return result_df


    def run(self,start,end,DP_rate=0.02,ROE_mean_filter=0.15,ROE_latest_filter=0.15,target_ftr='IF',industry_symbol='citics_lv1_name'):
        #weight = self.influx.getDataMultiprocess('DailyData_Gus','marketData',start,end,None)
        #weight_field = target_ftr + '_weight'
        # 获取权重股的代码，权重和行业信息
        #weight = weight.loc[weight[weight_field]>0,['code','status',weight_field,industry_symbol]]

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


if __name__ == '__main__':
    exe = stg_CC()
    exe.run(20120101,20190901)