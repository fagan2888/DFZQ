from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
import datetime
import joblib

# EP = 净利润(TTM)/总市值                                                 自行计算
# EPcut = 扣非后净利润(TTM)/总市值                                        自行计算
# BP = 净资产/总市值                                                      直接取值求倒数
# SP = 营业收入(TTM)/总市值                                               直接取值求倒数
# NCFP = 净现金流(TTM)/总市值                                             直接取值求倒数
# OCFP = 经营现金流(TTM)/总市值                                           直接取值求倒数
# FCFP = 自由现金流(最新年报）/总市值                                      自行计算
# DP = 近12个月现金红利(按除息日计)/总市值
# EV2EBITDA = 企业价值(扣除现金)/EBITDA(最新年报)                          自行计算
# PEG = (总市值)/净利润(TTM)) / 净利润(TTM)同比增长率                      自行计算


class Huatai_MF_2(FactorBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def job_check_ttm_availible(df, code):
        df = df.loc[df['code'] == code, :].copy().sort_values(['report_period'])
        report_periods = df['report_period']
        # 如果有上3个报告期的信息则是True,没有则是False
        df['check_ttm_availible'] = df['report_period'].apply(lambda x:
                True if {x + datetime.timedelta(days=1) - relativedelta(months=3) - datetime.timedelta(days=1),
                         x + datetime.timedelta(days=1) - relativedelta(months=6) - datetime.timedelta(days=1),
                         x + datetime.timedelta(days=1) - relativedelta(months=9) - datetime.timedelta(days=1)}
                        .issubset(set(report_periods)) else False)
        df['profit_ddt_ttm'] = df['profit_ddt_q'].rolling(4).sum()
        df = df.loc[df['check_ttm_availible']==True, ['code','date','report_period','profit_ddt_ttm']]
        return df


    def job_factors(self, df, code, start, end):
        df = df.loc[df['code'] == code, :].copy()
        df[['mv', 'profit_ddt_ttm', 'report_period', 'FCFF', 'EBITDA', 'total_cash', 'total_liab']] = \
            df[['mv','profit_ddt_ttm','report_period','FCFF','EBITDA','total_cash','total_liab']].fillna(method='ffill')

        df['EP_TTM'] = df['profit_ttm'] / df['mv'] / 10000
        df['EPcut_TTM'] = df['profit_ddt_ttm'] / df['mv'] / 10000
        df['BP'] = 1 / df['PB_TTM']
        df['SP'] = 1 / df['PS_TTM']
        df['NCFP'] = 1 / df['PNCF']
        df['OCFP'] = 1 / df['POCF']
        df['FCFP'] = df['FCFF'] / df['mv'] / 10000
        df['EV2EBITDA'] = (df['mv']*10000 - df['total_cash'] + df['total_liab'])/df['EBITDA']
        profit_ttm = pd.DataFrame(df.groupby(['report_period'])['profit_ttm'].first())
        if not profit_ttm.empty:
            profit_ttm.reset_index(inplace=True)
            last_Y_profit_ttm = profit_ttm.copy()
            last_Y_profit_ttm['report_period'] = last_Y_profit_ttm['report_period'].apply(lambda x:x+relativedelta(years=1))
            last_Y_profit_ttm.columns = ['report_period','last_Y_profit_ttm']
            profit_ttm = pd.merge(profit_ttm,last_Y_profit_ttm,left_on='report_period',right_on='report_period',how='left')
            profit_ttm['profit_TTM_YOY'] = profit_ttm.apply(lambda row:(row['profit_ttm']-row['last_Y_profit_ttm'])/
                                                                       abs(row['last_Y_profit_ttm']),axis=1)
            profit_ttm = profit_ttm.loc[:,['report_period','profit_TTM_YOY']]
            df = pd.merge(df,profit_ttm,left_on='report_period',right_on='report_period',how='left')
            df['PEG'] = df['mv']/df['profit_ttm']/df['profit_TTM_YOY']
        else:
            df['PEG'] = None
            df['profit_TTM_YOY'] = None
        df.set_index('date',inplace=True)
        df = df.loc[pd.notnull(df['report_period'])|pd.notnull(df['EP_TTM'])|pd.notnull(df['EPcut_TTM'])|
                    pd.notnull(df['BP'])|pd.notnull(df['SP'])|pd.notnull(df['NCFP'])|pd.notnull(df['OCFP'])|
                    pd.notnull(df['FCFP'])|pd.notnull(df['PEG'])|pd.notnull(df['profit_TTM_YOY']),:]
        df = df.loc[str(start):str(end), ['code', 'report_period', 'EP_TTM', 'EPcut_TTM', 'BP', 'SP', 'NCFP',
                                          'OCFP', 'FCFP','PEG', 'profit_TTM_YOY']]
        df['report_period'] = df['report_period'].apply(lambda x:x.strftime('%Y%m%d') if pd.notnull(x) else np.nan)
        df = df.where(pd.notnull(df), None)
        self.influx.saveData(df,'DailyFactor_Gus','Value')
        print('code: %s' %code)
        return


    def cal_factor(self,start,end):
        # 表:AShareFinancialIndicator
        # 字段:S_INFO_WINDCODE, ANN_DT, REPORT_PERIOD, S_QFA_DEDUCTEDPROFIT, S_FA_FCFF, S_FA_EBITDA
        # 起始时间往前推2年防止ttm为nan
        query = "select s_info_windcode, ann_dt, report_period, S_QFA_DEDUCTEDPROFIT, S_FA_FCFF, S_FA_EBITDA " \
                "from wind_filesync.AShareFinancialIndicator " \
                "where ann_dt >= {0} and ann_dt <= {1} " \
                "and (s_info_windcode like '0%' " \
                "or s_info_windcode like '3%' or s_info_windcode like '6%') "\
                "order by report_period"\
            .format((dtparser.parse(str(start))-relativedelta(years=2)).strftime('%Y%m%d'),str(end))
        self.rdf.curs.execute(query)
        FinIndi = pd.DataFrame(self.rdf.curs.fetchall(),
                               columns=['code','date','report_period','profit_ddt_q','FCFF','EBITDA'])
        FinIndi['report_period'] = pd.to_datetime(FinIndi['report_period'])
        FinIndi['date'] = pd.to_datetime(FinIndi['date'])
        profit_ddt = FinIndi.loc[:,['code','date','report_period','profit_ddt_q']].copy()
        profit_ddt.dropna(inplace=True)
        FinIndi = FinIndi.loc[:,['code','date','report_period','FCFF','EBITDA']]
        codes = list(profit_ddt['code'].unique())
        # 计算profit_ddt_ttm
        result_list = joblib.Parallel()(joblib.delayed(Huatai_MF_2.job_check_ttm_availible)
                                        (profit_ddt, code) for code in codes)
        profit_ddt = pd.concat(result_list)
        FinIndi = pd.merge(FinIndi,profit_ddt,how='left',left_on=['code','date','report_period'],
                           right_on=['code','date','report_period'])
        # 同一天可能有季报和年报
        FinIndi = FinIndi.groupby(['date','code']).last()
        FinIndi.reset_index(inplace=True)
        print('FinIndi data got!')

        # 表:AShareEODDerivativeIndicator
        # 字段:S_INFO_WINDCODE, TRADE_DT, S_VAL_MV, S_VAL_PE_TTM, S_VAL_PB_NEW, S_VAL_PS_TTM, S_VAL_PCF_NCFTTM,
        #       S_VAL_PCF_OCFTTM
        # 起始时间往前推2年防止ttm为nan
        query = "select s_info_windcode, TRADE_DT, S_VAL_PE_TTM, S_VAL_MV, NET_PROFIT_PARENT_COMP_TTM, " \
                "S_VAL_PB_NEW, S_VAL_PS_TTM, S_VAL_PCF_NCFTTM, S_VAL_PCF_OCFTTM " \
                "from wind_filesync.AShareEODDerivativeIndicator " \
                "where trade_dt >= {0} and trade_dt <= {1} " \
                "and (s_info_windcode like '0%' " \
                "or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "order by trade_dt"\
            .format((dtparser.parse(str(start))-relativedelta(years=2)).strftime('%Y%m%d'),str(end))
        self.rdf.curs.execute(query)
        EODDerIndi = pd.DataFrame(self.rdf.curs.fetchall(),columns=['code','date','PE_TTM','mv','profit_ttm',
                'PB_TTM','PS_TTM','PNCF','POCF'])
        EODDerIndi['date'] = pd.to_datetime(EODDerIndi['date'])
        print('EODDerIndi data got!')

        # 表:AShareCashFlow
        # 字段:S_INFO_WINDCODE, ANN_DT, CASH_CASH_EQU_END_PERIOD
        # 起始时间往前推1年
        query = "select s_info_windcode, ANN_DT, CASH_CASH_EQU_END_PERIOD " \
                "from wind_filesync.AShareCashFlow " \
                "where ANN_DT >= {0} and ANN_DT <= {1} " \
                "and (s_info_windcode like '0%' " \
                "or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "order by ANN_DT" \
            .format((dtparser.parse(str(start)) - relativedelta(years=1)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        CashFlow = pd.DataFrame(self.rdf.curs.fetchall(),columns=['code','date','total_cash'])
        CashFlow['date'] = pd.to_datetime(CashFlow['date'])
        CashFlow = CashFlow.groupby(['date', 'code']).last()
        CashFlow.reset_index(inplace=True)
        print('CashFlow data got!')

        # 表:AShareBalanceSheet
        # 字段:S_INFO_WINDCODE, ANN_DT, TOT_LIAB
        # 起始时间往前推1年
        query = "select s_info_windcode, ANN_DT, TOT_LIAB " \
                "from wind_filesync.AShareBalanceSheet " \
                "where ANN_DT >= {0} and ANN_DT <= {1} " \
                "and (s_info_windcode like '0%' " \
                "or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "order by ANN_DT" \
            .format((dtparser.parse(str(start)) - relativedelta(years=1)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        BalanceSheet = pd.DataFrame(self.rdf.curs.fetchall(), columns=['code', 'date', 'total_liab'])
        BalanceSheet['date'] = pd.to_datetime(BalanceSheet['date'])
        BalanceSheet = BalanceSheet.groupby(['date', 'code']).last()
        BalanceSheet.reset_index(inplace=True)
        print('BalanceSheet data got!')


        factors = pd.merge(EODDerIndi,FinIndi,how='outer',left_on=['date','code'],right_on=['date','code'])
        factors = pd.merge(factors,CashFlow,how='outer',left_on=['date','code'],right_on=['date','code'])
        factors = pd.merge(factors,BalanceSheet,how='outer',left_on=['date','code'],right_on=['date','code'])
        #codes= ['300513.SZ','300512.SZ','002800.SZ','002525.SZ']
        codes = factors['code'].unique()
        joblib.Parallel()(joblib.delayed(self.job_factors)(factors, code, start, end)
                                        for code in codes)
        print('factors process finish!')



if __name__ == '__main__':
    aaa = Huatai_MF_2()
    print(datetime.datetime.now())
    aaa.cal_factor(20100101,20190901)
    print(datetime.datetime.now())