from factor_base import FactorBase
import pandas as pd
import numpy as np
import joblib

class net_profit_ddt_ttm(FactorBase):
    def __init__(self):
        super().__init__()


    def get_former_data(self,series,Q_backward):
        later_report_period = net_profit_ddt_ttm.get_former_report_period(series['report_period'],Q_backward)
        if later_report_period not in series.index:
            return np.nan
        else:
            if later_report_period.month == 3:
                return series[later_report_period]
            else:
                former_report_period = net_profit_ddt_ttm.get_former_report_period(later_report_period,1)
                if former_report_period not in series.index:
                    return np.nan
                else:
                    return series[later_report_period] - series[former_report_period]


    def job_factors(self,code):
        code_net_profit_ddt = self.net_profit_ddt.loc[self.net_profit_ddt['code'] == code, :].copy()
        insert_dates = self.calendar - set(code_net_profit_ddt.index)
        content = [[np.nan] * len(self.columns)] * len(insert_dates)
        insert_df = pd.DataFrame(content, columns=self.columns, index=list(insert_dates))
        code_net_profit_ddt = code_net_profit_ddt.append(insert_df, ignore_index=False).sort_index()
        code_net_profit_ddt = code_net_profit_ddt.fillna(method='ffill')
        code_net_profit_ddt = code_net_profit_ddt.dropna(subset=['code'])
        code_net_profit_ddt['report_period'] = code_net_profit_ddt.apply(lambda row: row.dropna().index[-1], axis=1)
        code_net_profit_ddt['net_profit_ddt'] = code_net_profit_ddt.apply(lambda row: row[row['report_period']], axis=1)

        code_net_profit_ddt['net_profit_ddt_Q'] = code_net_profit_ddt.apply(lambda row: self.get_former_data(row, 0), axis=1)
        code_net_profit_ddt['net_profit_ddt_last1Q'] = code_net_profit_ddt.apply(lambda row: self.get_former_data(row, 1), axis=1)
        code_net_profit_ddt['net_profit_ddt_last2Q'] = code_net_profit_ddt.apply(lambda row: self.get_former_data(row, 2), axis=1)
        code_net_profit_ddt['net_profit_ddt_last3Q'] = code_net_profit_ddt.apply(lambda row: self.get_former_data(row, 3), axis=1)
        code_net_profit_ddt['net_profit_ddt_lastY']  = code_net_profit_ddt.apply(lambda row: self.get_former_data(row, 4), axis=1)
        code_net_profit_ddt['net_profit_ddt_TTM'] = code_net_profit_ddt['net_profit_ddt_Q'] + code_net_profit_ddt['net_profit_ddt_last1Q'] + \
                                            code_net_profit_ddt['net_profit_ddt_last2Q'] + code_net_profit_ddt['net_profit_ddt_last3Q']

        code_net_profit_ddt = code_net_profit_ddt.loc[self.start:self.end, ['code', 'report_period', 'net_profit_ddt',
                                                                    'net_profit_ddt_Q','net_profit_ddt_last1Q', 'net_profit_ddt_last2Q',
                                                                    'net_profit_ddt_last3Q', 'net_profit_ddt_lastY','net_profit_ddt_TTM']]
        code_net_profit_ddt['report_period'] = code_net_profit_ddt['report_period'].apply(lambda x:x.strftime('%Y%m%d'))
        code_net_profit_ddt = code_net_profit_ddt.where(pd.notnull(code_net_profit_ddt), None)
        self.influx.saveData(code_net_profit_ddt, 'Financial_Report_Gus', 'net_profit_ddt')
        print('code: %s' %code)


    def cal_factors(self,start,end):
        self.calendar = self.rdf.get_trading_calendar()
        self.start = str(start)
        self.end = str(end)
        self.calendar = set(self.calendar.loc[(self.calendar>=self.start)&(self.calendar<=self.end)])
        self.net_profit_ddt = pd.read_hdf('D:/github/quant_engine/Data_Resource/Income/net_profit_ddt.h5', key='data')
        self.net_profit_ddt = self.net_profit_ddt.sort_values(by=['report_period','date'])
        self.net_profit_ddt['date'] = pd.to_datetime(self.net_profit_ddt['date'])
        self.net_profit_ddt['report_period'] = pd.to_datetime(self.net_profit_ddt['report_period'])
        self.net_profit_ddt.set_index(['code','date','report_period'],inplace=True)
        self.net_profit_ddt = self.net_profit_ddt.unstack(level=2)
        self.net_profit_ddt = self.net_profit_ddt.loc[:,'net_profit_ddt']
        self.net_profit_ddt.reset_index(inplace=True)
        self.net_profit_ddt.set_index('date',inplace=True)
        self.columns = self.net_profit_ddt.columns

        joblib.Parallel()(joblib.delayed(self.job_factors)(code)
                          for code in self.net_profit_ddt['code'].unique())


if __name__ == '__main__':
    n = net_profit_ddt_ttm()
    n.cal_factors(20100101,20190901)
