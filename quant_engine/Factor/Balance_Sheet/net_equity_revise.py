from factor_base import FactorBase
import pandas as pd
import numpy as np
import joblib

class net_equity_revise(FactorBase):
    def __init__(self):
        super().__init__()


    def get_former_data(self,series,Q_backward):
        report_period = net_equity_revise.get_former_report_period(series['report_period'], Q_backward)
        if report_period not in series.index:
            return np.nan
        else:
            return series[report_period]


    def job_factors(self,code):
        code_net_equity = self.net_equity.loc[self.net_equity['code']==code,:].copy()
        insert_dates = self.calendar - set(code_net_equity.index)
        content = [[np.nan] * len(self.columns)] * len(insert_dates)
        insert_df = pd.DataFrame(content, columns=self.columns, index=list(insert_dates))
        code_net_equity = code_net_equity.append(insert_df, ignore_index=False).sort_index()
        code_net_equity = code_net_equity.fillna(method='ffill')
        code_net_equity = code_net_equity.dropna(subset=['code'])
        code_net_equity['report_period'] = code_net_equity.apply(lambda row: row.dropna().index[-1], axis=1)
        code_net_equity['net_equity'] = code_net_equity.apply(lambda row: self.get_former_data(row,0), axis=1)
        code_net_equity['net_equity_last1Q'] = code_net_equity.apply(lambda row: self.get_former_data(row,1), axis=1)
        code_net_equity['net_equity_last2Q'] = code_net_equity.apply(lambda row: self.get_former_data(row,2), axis=1)
        code_net_equity['net_equity_last3Q'] = code_net_equity.apply(lambda row: self.get_former_data(row,3), axis=1)
        code_net_equity['net_equity_lastY']  = code_net_equity.apply(lambda row: self.get_former_data(row,4), axis=1)

        code_net_equity = code_net_equity.loc[self.start:self.end,['code','report_period','net_equity',
            'net_equity_last1Q','net_equity_last2Q','net_equity_last3Q','net_equity_lastY']]
        code_net_equity['report_period'] = code_net_equity['report_period'].apply(lambda x: x.strftime('%Y%m%d'))
        code_net_equity = code_net_equity.where(pd.notnull(code_net_equity), None)
        print('code: %s' %code)
        self.influx.saveData(code_net_equity, 'Financial_Report_Gus', 'net_equity')


    def cal_factors(self,start,end):
        self.calendar = self.rdf.get_trading_calendar()
        self.start = str(start)
        self.end = str(end)
        self.calendar = set(self.calendar.loc[(self.calendar >= self.start) & (self.calendar <= self.end)])
        self.net_equity = pd.read_hdf('D:/github/quant_engine/Data_Resource/Balance_Sheet/net_equity.h5', key='data')
        self.net_equity = self.net_equity.sort_values(by=['report_period', 'date'])
        self.net_equity['date'] = pd.to_datetime(self.net_equity['date'])
        self.net_equity['report_period'] = pd.to_datetime(self.net_equity['report_period'])
        self.net_equity.set_index(['code', 'date', 'report_period'], inplace=True)
        self.net_equity = self.net_equity.unstack(level=2)
        self.net_equity = self.net_equity.loc[:, 'net_equity']
        self.net_equity.reset_index(inplace=True)
        self.net_equity.set_index('date',inplace=True)
        self.columns = self.net_equity.columns

        joblib.Parallel()(joblib.delayed(self.job_factors)(code)
                          for code in self.net_equity['code'].unique())


if __name__ == '__main__':
    n = net_equity_revise()
    n.cal_factors(20100101,20190901)