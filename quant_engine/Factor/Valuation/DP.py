from factor_base import FactorBase
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from influxdb_data import influxdbData
from dateutil.relativedelta import relativedelta
import datetime
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS

# DP_LYR = 最近年度分红汇总（税前）/股票市值
# 其中，若年报或者分红方案已经公布时，分子取上年度预案分红金额，若年报或分红方案均未公布时，则取上上年度的分红。

class DP(FactorBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def JOB_factors(df, codes, calendar, start, end):
        influx = influxdbData()
        save_res = []
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            dp_dict = dict(zip(code_df['DP_year'].values, code_df['DP_LYR'].values))
            blank_df = pd.DataFrame({'code': [code] * calendar.shape[0], 'date': calendar})
            code_df = pd.merge(code_df, blank_df, on=['date', 'code'], how='outer')
            code_df = code_df.set_index('date').sort_index()
            code_df['cur_month'] = code_df.index.strftime('%m').astype('int')
            code_df['withhold_year'] = code_df.index.strftime('%Y').astype('int') - 2
            # 大部分分红都在8月前，8月起 withhold_year 向前推1年
            code_df['withhold_year'] = np.where(code_df['cur_month'].values <= 7, code_df['withhold_year'].values,
                                                code_df['withhold_year'].values + 1)
            code_df['DP_year'] = code_df['DP_year'].fillna(method='ffill')
            code_df['DP_year'] = code_df[['DP_year', 'withhold_year']].max(axis=1)
            code_df['DP_LYR'] = code_df['DP_year'].map(dp_dict)
            code_df['DP_LYR'] = code_df['DP_LYR'].fillna(0)
            code_df = code_df.loc[str(start):str(end), ['code', 'DP_LYR']]
            print('code: %s' % code)
            r = influx.saveData(code_df, 'DailyFactors_Gus', 'DP_LYR')
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('DP_LYR Error: %s' % r)
        return save_res

    def cal_factors(self, start, end, n_jobs):
        # 获取分红信息
        # 日期使用分红确认的公告日
        query = "select S_INFO_WINDCODE, CASH_DVD_PER_SH_PRE_TAX, DVD_ANN_DT, REPORT_PERIOD " \
                "from wind_filesync.AShareDividend " \
                "where S_DIV_PROGRESS = 3 " \
                "and S_DIV_PRELANDATE >= {0} and S_DIV_PRELANDATE <= {1}" \
                "and (s_info_windcode like '0%' " \
                "or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "order by report_period" \
            .format((dtparser.parse(str(start)) - relativedelta(years=2)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        dvd = pd.DataFrame(self.rdf.curs.fetchall(), columns=['code', 'dvd_per_share', 'date', 'report_period'])
        dvd = dvd.dropna(subset=['date'])
        dvd['date'] = pd.to_datetime(dvd['date'])
        dvd['report_period'] = pd.to_datetime(dvd['report_period'])
        # 年报出分红才算分红完毕，其余report_period的日期都归到0630
        dvd['report_period'] = dvd['report_period'].apply(
            lambda x: x if (x.month == 6) & (x.day == 30) or (x.month == 12) & (x.day == 31)
            else datetime.datetime(x.year, 6, 30))
        # DP_year 是指按照那年来取DP的值，report_year 是指当前的分红归于哪一年
        dvd['DP_year'] = dvd['report_period'].apply(lambda x: x.year if x.month == 12 else x.year-1)
        dvd['report_year'] = dvd['report_period'].apply(lambda x: x.year)
        # 获取市值和股本信息
        # 股本单位： 万股， 市值单位： 万元
        query = "select TRADE_DT, S_INFO_WINDCODE, S_VAL_MV, TOT_SHR_TODAY " \
                "from wind_filesync.AShareEODDerivativeIndicator " \
                "where TRADE_DT >= {0} and TRADE_DT <= {1} " \
                "and (s_info_windcode like '0%' " \
                "or s_info_windcode like '3%' or s_info_windcode like '6%') " \
                "order by TRADE_DT" \
            .format((dtparser.parse(str(start)) - relativedelta(years=2)).strftime('%Y%m%d'), str(end))
        self.rdf.curs.execute(query)
        mv = pd.DataFrame(self.rdf.curs.fetchall(), columns=['date', 'code', 'mv', 'shares'])
        mv['mv'] = mv.groupby('code')['mv'].fillna(method='ffill')
        mv['shares'] = mv.groupby('code')['shares'].fillna(method='ffill')
        mv['date'] = pd.to_datetime(mv['date'])
        merge = pd.merge(dvd, mv, on=['date', 'code'], how='inner')
        # 由于 date 是实施公告日，所以不用担心 shares 改变的问题
        merge['DP_ratio'] = merge['dvd_per_share'] * merge['shares'] / merge['mv']
        stat = merge.groupby(['code', 'report_year'])['DP_ratio'].sum()
        stat = pd.DataFrame(stat).reset_index()
        stat.rename(columns={'DP_ratio': 'DP_LYR', 'report_year': 'DP_year'}, inplace=True)
        merge = pd.merge(merge, stat, on=['code', 'DP_year'], how='left')
        merge = merge.loc[:, ['date', 'code', 'DP_year', 'DP_LYR']]
        calendar = self.rdf.get_trading_calendar()
        calendar = calendar[(calendar >= str(start)) & (calendar <= str(end))].values
        codes = merge['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(DP.JOB_factors)
                             (merge, codes, calendar, start, end) for codes in split_codes)
        print('DP finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list

if __name__ == '__main__':
    print(datetime.datetime.now())
    dp = DP()
    r = dp.cal_factors(20190101, 20200205, N_JOBS)
    print('task finish')
    print(r)
    print(datetime.datetime.now())