from gogoal_data import GoGoal_data
import pandas as pd
import numpy as np
from factor_base import FactorBase
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
import datetime
from global_constant import N_JOBS
from joblib import Parallel, delayed, parallel_backend
from influxdb_data import influxdbData


class coverage_and_divergence(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'Analyst'
        self.gogoal = GoGoal_data()

    # 计算机构覆盖数的工具函数
    @staticmethod
    def JOB_anlst_cover(ranges, df, db, measure):
        influx = influxdbData()
        save_res = []
        for range_start, range_end in ranges:
            range_df = df.loc[range_start:range_end, :].copy()
            range_df = range_df.drop_duplicates(subset=['code', 'organ_name'])
            if range_df.empty:
                continue
            cov = range_df.groupby('code')['organ_name'].count()
            cov.name = 'anlst_cov'
            cov = pd.DataFrame(cov)
            cov['date'] = dtparser.parse(range_end)
            cov['sqrt_anlst_cov'] = np.sqrt(cov['anlst_cov'])
            cov = cov.reset_index().set_index('date')
            print('Time range: %s - %s' % (range_start, range_end))
            r = influx.saveData(cov, db, measure)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('FACTOR: %s \n range: %s - %s \n Error: %s' %
                                ('anlst_cover', range_start, range_end, r))
        return save_res

    # 计算机构净利润预测的分歧度
    @staticmethod
    def JOB_net_profit_divergence(ranges, df, db, measure):
        influx = influxdbData()
        save_res = []
        for range_start, range_end in ranges:
            range_df = df.loc[range_start:range_end, :].copy()
            # 只取对应当年的预测
            range_df = range_df.loc[range_df['year'] == int(range_end[:4]), :]
            # 某券商在6个月期间对某code有多篇报告的情况下，取最后一篇
            range_df = range_df.groupby(['code', 'organ_name'])['net_profit'].last().reset_index()
            # 取大于5家机构覆盖的code计算分歧度
            organ_count = range_df.groupby('code')['organ_name'].count()
            organ_count = organ_count[organ_count >= 5]
            range_df = range_df.loc[range_df['code'].isin(organ_count.index), :]
            if range_df.empty:
                continue
            mean = range_df.groupby('code')['net_profit'].mean()
            std = range_df.groupby('code')['net_profit'].std()
            divergence = std / abs(mean)
            divergence.name = 'net_profit_divergence'
            divergence = pd.DataFrame(divergence)
            divergence['date'] = dtparser.parse(range_end)
            divergence = divergence.reset_index().set_index('date')
            divergence = divergence.replace(np.inf, np.nan)
            divergence = divergence.dropna()
            if divergence.empty:
                continue
            print('Time range: %s - %s' % (range_start, range_end))
            r = influx.saveData(divergence, db, measure)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('FACTOR: %s \n range: %s - %s \n Error: %s' %
                                ('net_profit_div', range_start, range_end, r))
        return save_res

    def cal_anlst_cover(self, data, start, end):
        time_ranges = []
        dt_start = dtparser.parse(str(start))
        dt_end = dtparser.parse(str(end))
        dt_select = dt_start
        # 每次取6个月的区间
        while dt_select <= dt_end:
            time_ranges.append([(dt_select - relativedelta(months=6)).strftime('%Y%m%d'), dt_select.strftime('%Y%m%d')])
            dt_select += datetime.timedelta(days=1)
        split_ranges = np.array_split(time_ranges, self.n_jobs)
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            res = Parallel()(delayed(coverage_and_divergence.JOB_anlst_cover)
                             (ranges, data, self.db, self.measure) for ranges in split_ranges)
        print('FACTOR: anlst_cov FINISH!')
        print('-' * 30)
        for r in res:
            self.fail_list.extend(r)

    def cal_net_profit_divergence(self, data, start, end):
        time_ranges = []
        dt_start = dtparser.parse(str(start))
        dt_end = dtparser.parse(str(end))
        dt_select = dt_start
        # 每次取6个月的区间
        while dt_select <= dt_end:
            time_ranges.append([(dt_select - relativedelta(months=6)).strftime('%Y%m%d'), dt_select.strftime('%Y%m%d')])
            dt_select += datetime.timedelta(days=1)
        split_ranges = np.array_split(time_ranges, self.n_jobs)
        # 去掉没有预测net_profit的数据
        net_profit = data.dropna(subset=['net_profit'])
        # 只需要年报的预测数据（半年报的预测数据相对较少，一般不满足至少有5条数据的限制）
        net_profit = net_profit.loc[net_profit['quarter'] == 4, ['code', 'organ_name', 'year', 'net_profit']]
        # 原数据单位为万元
        net_profit['net_profit'] = net_profit['net_profit'] * 10000
        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            res = Parallel()(delayed(coverage_and_divergence.JOB_net_profit_divergence)
                             (ranges, net_profit, self.db, self.measure) for ranges in split_ranges)
        print('FACTOR: net_profit_divergence FINISH!')
        print('-' * 30)
        for r in res:
            self.fail_list.extend(r)

    def cal_factors(self, start, end, n_jobs):
        self.n_jobs = n_jobs
        self.fail_list = []
        query = "SELECT [A].[ORIGIN_ID], [A].[CODE], [A].[ORGAN_ID], [A].[SCORE_ID], [A].[CREATE_DATE], " \
                "[A].[INTO_DATE], [A].[TEXT5], " \
                "[B].[TIME_YEAR], [B].[QUARTER], [B].[FORECAST_INCOME], [B].[FORECAST_PROFIT], " \
                "[B].[FORECAST_INCOME_SHARE], [B].[FORECAST_RETURN_CASH_SHARE], " \
                "[B].[FORECAST_RETURN_CAPITAL_SHARE], [B].[FORECAST_RETURN], [B].[R_TAR3], " \
                "[C].[ORG_NAME] " \
                "FROM [{0}].[dbo].[CMB_REPORT_RESEARCH] [A] " \
                "LEFT JOIN [{0}].[dbo].[CMB_REPORT_SUBTABLE] [B] " \
                "on [A].[ID] = [B].[REPORT_SEARCH_ID] " \
                "LEFT JOIN [{0}].[dbo].[GG_ORG_LIST] [C] " \
                "on [A].[ORGAN_ID] = [C].[ID] " \
                "WHERE [A].[INTO_DATE] >= '{1}' and [A].[INTO_DATE] <= '{2}' " \
                "and [A].[INTO_DATE] - [A].[CREATE_DATE] <= 7 " \
            .format(self.gogoal.database,
                    (dtparser.parse(str(start)) - relativedelta(months=6)).strftime('%Y%m%d'), end)
        self.gogoal.cur.execute(query)
        data = pd.DataFrame(self.gogoal.cur.fetchall(),
                            columns=['report_id', 'code', 'organ_id', 'score', 'creat_date', 'into_date',
                                     'target_price', 'year', 'quarter', 'oper_rev', 'net_profit',
                                     'EPS', 'dvd_per_share', 'ROE', 'oper_income', 'EV/EBITDA', 'organ_name'])
        # 筛选出A股
        data['if_A'] = np.where((data['code'].str.len() == 6) &
                                ((data['code'].str[0] == '0') | (data['code'].str[0] == '3') |
                                 (data['code'].str[0] == '6')), 1, 0)
        data = data.loc[data['if_A'] == 1, :]
        data = data.dropna(subset=['year', 'quarter'])
        # 目标价，评级，净利润 只要有一个不为空，即纳入数据
        data = data.loc[pd.notnull(data['target_price']) |
                        pd.notnull(data['score']) | pd.notnull(data['net_profit']), :]
        # 朝阳永续的code不带exchange，修改统一格式
        data['code'] = np.where(data['code'].str[0] == '6', data['code'] + '.SH', data['code'] + '.SZ')
        data.set_index('into_date', inplace=True)
        data = data.sort_index()
        # 计算机构覆盖率
        self.cal_anlst_cover(data, start, end)
        # 计算净利润预测分歧率
        self.cal_net_profit_divergence(data, start, end)

        return self.fail_list


if __name__ == '__main__':
    time_start = datetime.datetime.now()
    af = coverage_and_divergence()
    f = af.cal_factors(20100101, 20200331, N_JOBS)
    print(f)
    time_end = datetime.datetime.now()
    print('Time token:', time_end-time_start)
