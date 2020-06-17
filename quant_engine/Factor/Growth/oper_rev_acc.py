from influxdb_data import influxdbData
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from factor_base import FactorBase
from joblib import Parallel, delayed, parallel_backend
import global_constant
import datetime


class NetProfitAcc(FactorBase):
    def __init__(self):
        super().__init__()
        self.db = 'DailyFactors_Gus'
        self.measure = 'oper_rev_Q_acc'

    @staticmethod
    def JOB_factors(codes, df, factor, db, measure):
        influx = influxdbData()
        save_res = []
        cols = []
        periods = []
        for i in range(1, 9):
            cols.append('{0}_last{1}Q'.format(factor, i - 1))
            periods.append(-i + 8)
        quadratic_featurizer = PolynomialFeatures(degree=2)
        regression_model = LinearRegression()
        for code in codes:
            code_df = df.loc[df['code'] == code, :].copy()
            code_df = code_df.loc[np.all(pd.notnull(code_df[cols]), axis=1), ['code', 'report_period'] + cols]
            if code_df.empty:
                continue
            x = np.array(periods)[:, np.newaxis]
            x = quadratic_featurizer.fit_transform(x)
            drop_dup_df = code_df.drop_duplicates(cols, 'first')
            drop_dup_values = drop_dup_df[cols].values
            acc = []
            for i in range(drop_dup_values.shape[0]):
                y = drop_dup_values[i, :]
                y = y[:, np.newaxis]
                regression_model.fit(x, y)
                acc.append(regression_model.coef_[0, 2])
            acc_dict = dict(zip(drop_dup_df.index, acc))
            code_df['oper_rev_Q_acc'] = code_df.index
            code_df['oper_rev_Q_acc'] = code_df['oper_rev_Q_acc'].map(acc_dict)
            code_df = code_df.replace(np.inf, np.nan)
            code_df = code_df.replace(-np.inf, np.nan)
            code_df['oper_rev_Q_acc'] = \
                code_df.groupby(['code', 'report_period'])['oper_rev_Q_acc'].fillna(method='ffill')
            code_df = code_df.loc[:, ['code', 'report_period', 'oper_rev_Q_acc']].dropna()
            print('code: %s' % code)
            r = influx.saveData(code_df, db, measure)
            if r == 'No error occurred...':
                pass
            else:
                save_res.append('%s Error: %s' % (measure, r))
        return save_res

    def cal_factors(self, start, end, n_jobs):
        fail_list = []
        # 计算 oper_rev_Q 的 加速度
        oper_rev = self.influx.getDataMultiprocess('FinancialReport_Gus', 'oper_rev_Q', start, end)
        oper_rev.index.names = ['date']
        oper_rev.rename(columns={'oper_rev_Q': 'oper_rev_Q_last0Q'}, inplace=True)
        # 归一化
        for i in range(8):
            oper_rev['oper_rev_Q_last{0}Q'.format(i)] = \
                oper_rev['oper_rev_Q_last{0}Q'.format(i)] / oper_rev['oper_rev_Q_last7Q']
        codes = oper_rev['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(NetProfitAcc.JOB_factors)
                             (codes, oper_rev, 'oper_rev_Q', self.db, self.measure) for codes in split_codes)
        print('oper_rev_Q_acc finish')
        print('-' * 30)
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    start_dt = datetime.datetime.now()
    acc = NetProfitAcc()
    r = acc.cal_factors(20100101, 20200616, global_constant.N_JOBS)
    print(r)
    print('time token:', datetime.datetime.now() - start_dt)
