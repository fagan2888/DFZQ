import numpy as np
from influxdb_data import influxdbData
from rdf_data import rdf_data
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
import datetime

class FactorBase:
    @staticmethod
    def remove_outlier(series):
        median = np.median(series)
        MAD = np.median(abs(series - median))
        ceil = median + 3 * 1.4826 * MAD
        floor = median - 3 * 1.4826 * MAD
        series.loc[series > ceil] = ceil
        series.loc[series < floor] = floor
        return series

    @staticmethod
    def Z_standardize(series):
        mean = np.mean(series)
        std = np.std(series)
        if std == 0:
            series[series != 0] = 0
        else:
            series = (series - mean) / std
        return series

    @staticmethod
    def get_former_report_period(dt,Q_num):
        return dt+datetime.timedelta(days=1)-relativedelta(months=3*Q_num)-datetime.timedelta(days=1)

    @staticmethod
    def rank_standardize(series):
        rank = series.rank()
        mean = np.mean(rank)
        std = np.std(rank)
        if std == 0:
            rank[rank != 0] = 0
        else:
            rank = (rank - mean) / std
        return rank

    def __init__(self):
        self.influx = influxdbData()
        self.rdf = rdf_data()

    # 计算单季度指标所用函数(当季-上季)
    def cal_Q_data(self,data_input,report_period_input):
        if report_period_input[-4:] == '0331':
            ret_data = data_input
        else:
            last_report_period = dtparser.parse(report_period_input) + datetime.timedelta(days=1) - \
                                 relativedelta(months=3) - datetime.timedelta(days=1)
            try:
                if last_report_period in self.data_cache:
                    ret_data = data_input - self.data_cache[last_report_period]
                else:
                    ret_data = np.nan
            except AttributeError:
                ret_data = np.nan
        self.data_cache = {dtparser.parse(report_period_input): data_input}
        return ret_data

    def save_factor_to_influx(self,data,db,measure):
        self.influx.saveData(data,db,measure)

    @staticmethod
    def cal_growth(former_data,later_data):
        if former_data == 0:
            growth = np.nan
        else:
            growth = (later_data - former_data) / abs(former_data)
        return growth