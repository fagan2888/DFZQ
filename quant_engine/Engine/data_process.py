import numpy as np

class DataProcess:
    @staticmethod
    def remove_outlier(series):
        median = np.median(series)
        MAD = np.median(abs(series - median))
        ceil = median + 3 * 1.4826 * MAD
        floor = median - 3 * 1.4826 * MAD
        series[series > ceil] = ceil
        series[series < floor] = floor
        return series

    @staticmethod
    def Z_standardize(series):
        mean = np.mean(series)
        std = np.std(series)
        if std == 0:
            series[series != 0] = 0
        else:
            series = (series-mean)/std
        return series

    @staticmethod
    def rank_standardize(series):
        rank = series.rank()
        mean = np.mean(rank)
        std = np.std(rank)
        print(rank)
        if std == 0:
            rank[rank != 0] = 0
        else:
            rank = (rank-mean)/std
        return rank

