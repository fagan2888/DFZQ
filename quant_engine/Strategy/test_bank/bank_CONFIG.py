from global_constant import FACTOR_DB, REPORT_DB

STRATEGY_CONFIG = \
    {
        'start': 20100101,
        'end': 20181231,
        'benchmark': 300,
        'select_range': 800,
        'industry': 'improved_lv1',
        'capital': 100000000,
        'adj_interval': 5,
        'n_selection': 5
    }


# 大类因子权重
CATEGORY_WEIGHT = \
    {
        'GROWTH': 1,
    }

# 因子权重
# list内分别为db, measure, factor, direction, fillna, weight, check_rp, neutralize
FACTOR_WEIGHT = \
    {
        'VALUE': [
            [FACTOR_DB, 'BP', 'BP', 1, 'median', 1, True, False],
            [FACTOR_DB, 'EP_FY1', 'EP_FY1', 1, 'median', 1, True, False]],
        'FIN_QUAL': [
            [FACTOR_DB, 'ROE2', 'ROE', 1, 'median', 1, True, False]],
        'GROWTH': [
            [FACTOR_DB, 'cost_income_ratio_delta', 'cost_income_ratio_deltaQ', 1, 'median', 1, True, False]]
    }