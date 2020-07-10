from global_constant import FACTOR_DB, REPORT_DB

STRATEGY_CONFIG = \
    {
        'start': 20120101,
        'end': 20200702,
        'benchmark': 300,
        'select_range': 800,
        'industry': 'improved_lv1',
        'capital': 100000000,
        'adj_interval': 5,
        'lambda': 25,
        'size_limit': 0.1
    }


# 大类因子权重
CATEGORY_WEIGHT = \
    {
        'VALUE': 1,
    }

# 因子权重
# list内分别为db, measure, factor, direction, fillna, weight, check_rp, neutralize
FACTOR_WEIGHT = \
    {
        'VALUE': [
            [FACTOR_DB, 'BP_M', 'BP_M', 1, None, 1, True, False],
            [FACTOR_DB, 'EP_M_TTM', 'EP_M_TTM', 1, None, 1, True, False],
            [FACTOR_DB, 'SP_M_TTM', 'SP_M_TTM', 1, None, 1, True, False]]
    }