from global_constant import FACTOR_DB, REPORT_DB

STRATEGY_CONFIG = \
    {
        'start': 20100101,
        'end': 20181231,
        'benchmark': 300,
        'select_range': 800,
        'industry': 'improved_lv1',
        'capital': 100000000,
        'adj_interval': 1,
        'n_selection': 5
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
            [FACTOR_DB, 'SP_Q', 'SP_Q', 1, 'median', 1, True, False]],
        'FIN_QUAL': [
            [FACTOR_DB, 'ROE2', 'ROE', 1, 'median', 1, True, False]],
        'GROWTH': [
            [FACTOR_DB, 'net_profit_Q_growth', 'net_profit_Q_growthY', 1, 'median', 1, True, False],
            [FACTOR_DB, 'oper_rev_Q_acc', 'oper_rev_Q_acc', 1, 'median', 1, True, False]]
    }