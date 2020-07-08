from global_constant import FACTOR_DB, REPORT_DB

STRATEGY_CONFIG = \
    {
        'start': 20100101,
        'end': 20200701,
        'benchmark': 300,
        'select_range': 800,
        'industry': 'improved_lv1',
        'capital': 100000000,
        'adj_interval': 5,
        'lambda': 25
    }


# 大类因子权重
CATEGORY_WEIGHT = \
    {
        'VALUE': 1,
        'FIN_QUAL': 1,
        'GROWTH': 1,
        'BANK': 1
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
            [FACTOR_DB, 'net_profit_Q_growth', 'net_profit_Q_growthY', 1, 'median', 1, True, False]],
        'BANK': [
            [REPORT_DB, 'provision_cov', 'provision_cov', 1, 'median', 1, True, False]]
    }