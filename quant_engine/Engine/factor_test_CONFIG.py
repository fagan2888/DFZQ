from global_constant import FACTOR_DB, REPORT_DB

TEST_CONFIG = \
    {
        'start': 20100101,
        'end': 20200701,
        'benchmark': 300,
        'select_range': 800,
        'industry': 'improved_lv1',
        'adj_interval': 5
    }

# 大类因子权重
CATEGORY_WEIGHT = \
    {
        'ANALYST': 1,
    }

# 因子权重
# list内分别为db, measure, factor, direction, fillna, weight, check_rp, neutralize
FACTOR_WEIGHT = \
    {
        'VALUE': [
            [FACTOR_DB, 'BP', 'BP', 1, 'median', 1, True],
            [FACTOR_DB, 'EP_FY1', 'EP_FY1', 1, 'median', 1, True]],
        'FIN_QUAL': [
            [FACTOR_DB, 'ROE2', 'ROE', 1, 'median', 1, True]],
        'GROWTH': [
            [FACTOR_DB, 'net_profit_Q_growth', 'net_profit_Q_growthY', 1, 'median', 1, True, True]],
        'ANALYST': [
            [FACTOR_DB, 'PEG', 'PEG2', -1, None, 1, False, True]]
    }