STRATEGY_CONFIG = \
    {
        'start': 20180101,
        'end': 20200101,
        'benchmark': 300,
        'select_range': 800,
        'industry': 'improved_lv1',
        'capital': 100000000,
        'adj_interval': 5,
        'npl_quantile': 0.8,
        'npl_field': 'NPL',
        'n_selection': 3
    }


# 大类因子权重
CATEGORY_WEIGHT = \
    {
        'VALUE': 1,
        'FIN_QUAL': 1,
        'GROWTH': 1,
    }

# 因子权重
# list内分别为measure, factor, direction, if_fillna, style, weight
FACTOR_WEIGHT = \
    {
        'VALUE': [
            ['BP', 'BP', 1, 'median', 1],
            ['Analyst', 'EP_FY1', 1, 'median', 1],
            ['net_interest_margin', 'net_interest_margin', 1, 'median', 1]],
        'FIN_QUAL': [
            ['ROE', 'ROE', 1, 'median', 1],
            ['RNOA', 'RNOA', 1, 'median', 1],
            ['GPOA', 'GPOA', 1, 'median', 1],
            ['CFROI', 'CFROI', 1, 'median', 1],
            ['ROA', 'ROA', 1, 'median', 1]],
        'GROWTH': [
            ['net_profit_growth', 'net_profit_Q_growthY', 1, 'median', 1],
            ['oper_rev_growth', 'oper_rev_Q_growthY', 1, 'median', 1]]
    }