# 策略的起始日，终止日，benchmark，选股范围，行业field，市值field
STRATEGY_CONFIG = \
    {
        'start': 20150101,
        'end': 20180101,
        'benchmark': 300,
        'select_range': 800,
        'industry': 'improved_lv1',
        'size_field': 'ln_market_cap',
        'select_pct': 0.6
    }

# 大类因子权重
CATEGORY_WEIGHT = \
    {
        'VALUE': 1,
        'FIN_QUAL': 1,
        'GROWTH': 1,
        'MOMENTUM': 1,
        'TURNOVER': 1
    }

# 因子权重
# list内分别为measure, factor, direction, if_fillna, weight
FACTOR_WEIGHT = \
    {
        'VALUE': [
            ['BP', 'BP', 1, True, 1],
            ['EP', 'EPcut_TTM', 1, True, 1],
            ['SP', 'SP', 1, True, 1],
            ['DP_LYR', 'DP_LYR', 1, True, 1]],
        'FIN_QUAL': [
            ['ROE', 'ROE_ddt', 1, False, 1]],
        'GROWTH': [
            ['ROE_growth', 'ROE_ddt_growthQ', 1, True, 1]],
        'MOMENTUM': [
            ['Momentum', 'free_exp_wgt_rtn_m1', -1, True, 1],
            ['Momentum', 'rtn_m1', -1, True, 1]],
        'TURNOVER': [
            ['Turnover', 'std_free_turn_1m', -1, False, 1],
            ['Turnover', 'bias_std_free_turn_1m', -1, False, 1]]
    }