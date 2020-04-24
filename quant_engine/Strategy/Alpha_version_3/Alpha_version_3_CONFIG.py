# 策略的起始日，终止日，benchmark，选股范围，行业field，市值field，初始资金

STRATEGY_CONFIG = \
    {
        'start': 20140101,
        'end': 20181231,
        'benchmark': 300,
        'select_range': None,
        'industry': 'improved_lv1',
        'size_field': 'ln_market_cap',
        'capital': 100000000,
        'adj_interval': 5,
        'target_sigma': 0.05,
        # 市值主动最大暴露
        'mv_max_exp': 0.3,
        # 市值主动最小暴露
        'mv_min_exp': 0.2,
        'n_codes': 250
    }

CATEGORY_WEIGHT = \
    {
        'VALUE': 1}
FACTOR_WEIGHT = \
    {
        'VALUE': [
            ['EP', 'EP_TTM', 1, True, 1]]}






'''
# 大类因子权重
CATEGORY_WEIGHT = \
    {
        'VALUE': 1,
        'FIN_QUAL': 1,
        'GROWTH': 1,
        'MOMENTUM': 1,
        'TURNOVER': 1,
        'Analyst': 1
    }

# 因子权重
# list内分别为measure, factor, direction, if_fillna, weight
FACTOR_WEIGHT = \
    {
        'VALUE': [
            ['BP', 'BP', 1, True, 1],
            ['EP', 'EP_TTM', 1, True, 1],
            ['SP', 'SP', 1, True, 1],
            ['DP_LYR', 'DP_LYR', 1, True, 1],
            ['Analyst', 'EP_FY1', 1, True, 1]],
        'FIN_QUAL': [
            ['ROE', 'ROE', 1, False, 1],
            ['RNOA', 'RNOA', 1, False, 1]],
        'GROWTH': [
            ['ROE_growth', 'ROE_Q_growthY', 1, True, 1],
            ['Surprise', 'sur_net_profit_Q_WD', 1, True, 1],
            ['Surprise', 'sur_net_profit_Q_WOD', 1, True, 1],
            ['Surprise', 'sur_oper_rev_Q_WD', 1, True, 1],
            ['Surprise', 'sur_oper_rev_Q_WOD', 1, True, 1]],
        'MOMENTUM': [
            ['Momentum', 'free_exp_wgt_rtn_m1', -1, True, 1],
            ['Momentum', 'rtn_m1', -1, True, 1]],
        'TURNOVER': [
            ['Turnover', 'std_free_turn_1m', -1, True, 1],
            ['Turnover', 'bias_std_free_turn_1m', -1, True, 1]],
        'Analyst': [
            ['Analyst', 'anlst_cov', 1, False, 1],
            ['Analyst', 'net_profit_divergence', -1, True, 1],
            ['Analyst', 'score', 1, True, 1],
            ['Analyst', 'PEG', -1, True, 1]]
    }
'''
