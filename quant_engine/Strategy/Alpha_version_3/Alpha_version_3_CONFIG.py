# 策略的起始日，终止日，benchmark，选股范围，行业field，市值field，初始资金

STRATEGY_CONFIG = \
    {
        'start': 20130101,
        'end': 20200401,
        'benchmark': 300,
        'select_range': 800,
        'industry': 'improved_lv1',
        'capital': 100000000,
        'adj_interval': 5,
        'target_sigma': 0.04,
        # 市值主动最大暴露
        'mv_max_exp': 0.1,
        # 市值主动最小暴露
        'mv_min_exp': -0.1,
        'weight_intercept': 1,
        'n_codes': 120
    }


# 大类因子权重
CATEGORY_WEIGHT = \
    {
        'VALUE': 1,
        'FIN_QUAL': 1,
        'GROWTH': 1,
        'ANALYST': 1,
        'ILIQUIDITY': 1,
        'REVERSE': 1
    }

# 因子权重
# list内分别为measure, factor, direction, if_fillna, style, weight
FACTOR_WEIGHT = \
    {
        'VALUE': [
            ['BP', 'BP', 1, 'median', 1],
            ['EP', 'EP_TTM', 1, 'median', 1],
            ['SP', 'SP', 1, 'median', 1],
            ['OCFP', 'OCFP', 1, 'median', 1],
            ['DP_LYR', 'DP_LYR', 1, 'zero', 1],
            ['Analyst', 'EP_FY1', 1, 'median', 1]],
        'FIN_QUAL': [
            ['ROE', 'ROE', 1, 'median', 1],
            ['RNOA', 'RNOA', 1, 'median', 1],
            ['GPOA', 'GPOA', 1, 'median', 1],
            ['CFROI', 'CFROI', 1, 'median', 1],
            ['ROA', 'ROA', 1, 'median', 1]],
        'GROWTH': [
            ['net_profit_growth', 'net_profit_Q_growthY', 1, 'median', 1],
            ['oper_rev_growth', 'oper_rev_Q_growthY', 1, 'median', 1],
            ['Surprise', 'sur_net_profit_Q_WD', 1, 'median', 1],
            ['Surprise', 'sur_net_profit_Q_WOD', 1, 'median', 1],
            ['Surprise', 'sur_oper_rev_Q_WD', 1, 'median', 1],
            ['Surprise', 'sur_oper_rev_Q_WOD', 1, 'median', 1]],
        'ANALYST': [
            ['Analyst', 'sqrt_anlst_cov', 1, 'zero', 1],
            ['Analyst', 'net_profit_divergence', -1, 'median', 1],
            ['Analyst', 'score', 1, 'median', 1],
            ['Analyst', 'PEG', -1, 'median', 1]],
        'ILIQUIDITY': [
            ['Amihud', 'amihud_20', 1, 'zero', 1],
            ['ln_ma_turnover', 'ln_turnover_60', -1, 'median', 1]],
        'REVERSE': [
            ['CGO', 'CGO_60', -1, 'median', 1],
            ['MaxRet', 'max_return_60', -1, 'median', 1],
            ['PeriodRet', 'ret_20', -1, 'median', 1]]
    }