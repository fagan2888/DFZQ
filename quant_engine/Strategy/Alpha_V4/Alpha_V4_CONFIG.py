from global_constant import FACTOR_DB, REPORT_DB

STRATEGY_CONFIG = \
    {
        'start': 20130101,
        'end': 20181231,
        'benchmark': 300,
        'select_range': 800,
        'industry': 'improved_lv1',
        'capital': 100000000,
        'adj_interval': 5,
        'lambda': 25,
        'non_fin_const': 2,
        'non_fin_ratio': 1,
        'bank_const': 20,
        'bank_ratio': 1,
        'sec_const': 20,
        'sec_ratio': 1
    }

# 银行大类
CATEGORY_BANK = \
    {
        'VALUE': 1,
        'FIN_QUAL': 1,
        'GROWTH': 1,
        'BANK': 1
    }

# 银行因子
FACTOR_BANK = \
    {
        'VALUE': [
            [FACTOR_DB, 'BP', 'BP', 1, 'median', 1, True, False],
            [FACTOR_DB, 'EP_FY1', 'EP_FY1', 1, 'median', 1, True, False]],
        'FIN_QUAL': [
            [FACTOR_DB, 'ROE2', 'ROE', 1, 'median', 1, True, False]],
        'GROWTH': [
            [FACTOR_DB, 'net_profit_Q_growth', 'net_profit_Q_growthY', 1, 'median', 1, True, False]],
        'BANK': [
            [REPORT_DB, 'provision_cov', 'provision_cov', 1, None, 1, True, False]]
    }

# 证券大类
CATEGORY_SEC = \
    {
        'VALUE': 1
    }

# 证券因子
FACTOR_SEC = \
    {
        'VALUE': [
            [FACTOR_DB, 'BP_M', 'BP_M', 1, None, 1, True, False],
            [FACTOR_DB, 'EP_M_TTM', 'EP_M_TTM', 1, None, 1, True, False],
            [FACTOR_DB, 'SP_M_TTM', 'SP_M_TTM', 1, None, 1, True, False]]
    }

# 大类因子权重
CATEGORY_NON_FIN = \
    {
        'VALUE': 1,
        'FIN_QUAL': 1,
        'GROWTH': 1,
        'ANALYST': 1,
        'ILIQUIDITY': 1,
        'REVERSE': 1
    }

# 因子权重
# list内分别为db, measure, factor, direction, fillna, weight, check_rp, neutralize
FACTOR_NON_FIN = \
    {
        'VALUE': [
            [FACTOR_DB, 'BP', 'BP', 1, 'median', 1, True, True],
            [FACTOR_DB, 'EP', 'EP_TTM', 1, 'median', 1, True, True],
            [FACTOR_DB, 'SP', 'SP', 1, 'median', 1, True, True],
            [FACTOR_DB, 'OCFP', 'OCFP', 1, 'median', 1, True, True],
            [FACTOR_DB, 'DP_LYR', 'DP_LYR', 1, 'zero', 1, False, True],
            [FACTOR_DB, 'EP_FY1', 'EP_FY1', 1, 'median', 1, True, True]],
        'FIN_QUAL': [
            [FACTOR_DB, 'ROE2', 'ROE', 1, 'median', 1, True, True],
            [FACTOR_DB, 'RNOA2', 'RNOA', 1, 'median', 1, True, True],
            [FACTOR_DB, 'GPOA2', 'GPOA', 1, 'median', 1, True, True],
            [FACTOR_DB, 'CFROI2', 'CFROI', 1, 'median', 1, True, True],
            [FACTOR_DB, 'ROA2', 'ROA', 1, 'median', 1, True, True]],
        'GROWTH': [
            [FACTOR_DB, 'net_profit_Q_growth', 'net_profit_Q_growthY', 1, 'median', 1, True, True],
            [FACTOR_DB, 'oper_rev_Q_growth', 'oper_rev_Q_growthY', 1, 'median', 1, True, True],
            [FACTOR_DB, 'Surprise', 'sur_net_profit_Q_WD', 1, 'median', 1, True, True],
            [FACTOR_DB, 'Surprise', 'sur_net_profit_Q_WOD', 1, 'median', 1, True, True],
            [FACTOR_DB, 'Surprise', 'sur_oper_rev_Q_WD', 1, 'median', 1, True, True],
            [FACTOR_DB, 'Surprise', 'sur_oper_rev_Q_WOD', 1, 'median', 1, True, True]],
        'ANALYST': [
            [FACTOR_DB, 'Analyst', 'sqrt_anlst_cov', 1, 'zero', 1, False, False],
            [FACTOR_DB, 'Analyst', 'net_profit_divergence', -1, 'median', 1, False, False],
            [FACTOR_DB, 'PEG', 'PEG2', -1, 'median', 1, False, True]],
        'ILIQUIDITY': [
            [FACTOR_DB, 'Amihud', 'amihud_20', 1, 'zero', 1, False, True],
            [FACTOR_DB, 'ln_ma_turnover', 'ln_turnover_60', -1, 'median', 1, False, True]],
        'REVERSE': [
            [FACTOR_DB, 'CGO', 'CGO_60', -1, 'median', 1, False, True],
            [FACTOR_DB, 'MaxRet', 'max_return_60', -1, 'median', 1, False, True],
            [FACTOR_DB, 'PeriodRet', 'ret_20', -1, 'median', 1, False, True]]
    }