# 存放期货相关参数
IH_MULTI = 300
IF_MULTI = 300
IC_MULTI = 200

IH_TICK = 0.2
IF_TICK = 0.2
IC_TICK = 0.2

IH_MARGIN_RATE = 0.1
IF_MARGIN_RATE = 0.1
IC_MARGIN_RATE = 0.12

IDX_FEE_RATE = 0.00003

MULTI_DICT  = {'IH':IH_MULTI,       'IF':IF_MULTI,       'IC':IC_MULTI}
MARGIN_DICT = {'IH':IH_MARGIN_RATE, 'IF':IF_MARGIN_RATE, 'IC':IC_MARGIN_RATE}
FEE_DICT    = {'IH':IDX_FEE_RATE,   'IF':IDX_FEE_RATE,   'IC':IDX_FEE_RATE}
TICK_DICT   = {'IH':IH_TICK,        'IF':IF_TICK,        'IC':IC_TICK}


# 取参工具
class FuturesTools:
    def __init__(self):
        pass
    @staticmethod
    def get_ftrs_multi(symbol):
        if symbol[0:2] not in MULTI_DICT:
            print('no %s multi-information in CONFIG!' %symbol[0:2])
        else:
            return MULTI_DICT[symbol[0:2]]
    @staticmethod
    def get_ftrs_margin(symbol):
        if symbol[0:2] not in MARGIN_DICT:
            print('no %s margin-information in CONFIG!' %symbol[0:2])
        else:
            return MARGIN_DICT[symbol[0:2]]
    @staticmethod
    def get_ftrs_fee(symbol):
        if symbol[0:2] not in FEE_DICT:
            print('no %s fee-information in CONFIG!' %symbol[0:2])
        else:
            return FEE_DICT[symbol[0:2]]
    @staticmethod
    def get_ftrs_tick(symbol):
        if symbol[0:2] not in TICK_DICT:
            print('no %s tick-information in CONFIG!' %symbol[0:2])
        else:
            return TICK_DICT[symbol[0:2]]