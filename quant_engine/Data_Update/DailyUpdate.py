import sys
# 本地
root_dir = 'D:\\github\\quant_engine'
# 服务器
#root_dir = 'C:\\Users\\trader_9\\PycharmProjects\\DFZQ\\quant_engine'

sys.path.extend([root_dir, root_dir + '\\Data_Resource', root_dir + '\\Engine', root_dir + '\\Config',
                 root_dir + '\\Data_Update\\marketData', root_dir + '\\Data_Update\\Indicators',
                 root_dir + '\\Data_Update\\FinancialReport', root_dir + '\\Factor\\Size',
                 root_dir + '\\Factor\\Valuation', root_dir + '\\Factor\\Financial_Quality',
                 root_dir + '\\Factor\\Growth', root_dir + '\\Factor\\Turnover',
                 root_dir + '\\Factor\\Momentum', root_dir + '\\Factor\\Analyst',
                 root_dir + '\\Factor\\Risk', root_dir + '\\Factor\\Iliquidity',
                 root_dir + '\\Factor\\Reverse', root_dir + '\\Factor\\Banks'])
# -------------------------------
from rdf_data import rdf_data
import logging
import datetime
import dateutil.parser as dtparser
from dateutil.relativedelta import relativedelta
from global_constant import N_JOBS
from market_ohlc import mkt_ohlc
from is_ST import isst
from ex_right import ex_right
from stk_swap import stk_swap
from stk_industry import StkIndustry
from index_weight import index_weight
from shares_turnover import shares_turnover
from BalanceSheet import BalanceSheetUpdate
from Income import IncomeUpdate
from CashFlow import CashFlowUpdate
from QnTTM import QnTTMUpdate
from Banks import Banks
from loan_classification import LoanClassification
from market_cap import market_cap
from EPandEPcut import EPandEPcut
from BP import BP
from SP import SP
from DP import DP
from NCFP import NCFP
from OCFP import OCFP
from ROE2 import ROE2
from ROA2 import ROA2
from RNOA2 import RNOA2
from CFROI2 import CFROI2
from GPOA2 import GPOA2
from net_profit_growth import net_profit_growth
from oper_rev_growth import oper_rev_growth
from Surprise import Surprise
from amihud import Amihud
from ln_turnover_60 import LnTurnover
from MaxRet import MaxRet
from PeriodRet import PeriodRet
from CGO import CGO
from coverage_divergence import coverage_and_divergence
from consensus_net_profit import consensus_net_profit
from EP_FY1 import EP_FY1
from score_TPER import score_TPER
from PEG import PEG
from risk_exposure import RiskFactorsExposure
from risk_cov import RiskCov
from specific_risk import SpecificRisk
from NPL_leverage import NPL_leverage
from NPL_diff import NPL_diff
from provision_cov_growth import provision_cov_growth
from interest_income_growth import InterestIncome_growth


class DailyUpdate:
    def __init__(self):
        self.rdf = rdf_data()
        self.calendar = self.rdf.get_trading_calendar()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)
        self.log_file = root_dir + '\\Data_Update\\DailyUpdate.log'
        handler = logging.FileHandler(self.log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_res(self, res_list):
        for r in res_list:
            self.logger.info(r)

    def run(self, n_jobs, date=None):
        # 当date有输入时，为目标日期，当没有输入时，为每天执行，定时刷新上一天的数据
        if date:
            dt_today = dtparser.parse(str(date))
            dt_today = self.calendar[self.calendar > dt_today].iloc[0]
        else:
            dt_today = dtparser.parse(datetime.datetime.now().strftime('%Y%m%d'))
        if self.calendar[self.calendar == dt_today].empty:
            print('Not Trade Day...')
        else:
            dt_last_trade_day = self.calendar[self.calendar < dt_today].iloc[-1]
            dt_last_week = dt_last_trade_day - relativedelta(weeks=1)
            dt_last_1yr = dt_last_trade_day - relativedelta(years=1)
            last_trade_day = dt_last_trade_day.strftime('%Y%m%d')
            last_week = dt_last_week.strftime('%Y%m%d')
            last_1yr = dt_last_1yr.strftime('%Y%m%d')
            # ---------------------------------------------
            # 更新每日行情
            self.logger.info('//////////////////////////////////////////////////////')
            self.logger.info('Date:  %s' % (dt_today.strftime('%Y%m%d')))
            self.logger.info('BeginTime: %s' % datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'))
            self.logger.info('//////////////////////////////////////////////////////\n')

            # ---------------------------------------------
            self.logger.info('******************************************************')
            self.logger.info('===================基础数据日常更新=====================')
            self.logger.info('******************************************************')
            mkt = mkt_ohlc()
            res = mkt.process_data(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('------------------market data finish------------------')
            st = isst()
            res = st.process_data(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('-------------------st data finish---------------------')
            ex = ex_right()
            res = ex.process_data(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('-------------------ex right finish--------------------')
            swap = stk_swap()
            res = swap.process_data(last_1yr, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('-------------------swap data finish-------------------')
            indu = StkIndustry()
            res = indu.process_data(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('------------------stk industry finish-----------------')
            idx_weight = index_weight()
            res = idx_weight.process_data(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('------------------index weight finish-----------------')
            sh = shares_turnover()
            res = sh.process_data(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('-------------shares and turnover finish---------------')

            # ---------------------------------------------
            self.logger.info('******************************************************')
            self.logger.info('===================财报数据日常更新=====================')
            self.logger.info('******************************************************')
            bs = BalanceSheetUpdate()
            res = bs.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('----------------balance sheet finish------------------')
            income = IncomeUpdate()
            res = income.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('-------------------income finish----------------------')
            cf = CashFlowUpdate()
            res = cf.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('------------------cash flow finish--------------------')
            QnTTM = QnTTMUpdate()
            res = QnTTM.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('---------------------QnTTM finish---------------------')
            bank = Banks()
            res = bank.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('--------------------Banks finish----------------------')
            loan = LoanClassification()
            res = loan.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('---------------------Loan finish----------------------')

            # ---------------------------------------------
            self.logger.info('******************************************************')
            self.logger.info('===================因子数据日常更新=====================')
            self.logger.info('******************************************************')
            sz = market_cap()
            res = sz.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('------------------market_cap finish-------------------')
            risk_exp = RiskFactorsExposure()
            res = risk_exp.cal_factors(last_week, last_trade_day)
            self.log_res(res)
            self.logger.info('----------------risk exposure finish------------------')
            risk_cov = RiskCov()
            res = risk_cov.cal_factors(last_week, last_trade_day)
            self.log_res(res)
            self.logger.info('------------------risk cov finish---------------------')
            spec_risk = SpecificRisk()
            res = spec_risk.cal_factors(last_week, last_trade_day)
            self.log_res(res)
            self.logger.info('-----------------spec risk finish---------------------')
            ep = EPandEPcut()
            res = ep.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('---------------------EP finish------------------------')
            bp = BP()
            res = bp.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('---------------------BP finish------------------------')
            sp = SP()
            res = sp.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('---------------------SP finish------------------------')
            ncfp = NCFP()
            res = ncfp.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('--------------------NCFP finish-----------------------')
            ocfp = OCFP()
            res = ocfp.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('--------------------OCFP finish-----------------------')
            dp = DP()
            res = dp.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('--------------------DP_LYR finish---------------------')
            roe2 = ROE2()
            res = roe2.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('--------------------ROE2 finish-----------------------')
            rnoa = RNOA2()
            res = rnoa.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('--------------------RNOA finish-----------------------')
            roa = ROA2()
            res = roa.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('--------------------ROA2 finish-----------------------')
            cfroi = CFROI2()
            res = cfroi.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('--------------------CFROI finish----------------------')
            gpoa = GPOA2()
            res = gpoa.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('--------------------GPOA finish-----------------------')
            npg = net_profit_growth()
            res = npg.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('---------------net profit growth finish---------------')
            org = oper_rev_growth()
            res = org.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('----------------oper rev growth finish----------------')
            surp = Surprise()
            res = surp.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('------------------Surprise finish---------------------')
            amihud = Amihud()
            res = amihud.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('-------------------Amihud finish----------------------')
            ln_turnover = LnTurnover()
            res = ln_turnover.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('-----------------LnTurnover finish--------------------')
            maxret = MaxRet()
            res = maxret.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('-------------------MaxRet finish----------------------')
            pr = PeriodRet()
            res = pr.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('-----------------PeriodRet finish---------------------')
            cgo = CGO()
            res = cgo.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('--------------------CGO finish------------------------')
            cov_div = coverage_and_divergence()
            res = cov_div.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('---------------anlst cov div finish-------------------')
            cnp = consensus_net_profit()
            res = cnp.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('-----------consensus net profit finish----------------')
            epfy1 = EP_FY1()
            res = epfy1.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('--------------------EP_FY1 finish---------------------')
            score = score_TPER()
            res = score.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('----------------score and TPER finish-----------------')
            peg = PEG()
            res = peg.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('---------------------PEG finish-----------------------')
            npll = NPL_leverage()
            res = npll.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('-----------------NPL leverage finish------------------')
            npldiff = NPL_diff()
            res = npldiff.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('-------------------NPL diff finish--------------------')
            pcg = provision_cov_growth()
            res = pcg.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('-------------provision cov growth finish--------------')
            iig = InterestIncome_growth()
            res = iig.cal_factors(last_week, last_trade_day, n_jobs)
            self.log_res(res)
            self.logger.info('------------interest income growth finish-------------')

            self.logger.info('//////////////////////////////////////////////////////')
            self.logger.info('EndTime: %s' % datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'))
            self.logger.info('//////////////////////////////////////////////////////\n')

if __name__ == '__main__':
    du = DailyUpdate()
    du.run(N_JOBS)