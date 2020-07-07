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
import datetime
import logging
from global_constant import N_JOBS
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


class FactorsRerun:
    def __init__(self):
        self.rdf = rdf_data()
        self.calendar = self.rdf.get_trading_calendar()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)
        self.log_file = root_dir + '\\Data_Update\\FactorsRerun.log'
        handler = logging.FileHandler(self.log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_res(self, res_list):
        for r in res_list:
            self.logger.info(r)

    def run(self, start, end, n_jobs):
        self.logger.info('FACTORS RERUN: \n Time period: %i ~ %i' % (start, end))
        '''
        sz = market_cap()
        res = sz.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('------------------market_cap finish-------------------')
        risk_exp = RiskFactorsExposure()
        res = risk_exp.cal_factors(start, end)
        self.log_res(res)
        self.logger.info('----------------risk exposure finish------------------')
        risk_cov = RiskCov()
        res = risk_cov.cal_factors(start, end)
        self.log_res(res)
        self.logger.info('------------------risk cov finish---------------------')
        spec_risk = SpecificRisk()
        res = spec_risk.cal_factors(start, end)
        self.log_res(res)
        self.logger.info('-----------------spec risk finish---------------------')
        ep = EPandEPcut()
        res = ep.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('---------------------EP finish------------------------')
        bp = BP()
        res = bp.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('---------------------BP finish------------------------')
        sp = SP()
        res = sp.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('---------------------SP finish------------------------')
        ncfp = NCFP()
        res = ncfp.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('--------------------NCFP finish-----------------------')
        ocfp = OCFP()
        res = ocfp.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('--------------------OCFP finish-----------------------')
        dp = DP()
        res = dp.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('--------------------DP_LYR finish---------------------')
        roe2 = ROE2()
        res = roe2.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('--------------------ROE2 finish-----------------------')
        rnoa = RNOA2()
        res = rnoa.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('--------------------RNOA finish-----------------------')
        roa = ROA2()
        res = roa.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('--------------------ROA2 finish-----------------------')
        cfroi = CFROI2()
        res = cfroi.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('--------------------CFROI finish----------------------')
        gpoa = GPOA2()
        res = gpoa.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('--------------------GPOA finish-----------------------')
        npg = net_profit_growth()
        res = npg.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('---------------net profit growth finish---------------')
        org = oper_rev_growth()
        res = org.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('----------------oper rev growth finish----------------')
        surp = Surprise()
        res = surp.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('------------------Surprise finish---------------------')
        amihud = Amihud()
        res = amihud.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('-------------------Amihud finish----------------------')
        ln_turnover = LnTurnover()
        res = ln_turnover.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('-----------------LnTurnover finish--------------------')
        maxret = MaxRet()
        res = maxret.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('-------------------MaxRet finish----------------------')
        pr = PeriodRet()
        res = pr.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('-----------------PeriodRet finish---------------------')
        cgo = CGO()
        res = cgo.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('--------------------CGO finish------------------------')
        '''
        cov_div = coverage_and_divergence()
        res = cov_div.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('---------------anlst cov div finish-------------------')
        cnp = consensus_net_profit()
        res = cnp.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('-----------consensus net profit finish----------------')
        epfy1 = EP_FY1()
        res = epfy1.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('--------------------EP_FY1 finish---------------------')
        score = score_TPER()
        res = score.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('----------------score and TPER finish-----------------')
        peg = PEG()
        res = peg.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('---------------------PEG finish-----------------------')
        '''
        npll = NPL_leverage()
        res = npll.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('-----------------NPL leverage finish------------------')
        npldiff = NPL_diff()
        res = npldiff.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('-------------------NPL diff finish--------------------')
        pcg = provision_cov_growth()
        res = pcg.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('-----------------provision cov finish-----------------')
        iig = InterestIncome_growth()
        res = iig.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('------------interest income growth finish-------------')
        '''

        self.logger.info('//////////////////////////////////////////////////////')
        self.logger.info('EndTime: %s' % datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'))
        self.logger.info('//////////////////////////////////////////////////////\n')

if __name__ == '__main__':
    du = FactorsRerun()
    du.run(20090101, 20200706, N_JOBS)