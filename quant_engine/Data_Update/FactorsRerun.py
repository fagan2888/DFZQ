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
                 root_dir + '\\Factor\\Momentum', root_dir + '\\Factor\\Analyst'])
# -------------------------------
from rdf_data import rdf_data
import logging
from global_constant import N_JOBS
from EPandEPcut import EPandEPcut
from BP import BP
from SP import SP
from ROE import ROE_series
from RNOA import RNOA_series
from ROE_growth import ROE_growth
from RNOA_growth import RNOA_growth
from Surprise import Surprise


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
        roe = ROE_series()
        res = roe.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('---------------------ROE finish-----------------------')
        rnoa = RNOA_series()
        res = rnoa.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('--------------------RNOA finish-----------------------')
        rg = ROE_growth()
        res = rg.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('-----------------ROE growth finish--------------------')
        rnoag = RNOA_growth()
        res = rnoag.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('-----------------RNOA growth finish-------------------')
        surp = Surprise()
        res = surp.cal_factors(start, end, n_jobs)
        self.log_res(res)
        self.logger.info('------------------Surprise finish---------------------')

if __name__ == '__main__':
    du = FactorsRerun()
    du.run(20100101, 20200407, N_JOBS)