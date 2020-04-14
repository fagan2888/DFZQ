# 回测所用日线数据维护
# 数据包含：个股行业数据

from influxdb_data import influxdbData
from rdf_data import rdf_data
import pandas as pd
import numpy as np
import datetime
from joblib import Parallel, delayed, parallel_backend
from global_constant import N_JOBS


class StkIndustry:
    def __init__(self):
        self.rdf = rdf_data()
        self.influx = influxdbData()
        self.db = 'DailyMarket_Gus'
        self.measure = 'industry'

    @staticmethod
    def JOB_get_indu(indu_type, start, end, calendar):
        sw_lv1_index = ['801010.SI', '801020.SI', '801030.SI', '801040.SI', '801050.SI', '801080.SI', '801110.SI',
                        '801120.SI', '801130.SI', '801140.SI', '801150.SI', '801160.SI', '801170.SI', '801180.SI',
                        '801200.SI', '801210.SI', '801230.SI', '801710.SI', '801720.SI', '801730.SI', '801740.SI',
                        '801750.SI', '801760.SI', '801770.SI', '801780.SI', '801790.SI', '801880.SI', '801890.SI']
        sw_lv2_index = ['801011.SI', '801012.SI', '801013.SI', '801014.SI', '801015.SI', '801016.SI', '801017.SI',
                        '801018.SI', '801021.SI', '801022.SI', '801023.SI', '801024.SI', '801032.SI', '801033.SI',
                        '801034.SI', '801035.SI', '801036.SI', '801037.SI', '801041.SI', '801051.SI', '801053.SI',
                        '801054.SI', '801055.SI', '801072.SI', '801073.SI', '801074.SI', '801075.SI', '801076.SI',
                        '801081.SI', '801082.SI', '801083.SI', '801084.SI', '801085.SI', '801092.SI', '801093.SI',
                        '801094.SI', '801101.SI', '801102.SI', '801111.SI', '801112.SI', '801123.SI', '801124.SI',
                        '801131.SI', '801132.SI', '801141.SI', '801142.SI', '801143.SI', '801144.SI', '801151.SI',
                        '801152.SI', '801153.SI', '801154.SI', '801155.SI', '801156.SI', '801161.SI', '801162.SI',
                        '801163.SI', '801164.SI', '801171.SI', '801172.SI', '801173.SI', '801174.SI', '801175.SI',
                        '801176.SI', '801177.SI', '801178.SI', '801181.SI', '801182.SI', '801191.SI', '801192.SI',
                        '801193.SI', '801194.SI', '801202.SI', '801203.SI', '801204.SI', '801205.SI', '801211.SI',
                        '801212.SI', '801213.SI', '801214.SI', '801215.SI', '801222.SI', '801223.SI', '801231.SI',
                        '801711.SI', '801712.SI', '801713.SI', '801721.SI', '801722.SI', '801723.SI', '801724.SI',
                        '801725.SI', '801731.SI', '801732.SI', '801733.SI', '801734.SI', '801741.SI', '801742.SI',
                        '801743.SI', '801744.SI', '801751.SI', '801752.SI', '801761.SI', '801881.SI']
        query_dict = {
            'citics_lv1': "select S_INFO_WINDCODE,s_con_windcode,s_con_indate,s_con_outdate "
                          "from wind_filesync.AIndexMembersCITICS ",
            'citics_lv2': "select S_INFO_WINDCODE,s_con_windcode,s_con_indate,s_con_outdate " 
                          "from wind_filesync.AIndexMembersCITICS2",
            'citics_lv3': "select S_INFO_WINDCODE,s_con_windcode,s_con_indate,s_con_outdate "
                          "from wind_filesync.AIndexMembersCITICS3",
            'sw_lv1': "select S_INFO_WINDCODE, S_CON_WINDCODE, S_CON_INDATE, S_CON_OUTDATE "
                      "from wind_filesync.SWIndexMembers "
                      "where S_INFO_WINDCODE in " + str(tuple(sw_lv1_index)),
            'sw_lv2': "select S_INFO_WINDCODE, S_CON_WINDCODE, S_CON_INDATE, S_CON_OUTDATE "
                      "from wind_filesync.SWIndexMembers "
                      "where S_INFO_WINDCODE in " + str(tuple(sw_lv2_index))}
        # 读取行业代码, entry_date, exit_date
        query = query_dict[indu_type]
        r = rdf_data()
        r.curs.execute(query)
        indu_code_col = indu_type + '_code'
        indu_name_col = indu_type + '_name'
        raw_df = pd.DataFrame(r.curs.fetchall(), columns=[indu_code_col, 'code', 'entry_date', 'exit_date'])
        raw_df = raw_df.loc[(raw_df['entry_date'] <= str(end)) &
                            ((raw_df['exit_date'] >= str(start)) | pd.isnull(raw_df['exit_date'])), :]
        # 不填充直接groupby会把nan过滤掉
        raw_df['exit_date'] = raw_df['exit_date'].fillna(str(end))
        raw_df = raw_df.groupby(['code', 'entry_date', 'exit_date'])[indu_code_col].last()
        raw_df = pd.DataFrame(raw_df).reset_index()
        # 读取行业名称
        query = "select s_info_windcode, S_INFO_NAME " \
                "from wind_filesync.AIndexDescription " \
                "where s_info_windcode in " + str(tuple(raw_df[indu_code_col].unique()))
        r.curs.execute(query)
        code_name_dict = dict(r.curs.fetchall())
        raw_df[indu_name_col] = raw_df[indu_code_col].map(code_name_dict)
        # 变换行业到每天
        trade_day_values = []
        indu_code_values = []
        indu_name_values = []
        code_values = []
        for idx, row in raw_df.iterrows():
            s_date = max(row['entry_date'], str(start))
            if pd.isnull(row['exit_date']):
                e_date = str(end)
            else:
                e_date = min(row['exit_date'], str(end))
            trade_days = list(calendar[(calendar >= s_date) & (calendar <= e_date)])
            if not trade_days:
                continue
            trade_day_values.extend(trade_days)
            indu_code_values.extend([row[indu_code_col]] * len(trade_days))
            indu_name_values.extend([row[indu_name_col]] * len(trade_days))
            code_values.extend([row['code']] * len(trade_days))
        res_df = pd.DataFrame({'date': trade_day_values, 'code': code_values,
                               indu_code_col: indu_code_values, indu_name_col: indu_name_values})
        return res_df

    def process_data(self, start, end, n_jobs):
        calendar = self.rdf.get_trading_calendar()
        indu_types = ['citics_lv1', 'citics_lv2', 'citics_lv3', 'sw_lv1', 'sw_lv2']
        with parallel_backend('multiprocessing', n_jobs=len(indu_types)):
            res = Parallel()(delayed(StkIndustry.JOB_get_indu)
                             (indu_type, start, end, calendar) for indu_type in indu_types)
        df = res[0]
        for i in range(1, len(indu_types)):
            df = pd.merge(df, res[i], how='outer', on=['date', 'code'])
        df.set_index('date', inplace=True)
        conditions = [df['citics_lv2_name'].values == '证券Ⅱ(中信)', df['citics_lv2_name'] == '保险Ⅱ(中信)']
        choices = ['证券Ⅱ(中信)', '保险Ⅱ(中信)']
        df['improved_lv1'] = np.select(conditions, choices, default=df['citics_lv1_name'].values)
        df = df.where(pd.notnull(df), None)
        codes = df['code'].unique()
        split_codes = np.array_split(codes, n_jobs)
        with parallel_backend('multiprocessing', n_jobs=n_jobs):
            res = Parallel()(delayed(influxdbData.JOB_saveData)
                             (df, 'code', codes, self.db, self.measure) for codes in split_codes)
        print('Stock Industry finish')
        print('-' * 30)
        fail_list = []
        for r in res:
            fail_list.extend(r)
        return fail_list


if __name__ == '__main__':
    print(datetime.datetime.now())
    btd = StkIndustry()
    start = 20100101
    end = 20200410
    res = btd.process_data(start=start, end=end, n_jobs=N_JOBS)
    print(res)
    print("start: %i ~ end: %i is finish!" % (start, end))
    print(datetime.datetime.now())
