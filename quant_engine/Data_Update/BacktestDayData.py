# 回测所用日线数据维护
# 数据包含：

from influxdb_data import influxdbData
from rdf_data import rdf_data
import pandas as pd
import datetime
import dateutil.parser as dtparser

class BacktestDayData:
    def __init__(self):
        self.rdf = rdf_data()
        self.influx = influxdbData()

    def process_data(self,codelist,start_input,end_input):

        # 获取股票、指数、期货高开低收量价状态
        ohlc = self.rdf.get_ohlc(code_list=codelist,start_input=start_input,end_input=end_input)
        ohlc['date'] = pd.to_datetime(ohlc['date'],format="%Y%m%d")
        ohlc_index = self.rdf.get_index_ohlc(start_input=start_input,end_input=end_input)
        ohlc_index['date'] = pd.to_datetime(ohlc_index['date'],format="%Y%m%d")
        ohlc_futures = self.rdf.get_futures_ohlc(start_input=start_input,end_input=end_input)
        ohlc_futures['date'] = pd.to_datetime(ohlc_futures['date'],format="%Y%m%d")

        ohlc = pd.concat([ohlc,ohlc_index,ohlc_futures],axis=0)
        ohlc.set_index(['date', 'code'], inplace=True)
        print('ohlc loaded!')


        # 获取除权除息信息
        ex_right = self.rdf.get_EX_right_dvd(start_input=start_input,end_input=end_input)
        ex_right['date'] = pd.to_datetime(ex_right['date'])
        ex_right.set_index(['date','code'],inplace=True)
        merged_data = pd.concat([ohlc,ex_right],join='outer',axis=1)
        print('ex_right loaded!')


        # 获取权重信息
        IH_weight = self.rdf.get_index_comp_in_period('IH',start_input,end_input)
        IF_weight = self.rdf.get_index_comp_in_period('IF',start_input,end_input)
        IC_weight = self.rdf.get_index_comp_in_period('IC',start_input,end_input)
        IH_weight['date'] = pd.to_datetime(IH_weight['date'])
        IF_weight['date'] = pd.to_datetime(IF_weight['date'])
        IC_weight['date'] = pd.to_datetime(IC_weight['date'])
        IH_weight.rename(columns={'weight':'IH_weight'}, inplace = True)
        IF_weight.rename(columns={'weight':'IF_weight'}, inplace = True)
        IC_weight.rename(columns={'weight':'IC_weight'}, inplace = True)
        IH_weight.set_index(['date','stk_code'], inplace=True)
        IF_weight.set_index(['date','stk_code'], inplace=True)
        IC_weight.set_index(['date','stk_code'], inplace=True)

        merged_data = pd.concat([merged_data,IH_weight['IH_weight'],IF_weight['IF_weight'],IC_weight['IC_weight']],axis=1)
        print('index weight loaded!')


        # 获取st信息
        merged_data['isST'] = False
        merged_data = merged_data.swaplevel()
        merged_data.sort_index(level=0,inplace=True)
        st = self.rdf.get_st()
        st['entry_date'] = pd.to_datetime(st['entry_date'])
        st['exit_date'] = pd.to_datetime(st['exit_date'])

        for idx,row in st.iterrows():
            if pd.isnull(row['exit_date']):
                merged_data.loc[(row['code'],row['entry_date']):(row['code'],dtparser.parse(str(end_input))),'isST'] = True
            else:
                merged_data.loc[(row['code'],row['entry_date']):(row['code'],row['exit_date']),'isST'] = True
        print('st loaded!')


        # 获取行业信息
        # 中信
        merged_data['citics_lv1_code'] = None
        merged_data['citics_lv1_name'] = None
        merged_data['citics_lv2_code'] = None
        merged_data['citics_lv2_name'] = None
        merged_data['citics_lv3_code'] = None
        merged_data['citics_lv3_name'] = None
        citics_lv1 = self.rdf.get_citics_lv1()
        citics_lv1['entry_date'] = pd.to_datetime(citics_lv1['entry_date'])
        citics_lv1['exit_date'] = pd.to_datetime(citics_lv1['exit_date'])
        citics_lv2 = self.rdf.get_citics_lv2()
        citics_lv2['entry_date'] = pd.to_datetime(citics_lv2['entry_date'])
        citics_lv2['exit_date'] = pd.to_datetime(citics_lv2['exit_date'])
        citics_lv3 = self.rdf.get_citics_lv3()
        citics_lv3['entry_date'] = pd.to_datetime(citics_lv3['entry_date'])
        citics_lv3['exit_date'] = pd.to_datetime(citics_lv3['exit_date'])

        for idx,row in citics_lv1.iterrows():
            if pd.isnull(row['exit_date']):
                merged_data.loc[(row['stock_code'],row['entry_date']):(row['stock_code'],dtparser.parse(str(end_input))),'citics_lv1_code'] \
                    = row['index_lv1_code']
                merged_data.loc[(row['stock_code'],row['entry_date']):(row['stock_code'],dtparser.parse(str(end_input))),'citics_lv1_name'] \
                    = row['industry_lv1_name']
            else:
                merged_data.loc[(row['stock_code'],row['entry_date']):(row['stock_code'],row['exit_date']),'citics_lv1_code'] \
                    = row['index_lv1_code']
                merged_data.loc[(row['stock_code'],row['entry_date']):(row['stock_code'],row['exit_date']),'citics_lv1_name'] \
                    = row['industry_lv1_name']

        for idx, row in citics_lv2.iterrows():
            if pd.isnull(row['exit_date']):
                merged_data.loc[(row['stock_code'],row['entry_date']):(row['stock_code'],dtparser.parse(str(end_input))),'citics_lv2_code'] \
                    = row['index_lv2_code']
                merged_data.loc[(row['stock_code'],row['entry_date']):(row['stock_code'],dtparser.parse(str(end_input))),'citics_lv2_name'] \
                    = row['industry_lv2_name']
            else:
                merged_data.loc[(row['stock_code'],row['entry_date']):(row['stock_code'],row['exit_date']),'citics_lv2_code'] \
                    = row['index_lv2_code']
                merged_data.loc[(row['stock_code'],row['entry_date']):(row['stock_code'],row['exit_date']),'citics_lv2_name'] \
                    = row['industry_lv2_name']

        for idx, row in citics_lv3.iterrows():
            if pd.isnull(row['exit_date']):
                merged_data.loc[(row['stock_code'],row['entry_date']):(row['stock_code'],dtparser.parse(str(end_input))),'citics_lv3_code'] \
                    = row['index_lv3_code']
                merged_data.loc[(row['stock_code'],row['entry_date']):(row['stock_code'],dtparser.parse(str(end_input))),'citics_lv3_name'] \
                    = row['industry_lv3_name']
            else:
                merged_data.loc[(row['stock_code'],row['entry_date']):(row['stock_code'],row['exit_date']),'citics_lv3_code'] \
                    = row['index_lv3_code']
                merged_data.loc[(row['stock_code'],row['entry_date']):(row['stock_code'],row['exit_date']),'citics_lv3_name'] \
                    = row['industry_lv3_name']
        print('industry(citics) loaded!')

        # 申万
        merged_data['sw_lv1_code'] = None
        merged_data['sw_lv1_name'] = None
        merged_data['sw_lv2_code'] = None
        merged_data['sw_lv2_name'] = None
        sw_lv1 = self.rdf.get_SW_lv1()
        sw_lv1['entry_date'] = pd.to_datetime(sw_lv1['entry_date'])
        sw_lv1['exit_date'] = pd.to_datetime(sw_lv1['exit_date'])
        sw_lv2 = self.rdf.get_SW_lv2()
        sw_lv2['entry_date'] = pd.to_datetime(sw_lv2['entry_date'])
        sw_lv2['exit_date'] = pd.to_datetime(sw_lv2['exit_date'])

        for idx, row in sw_lv1.iterrows():
            if pd.isnull(row['exit_date']):
                merged_data.loc[(row['stock_code'],row['entry_date']):(row['stock_code'],dtparser.parse(str(end_input))),'sw_lv1_code'] \
                    = row['index_lv1_code']
                merged_data.loc[(row['stock_code'],row['entry_date']):(row['stock_code'],dtparser.parse(str(end_input))),'sw_lv1_name'] \
                    = row['industry_lv1_name']
            else:
                merged_data.loc[(row['stock_code'],row['entry_date']):(row['stock_code'],row['exit_date']),'sw_lv1_code'] \
                    = row['index_lv1_code']
                merged_data.loc[(row['stock_code'],row['entry_date']):(row['stock_code'],row['exit_date']),'sw_lv1_name'] \
                    = row['industry_lv1_name']

        for idx, row in sw_lv2.iterrows():
            if pd.isnull(row['exit_date']):
                merged_data.loc[(row['stock_code'],row['entry_date']):(row['stock_code'],dtparser.parse(str(end_input))),'sw_lv2_code'] \
                    = row['index_lv2_code']
                merged_data.loc[(row['stock_code'],row['entry_date']):(row['stock_code'],dtparser.parse(str(end_input))),'sw_lv2_name'] \
                    = row['industry_lv2_name']
            else:
                merged_data.loc[(row['stock_code'],row['entry_date']):(row['stock_code'],row['exit_date']),'sw_lv2_code'] \
                    = row['index_lv2_code']
                merged_data.loc[(row['stock_code'],row['entry_date']):(row['stock_code'],row['exit_date']),'sw_lv2_name'] \
                    = row['industry_lv2_name']

        print('industry(SW) loaded!')


        # 存数据前整理
        merged_data.reset_index(inplace=True)
        merged_data.rename(columns={'level_0':'code','level_1':'date'},inplace=True)
        merged_data.set_index('date',inplace=True)
        merged_data = merged_data.where(pd.notnull(merged_data), None)
        print('data prepared!')

        return merged_data



if __name__ == '__main__':
    print(datetime.datetime.now())
    btd = BacktestDayData()
    start = 20100101
    end = 20130101
    data = btd.process_data(codelist=None,start_input=start,end_input=end)
    btd.influx.saveData(data,'DailyData_backtest','marketData')
    print("start: %i ~ end: %i is finish!" % (start, end))
    print(datetime.datetime.now())