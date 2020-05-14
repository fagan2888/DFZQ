from csindex_ftp_down import *
import datetime
from dateutil.relativedelta import relativedelta
import dateutil.parser as dtparser
import pandas as pd
import numpy as np
import os
import cx_Oracle as oracle
from rdf_data import rdf_data
from ftp_data import ftp_data
from joblib import Parallel, delayed, parallel_backend

import warnings


class div_pred_statistic:
    def __init__(self):
        # ftp initialize
        self.ftp = ftp_data()
        self.ftp_client = FtpClient('192.168.38.213', 'index', 'dfzq1234')

        # get today 0:00:00
        today = datetime.datetime.today().date()
        self.dt = datetime.datetime(year=today.year, month=today.month, day=today.day)
        # self.dt = self.dt - datetime.timedelta(days=1)
        self.date_needed = self.get_date_needed(self.dt)

        self.this_year = datetime.datetime.now().date().year
        self.this_year_start_date = datetime.datetime(year=self.this_year, month=1, day=1)
        self.this_year_end_date = datetime.datetime(year=self.this_year, month=12, day=31)
        self.last_year_start_date = datetime.datetime(year=self.this_year - 1, month=1, day=1)
        self.last_year_end_date = datetime.datetime(year=self.this_year - 1, month=12, day=31)

        # rdf initialize
        self.rdf = rdf_data()
        self.trading_calendar = self.rdf.get_trading_calendar()
        self.last_trading_date = self.trading_calendar[self.trading_calendar < self.dt].iloc[-1]

    def get_index_constituent_from_csv(self, index_type):
        file_name = index_type + "_weight.csv"
        index_constituent_df = pd.read_csv(file_name, encoding='gbk', index_col=0)
        index_constituent_df = index_constituent_df.loc[:, ['index_code', 'code', 'close', 'weight']]
        index_constituent_df.columns = ['index_code', 'constituent_code', 'close', 'weight']
        # print('from CSV: %s constituent got!' %index_type)
        return index_constituent_df

    def cal_div_point(self, div_pre_tax, stock_close_price, index_close_price, weight):
        div_point_value = div_pre_tax / stock_close_price * weight * index_close_price / 100
        return div_point_value

    def process_prediction(self, code, stock_div_df, stock_weight, index_close_df, index_type, mode):
        result_df = pd.DataFrame(columns=['股票代码', '股票名称', '状态', '每股转送', '股权登记日', '派息日',
                                          '预案公告日', '税前派息', '分红点数', '是否预测'])

        stock_code_processing = code
        stock_name_processing = self.rdf.get_stock_name(stock_code_processing)[stock_code_processing]
        stock_close_df = self.rdf.get_stock_close(stock_code_processing, self.last_trading_date)
        stock_close = stock_close_df['收盘价'].iloc[0]
        index_close = index_close_df.loc[self.last_trading_date, '收盘价']
        stock_div_df['派息日'] = pd.to_datetime(stock_div_df['派息日'], format="%Y%m%d")
        stock_div_df['股权登记日'] = pd.to_datetime(stock_div_df['股权登记日'], format="%Y%m%d")

        unconfirmed_div_df = stock_div_df.loc[stock_div_df['派息日'].isna(), :].copy()
        confirmed_div_df = stock_div_df.loc[stock_div_df['派息日'].notna(), :].copy()

        div_this_year_df = confirmed_div_df.loc[(confirmed_div_df['派息日'] >= self.this_year_start_date) &
                                                (confirmed_div_df['派息日'] <= self.this_year_end_date), :]
        div_last_year_df = confirmed_div_df.loc[(confirmed_div_df['派息日'] >= self.last_year_start_date) &
                                                (confirmed_div_df['派息日'] <= self.last_year_end_date), :]
        this_year_div_times = len(unconfirmed_div_df) + len(div_this_year_df)

        # 去年没分，今年不论有没有，都不预测
        if div_last_year_df.empty:
            if this_year_div_times == 0:
                return
            else:
                if len(unconfirmed_div_df) > 0:
                    div_dt = self.trading_calendar[self.trading_calendar >= unconfirmed_div_df.iloc[0]['预案公告日']
                                                   + relativedelta(months=2)].reset_index(drop=True)[0]
                    temp_series = self.trading_calendar[self.trading_calendar <= div_dt].reset_index(drop=True)
                    reg_dt = temp_series.iloc[-1]
                    if reg_dt <= self.dt + datetime.timedelta(days=7):
                        reg_dt = np.nan
                        div_dt = np.nan
                    status = '去年无分红，今年已有预案，预测派息日和股权登记日'
                    if_predicted = True
                    conversed_ratio = unconfirmed_div_df.iloc[0]['每股转送']
                    ann_dt = unconfirmed_div_df.iloc[0]['预案公告日']
                    div_pretax = unconfirmed_div_df.iloc[0]['税前派息']
                    bonus_point = self.cal_div_point(div_pretax, stock_close, index_close, stock_weight)
                    temp_df = pd.DataFrame({'股票代码': [stock_code_processing], '股票名称': [stock_name_processing],
                                            '状态': [status], '是否预测': [if_predicted], '每股转送': [conversed_ratio],
                                            '股权登记日': [reg_dt], '派息日': [div_dt],
                                            '预案公告日': [ann_dt], '税前派息': [div_pretax],
                                            '分红点数': [bonus_point]}, index=[0])
                    result_df = result_df.append(temp_df, ignore_index=True)

                if len(div_this_year_df) > 0:
                    for i in range(len(div_this_year_df)):
                        status = '去年无分红，今年已有公告，实际派息日和股权登记日'
                        if_predicted = False
                        conversed_ratio = div_this_year_df.iloc[i]['每股转送']
                        reg_dt = div_this_year_df.iloc[i]['股权登记日']
                        div_dt = div_this_year_df.iloc[i]['派息日']
                        ann_dt = div_this_year_df.iloc[i]['预案公告日']
                        div_pretax = div_this_year_df.iloc[i]['税前派息']
                        if div_dt < self.last_trading_date:
                            if mode == 'rdf':
                                idx_constituent_that_day = self.rdf.get_index_constituent_df(index_type, div_dt)
                                stk_weight_series = idx_constituent_that_day.loc[
                                    idx_constituent_that_day['constituent_code'] == code, 'weight']
                                if stk_weight_series.empty:
                                    stk_weight_that_day = 0
                                else:
                                    stk_weight_that_day = stk_weight_series.iloc[0]
                            else:
                                stk_weight_that_day = stock_weight
                            stk_close_that_day = self.rdf.get_stock_close(stock_code_processing, div_dt)['收盘价'].iloc[0]
                            idx_close_that_day = index_close_df.loc[div_dt, '收盘价']
                            bonus_point = self.cal_div_point(div_pretax, stk_close_that_day, idx_close_that_day,
                                                             stk_weight_that_day)
                        else:
                            bonus_point = self.cal_div_point(div_pretax, stock_close, index_close, stock_weight)
                        temp_df = pd.DataFrame({'股票代码': [stock_code_processing], '股票名称': [stock_name_processing],
                                                '状态': [status], '是否预测': [if_predicted], '每股转送': [conversed_ratio],
                                                '股权登记日': [reg_dt], '派息日': [div_dt],
                                                '预案公告日': [ann_dt], '税前派息': [div_pretax],
                                                '分红点数': [bonus_point]}, index=[0])
                        result_df = result_df.append(temp_df, ignore_index=True)

        # 去年分红
        else:
            stock_eps_df = self.eps_df.loc[self.eps_df['code'] == stock_code_processing, :]
            eps_df_before_dt = stock_eps_df.loc[:self.dt, :]
            if eps_df_before_dt.empty:
                eps_on_dt = 1
                eps_last_year = 1
            else:
                eps_on_dt = eps_df_before_dt.iloc[-1]['eps']
                eps_date = eps_df_before_dt.index[-1]
                eps_date_last_year = eps_date - relativedelta(years=1)
                if eps_df_before_dt.loc[:eps_date_last_year, :].empty:
                    eps_on_dt = 1
                    eps_last_year = 1
                else:
                    eps_last_year = eps_df_before_dt.loc[:eps_date_last_year, 'eps'].iloc[-1]

            # 今年没有分红预案或公告
            if this_year_div_times == 0:
                if self.dt.month < 10:
                    for i in range(len(div_last_year_df)):
                        status = '去年有分红，今年无预案无公告，预测分红'
                        if_predicted = True
                        ann_dt = np.nan
                        div_on_hist_date = div_last_year_df.iloc[i]['税前派息']
                        if eps_last_year < 0 and eps_on_dt >= 0:
                            div_pretax = div_on_hist_date
                        elif eps_last_year > 0 and eps_on_dt > 0:
                            div_pretax = div_on_hist_date / eps_last_year * eps_on_dt
                        else:
                            continue

                        conversed_ratio = np.nan
                        bonus_point = self.cal_div_point(div_pretax, stock_close, index_close, stock_weight)
                        hist_div_date = div_last_year_df.iloc[i]['派息日']
                        div_dt = hist_div_date + relativedelta(years=1)
                        div_dt = self.trading_calendar[self.trading_calendar >=
                                                       (div_dt - datetime.timedelta(days=1))].reset_index(drop=True)[0]
                        temp_series = self.trading_calendar[self.trading_calendar <= div_dt].reset_index(drop=True)
                        reg_dt = temp_series.iloc[-1]
                        if reg_dt <= self.dt + datetime.timedelta(days=7):
                            reg_dt = np.nan
                            div_dt = np.nan
                        temp_df = pd.DataFrame({'股票代码': [stock_code_processing], '股票名称': [stock_name_processing],
                                                '状态': [status], '是否预测': [if_predicted], '每股转送': [conversed_ratio],
                                                '股权登记日': [reg_dt], '派息日': [div_dt],
                                                '预案公告日': [ann_dt], '税前派息': [div_pretax],
                                                '分红点数': [bonus_point]}, index=[0])
                        result_df = result_df.append(temp_df, ignore_index=True)
                else:
                    pass

            # 今年有公告
            if len(div_this_year_df) > 0:
                for i in range(len(div_this_year_df)):
                    status = '去年有分红，今年已有公告，实际派息日和股权登记日'
                    if_predicted = False
                    conversed_ratio = div_this_year_df.iloc[i]['每股转送']
                    reg_dt = div_this_year_df.iloc[i]['股权登记日']
                    div_dt = div_this_year_df.iloc[i]['派息日']
                    ann_dt = div_this_year_df.iloc[i]['预案公告日']
                    div_pretax = div_this_year_df.iloc[i]['税前派息']
                    if div_dt < self.last_trading_date:
                        if mode == 'rdf':
                            idx_constituent_that_day = self.rdf.get_index_constituent_df(index_type, div_dt)
                            stk_weight_series = idx_constituent_that_day. \
                                loc[idx_constituent_that_day['constituent_code'] == code, 'weight']
                            if stk_weight_series.empty:
                                stk_weight_that_day = 0
                            else:
                                stk_weight_that_day = stk_weight_series.iloc[0]
                        else:
                            stk_weight_that_day = stock_weight
                        stk_close_that_day = self.rdf.get_stock_close(stock_code_processing, div_dt)['收盘价'].iloc[0]
                        idx_close_that_day = index_close_df.loc[div_dt, '收盘价']
                        bonus_point = self.cal_div_point(div_pretax, stk_close_that_day, idx_close_that_day,
                                                         stk_weight_that_day)
                    else:
                        bonus_point = self.cal_div_point(div_pretax, stock_close, index_close, stock_weight)
                    temp_df = pd.DataFrame({'股票代码': [stock_code_processing], '股票名称': [stock_name_processing],
                                            '状态': [status], '是否预测': [if_predicted], '每股转送': [conversed_ratio],
                                            '股权登记日': [reg_dt], '派息日': [div_dt],
                                            '预案公告日': [ann_dt], '税前派息': [div_pretax],
                                            '分红点数': [bonus_point]}, index=[0])
                    result_df = result_df.append(temp_df, ignore_index=True)

            # 今年有预案
            if len(unconfirmed_div_df) > 0:
                conversed_ratio = unconfirmed_div_df.iloc[0]['每股转送']
                ann_dt = unconfirmed_div_df.iloc[0]['预案公告日']
                div_pretax = unconfirmed_div_df.iloc[0]['税前派息']
                bonus_point = self.cal_div_point(div_pretax, stock_close, index_close, stock_weight)
                status = '去年有分红，今年已有预案，预测派息日和股权登记日'
                if_predicted = True

                estimated_df = div_last_year_df.copy().reset_index()
                for i in range(len(estimated_df)):
                    estimated_df.loc[i, 'estimated_div_date'] = \
                        estimated_df.loc[i, '派息日'] + relativedelta(years=1) - datetime.timedelta(days=1)
                # 今年没有分红，分红日期参照去年+1年，且>=今天
                if div_this_year_df.empty:
                    estimated_df = estimated_df.loc[estimated_df['estimated_div_date'] > self.dt, :]
                    # 若没符合的，预案日期+2个月
                    if estimated_df.empty:
                        div_dt = self.trading_calendar[self.trading_calendar >=
                                                       unconfirmed_div_df.iloc[0]['预案公告日'] + relativedelta(
                            months=2)].reset_index(drop=True)[0]
                        temp_series = \
                            self.trading_calendar[self.trading_calendar <= div_dt].reset_index(drop=True)
                        reg_dt = temp_series.iloc[-1]
                        if reg_dt <= self.dt + datetime.timedelta(days=7):
                            reg_dt = np.nan
                            div_dt = np.nan
                    else:
                        div_dt = self.trading_calendar[self.trading_calendar >=
                                                       estimated_df.iloc[0]['estimated_div_date']].reset_index(
                            drop=True)[0]
                        temp_series = \
                            self.trading_calendar[self.trading_calendar < div_dt].reset_index(drop=True)
                        reg_dt = temp_series.iloc[-1]
                        if reg_dt <= self.dt + datetime.timedelta(days=7):
                            reg_dt = np.nan
                            div_dt = np.nan

                # 今年已有分红，分红日期参照去年+1年，且下次分红日期 >= 最新已确认日期 + 1个月
                else:
                    estimated_df = estimated_df.loc[estimated_df['estimated_div_date'] >
                                                    div_this_year_df.iloc[-1]['派息日'] + relativedelta(months=1), :]
                    # 若没符合的，预案日期+2个月
                    if estimated_df.empty:
                        div_dt = self.trading_calendar[self.trading_calendar >=
                                                       (unconfirmed_div_df.iloc[0]['预案公告日'] + relativedelta(
                                                           months=2))].reset_index(drop=True)[0]
                        temp_series = \
                            self.trading_calendar[self.trading_calendar <= div_dt].reset_index(drop=True)
                        reg_dt = temp_series.iloc[-1]
                        if reg_dt <= self.dt + datetime.timedelta(days=7):
                            reg_dt = np.nan
                            div_dt = np.nan
                    else:
                        div_dt = self.trading_calendar[self.trading_calendar >=
                                                       estimated_df.iloc[0]['estimated_div_date']].reset_index(
                            drop=True)[0]
                        temp_series = \
                            self.trading_calendar[self.trading_calendar < div_dt].reset_index(drop=True)
                        reg_dt = temp_series.iloc[-1]
                        if reg_dt <= self.dt + datetime.timedelta(days=7):
                            reg_dt = np.nan
                            div_dt = np.nan

                temp_df = pd.DataFrame({'股票代码': [stock_code_processing], '股票名称': [stock_name_processing],
                                        '状态': [status], '是否预测': [if_predicted], '每股转送': [conversed_ratio],
                                        '股权登记日': [reg_dt], '派息日': [div_dt],
                                        '预案公告日': [ann_dt], '税前派息': [div_pretax],
                                        '分红点数': [bonus_point]}, index=[0])
                result_df = result_df.append(temp_df, ignore_index=True)

            # 同股息率下分红额未满，预测下一次
            # 日期超过10月时，不再预测分红未满的情况
            if self.dt.month < 10:
                div_sum_last_year = 0
                for i in range(len(div_last_year_df)):
                    div_sum_last_year += div_last_year_df.iloc[i]['税前派息']
                div_sum_this_year = 0
                if len(unconfirmed_div_df) > 0:
                    div_sum_this_year += unconfirmed_div_df.iloc[0]['税前派息']
                if len(div_this_year_df) > 0:
                    div_sum_this_year += div_this_year_df['税前派息'].sum()

                to_div_times = len(div_last_year_df) - this_year_div_times
                div_pretax = div_sum_last_year / eps_last_year * eps_on_dt - div_sum_this_year
                if div_pretax <= 0 or to_div_times <= 0 or this_year_div_times == 0:
                    pass
                else:
                    status = '去年有分红，今年金额未满，预测仍将分红'
                    if_predicted = True
                    conversed_ratio = np.nan
                    ann_dt = np.nan
                    bonus_point = self.cal_div_point(div_pretax, stock_close, index_close, stock_weight)
                    if div_last_year_df.iloc[-1]['派息日'] + relativedelta(years=1) <= self.dt:
                        div_dt = np.nan
                        reg_dt = np.nan
                    else:
                        div_dt = self.trading_calendar[self.trading_calendar >= div_last_year_df.iloc[-1]['派息日']
                                                       + relativedelta(years=1) - datetime.timedelta(
                            days=1)].reset_index(drop=True)[0]
                        temp_series = self.trading_calendar[self.trading_calendar < div_dt].reset_index(drop=True)
                        reg_dt = temp_series.iloc[-1]
                        if reg_dt <= self.dt + datetime.timedelta(days=7):
                            reg_dt = np.nan
                            div_dt = np.nan

                    temp_df = pd.DataFrame({'股票代码': [stock_code_processing], '股票名称': [stock_name_processing],
                                            '状态': [status], '是否预测': [if_predicted], '每股转送': [conversed_ratio],
                                            '股权登记日': [reg_dt], '派息日': [div_dt],
                                            '预案公告日': [ann_dt], '税前派息': [div_pretax],
                                            '分红点数': [bonus_point]}, index=[0])
                    result_df = result_df.append(temp_df, ignore_index=True)
            else:
                pass

        # >10月时,开始预测下一年情况
        if self.dt.month >= 10:
            confirmed_next_year_df = stock_div_df.loc[stock_div_df['派息日'] > self.this_year_end_date, :].copy()
            for idx, row in confirmed_next_year_df.iterrows():
                div_dt = row['派息日']
                reg_dt = row['股权登记日']
                status = '今年无分红，明年已有公告，实际派息日和股权登记日'
                if_predicted = False
                conversed_ratio = row['每股转送']
                ann_dt = row['预案公告日']
                div_pretax = row['税前派息']
                bonus_point = self.cal_div_point(div_pretax, stock_close, index_close, stock_weight)
                temp_df = pd.DataFrame({'股票代码': [stock_code_processing], '股票名称': [stock_name_processing],
                                        '状态': [status], '是否预测': [if_predicted], '每股转送': [conversed_ratio],
                                        '股权登记日': [reg_dt], '派息日': [div_dt],
                                        '预案公告日': [ann_dt], '税前派息': [div_pretax],
                                        '分红点数': [bonus_point]}, index=[0])
                result_df = result_df.append(temp_df, ignore_index=True)
            if confirmed_next_year_df.empty:
                div_this_year_df = confirmed_div_df.loc[(confirmed_div_df['派息日'] >= self.this_year_start_date) &
                                                        (confirmed_div_df['派息日'] <=
                                                         self.date_needed[-1] - relativedelta(years=1)), :]
            else:
                div_this_year_df = confirmed_div_df.loc[(confirmed_div_df['派息日'] >= div_dt + relativedelta(months=1)) &
                                                        (confirmed_div_df['派息日'] <=
                                                         self.date_needed[-1] - relativedelta(years=1)), :]

            stock_eps_df = self.eps_df.loc[self.eps_df['code'] == stock_code_processing, :]
            eps_df_before_dt = stock_eps_df.loc[:self.dt, :]

            for idx, row in div_this_year_df.iterrows():
                status = '今年有分红，明年无预案无公告，预测分红'
                if_predicted = True
                ann_dt = np.nan
                if eps_df_before_dt.loc[:row['派息日'], :].empty:
                    div_pretax = row['税前派息']
                else:
                    eps_on_dt = eps_df_before_dt.iloc[-1]['eps']
                    eps_div_dt = eps_df_before_dt.loc[:row['派息日'], 'eps'].iloc[-1]
                    if eps_div_dt < 0 and eps_div_dt > 0:
                        div_pretax = row['税前派息']
                    elif eps_div_dt > 0 and eps_on_dt > 0:
                        div_pretax = row['税前派息'] / eps_div_dt * eps_on_dt
                    else:
                        continue
                conversed_ratio = np.nan
                bonus_point = self.cal_div_point(div_pretax, stock_close, index_close, stock_weight)
                hist_div_dt = row['派息日']
                div_dt = hist_div_dt + relativedelta(years=1)
                div_dt = self.trading_calendar[self.trading_calendar >=
                                               (div_dt - datetime.timedelta(days=1))].reset_index(drop=True)[0]
                temp_series = self.trading_calendar[self.trading_calendar < div_dt].reset_index(drop=True)
                reg_dt = temp_series.iloc[-1]
                temp_df = pd.DataFrame({'股票代码': [stock_code_processing], '股票名称': [stock_name_processing],
                                        '状态': [status], '是否预测': [if_predicted], '每股转送': [conversed_ratio],
                                        '股权登记日': [reg_dt], '派息日': [div_dt],
                                        '预案公告日': [ann_dt], '税前派息': [div_pretax],
                                        '分红点数': [bonus_point]}, index=[0])
                result_df = result_df.append(temp_df, ignore_index=True)

        return result_df

    def constituent_div_prediction(self, dt_input, index_type, years_back=2, mode='rdf'):
        # get constituent
        if mode == 'rdf':
            # 有问题改日期
            index_constituent_df = self.rdf.get_index_constituent_df(index_type, self.dt)
        else:
            index_constituent_df = self.get_index_constituent_from_csv(index_type)

        index_close_df = self.rdf.get_index_close(index_type, self.last_trading_date - relativedelta(years=1),
                                                  self.last_trading_date)
        index_close_df.set_index('日期', inplace=True)
        index_constituent_list = index_constituent_df['constituent_code'].tolist()

        # get eps data
        self.eps_df = self.rdf.get_eps_df(index_constituent_list)
        # get constituent div
        div_df_list = []
        dt_input = dt_input + datetime.timedelta(days=2)
        while years_back > 1:
            dt_start = dt_input - relativedelta(years=years_back) - datetime.timedelta(days=years_back + 1)
            dt_end = dt_start + relativedelta(years=1)
            div_df_list.append(self.rdf.get_constituent_div_df(index_constituent_list, dt_start, dt_end))
            years_back = years_back - 1
        div_df_list.append(self.rdf.get_constituent_div_df(index_constituent_list, dt_end + datetime.timedelta(1)))

        history_div_df = pd.concat(div_df_list, ignore_index=True)
        history_div_df = history_div_df.loc[history_div_df.loc[:, '税前派息'] != 0, :].copy()
        history_div_df.loc[:, '预案公告日'] = pd.to_datetime(history_div_df.loc[:, '预案公告日'], format="%Y%m%d")
        # process every single stock
        prediction_df_list = []
        for code in index_constituent_list:
            stock_div_df = history_div_df.loc[history_div_df.loc[:, '股票代码'] == code, :]
            stock_weight = index_constituent_df.loc[index_constituent_df['constituent_code'] == code, 'weight'].iloc[0]
            prediction_df_list.append(
                self.process_prediction(code, stock_div_df, stock_weight, index_close_df, index_type, mode))
        constituent_div_df = pd.concat(prediction_df_list, ignore_index=True)
        constituent_div_df.sort_values('派息日', ascending=True, inplace=True)
        constituent_div_df = constituent_div_df.loc[:, ['股票代码', '股票名称', '状态', '税前派息', '分红点数',
                                                        '每股转送', '预案公告日', '股权登记日', '派息日', '是否预测']]
        return constituent_div_df

    def get_prompt_date(self, dt_input):
        start_date_this_month = datetime.datetime(year=dt_input.date().year, month=dt_input.date().month, day=1)
        if start_date_this_month.weekday() <= 4:
            prompt_date = start_date_this_month + datetime.timedelta(days=(14 + (4 - start_date_this_month.weekday())))
        else:
            prompt_date = start_date_this_month + datetime.timedelta(days=(21 - (start_date_this_month.weekday() - 4)))
        return prompt_date

    def get_date_needed(self, dt_input):
        date_needed_list = []
        prompt_date_this_month = self.get_prompt_date(dt_input)
        dt_next_month = dt_input + relativedelta(months=1)
        prompt_date_next_month = self.get_prompt_date(dt_next_month)

        if dt_input <= prompt_date_this_month:
            near_prompt = prompt_date_this_month
            if dt_input <= (prompt_date_this_month - datetime.timedelta(days=7)):
                friday_before_near_prompt = (prompt_date_this_month - datetime.timedelta(days=7))
                date_needed_list.append(friday_before_near_prompt)
            next_prompt = prompt_date_next_month
            friday_before_next_prompt = prompt_date_next_month - datetime.timedelta(days=7)
        else:
            friday_before_near_prompt = prompt_date_next_month - datetime.timedelta(days=7)
            date_needed_list.append(friday_before_near_prompt)
            near_prompt = prompt_date_next_month
            next_prompt = self.get_prompt_date(near_prompt + relativedelta(months=1))
            friday_before_next_prompt = next_prompt - datetime.timedelta(days=7)
        date_needed_list.extend([near_prompt, friday_before_next_prompt, next_prompt])

        month_exceed = near_prompt.date().month % 3
        if month_exceed == 2:
            date_needed_list.extend(
                [self.get_prompt_date(near_prompt + relativedelta(months=4)) - datetime.timedelta(days=7),
                 self.get_prompt_date(near_prompt + relativedelta(months=4)),
                 self.get_prompt_date(near_prompt + relativedelta(months=7)) - datetime.timedelta(days=7),
                 self.get_prompt_date(near_prompt + relativedelta(months=7))])
        else:
            date_needed_list.extend(
                [self.get_prompt_date(near_prompt + relativedelta(months=3 - month_exceed)) - datetime.timedelta(
                    days=7),
                 self.get_prompt_date(near_prompt + relativedelta(months=3 - month_exceed)),
                 self.get_prompt_date(near_prompt + relativedelta(months=6 - month_exceed)) - datetime.timedelta(
                     days=7),
                 self.get_prompt_date(near_prompt + relativedelta(months=6 - month_exceed))])

        return date_needed_list

    def modify(self, raw_df, index_type):
        path_prefix = 'D:/div_statistics/revising_file/'
        revise_filepath = path_prefix + index_type + '_modify' + '.csv'
        revising_df = pd.read_csv(revise_filepath, encoding='utf-8', index_col=0)
        revising_df['预案公告日'] = pd.to_datetime(revising_df['预案公告日'], format='%Y-%m-%d %H:%M:%S')
        revising_df['股权登记日'] = pd.to_datetime(revising_df['股权登记日'], format='%Y-%m-%d %H:%M:%S')
        revising_df['派息日'] = pd.to_datetime(revising_df['派息日'], format='%Y-%m-%d %H:%M:%S')
        print(revising_df)

        modified_code_list = []
        modified_df = raw_df.copy()
        for idx, row in revising_df.iterrows():
            code = row['股票代码']
            status = row['状态']
            if raw_df.loc[(raw_df['股票代码'] == code) & (raw_df['状态'] == status), :].empty:
                continue
            if not raw_df.loc[(raw_df['股票代码'] == code) & (raw_df['状态'] == status), '是否预测'].iloc[0]:
                continue
            else:
                if [code, status] in modified_code_list:
                    tmp_df = raw_df.loc[(raw_df['股票代码'] == code) & (raw_df['状态'] == status), :].copy()
                    if pd.notnull(row['税前派息']):
                        multi = float(row['税前派息'].split('*')[1])
                        tmp_df.loc[:, '税前派息'] = multi * tmp_df.loc[:, '税前派息']
                    if pd.notnull(row['分红点数']):
                        multi = float(row['分红点数'].split('*')[1])
                        tmp_df.loc[:, '分红点数'] = multi * tmp_df.loc[:, '分红点数']
                    if pd.notnull(row['每股转送']):
                        tmp_df.loc[:, '每股转送'] = row['每股转送']
                    if pd.notnull(row['股权登记日']):
                        tmp_df.loc[:, '股权登记日'] = row['股权登记日']
                    if pd.notnull(row['派息日']):
                        tmp_df.loc[:, '派息日'] = row['派息日']
                    modified_df = pd.concat([modified_df, tmp_df], ignore_index=True)
                else:
                    if pd.notnull(row['税前派息']):
                        multi = float(row['税前派息'].split('*')[1])
                        modified_df.loc[(modified_df['股票代码'] == code) & (modified_df['状态'] == status), '税前派息'] \
                            = multi * modified_df.loc[(modified_df['股票代码'] == code) &
                                                      (modified_df['状态'] == status), '税前派息']
                    if pd.notnull(row['分红点数']):
                        multi = float(row['分红点数'].split('*')[1])
                        modified_df.loc[(modified_df['股票代码'] == code) & (modified_df['状态'] == status), '分红点数'] \
                            = multi * modified_df.loc[(modified_df['股票代码'] == code) &
                                                      (modified_df['状态'] == status), '分红点数']
                    if pd.notnull(row['每股转送']):
                        modified_df.loc[(modified_df['股票代码'] == code) & (modified_df['状态'] == status), '每股转送'] \
                            = row['每股转送']
                    if pd.notnull(row['股权登记日']):
                        modified_df.loc[(modified_df['股票代码'] == code) & (modified_df['状态'] == status), '股权登记日']\
                            = row['股权登记日']
                    if pd.notnull(row['派息日']):
                        modified_df.loc[(modified_df['股票代码'] == code) & (modified_df['状态'] == status), '派息日'] \
                            = row['派息日']
                    modified_code_list.append([code, status])
        return modified_df

    def get_summary_date_dict(self, df):
        date_dict = {}
        df = df.dropna(subset=['派息日']).sort_values(by='派息日')
        df.set_index("派息日", inplace=True)
        for date in self.date_needed:
            date_dict[date] = df.loc[self.dt + datetime.timedelta(days=1):date, '分红点数'].sum()
        return date_dict

    def run_daily(self, exe_date=None):
        local_prefix = 'D:/div_statistics/'
        remote_utf8_prefix = 'dividend/indexBonus/'
        remote_gbk_prefix = 'dividend/indexBonus_gbk/'
        code_dict = {"IH": "000016", "IF": "000300", "IC": "000905"}

        print(self.dt)
        if exe_date:
            exe_dt = datetime.datetime.strptime(str(exe_date), "%Y%m%d")

        summary_dict = {}
        to_do_list = ['IH', 'IF', 'IC']
        for index_type in to_do_list:
            print(index_type, 'is processing...')
            print('*********************************************************')
            # regularly use RDF to get constituent df
            index_constituent_div_df = self.constituent_div_prediction(self.dt, index_type)
            if not exe_date:
                pass
            else:
                before_exe_div_df = index_constituent_div_df.loc[index_constituent_div_df['派息日'] < exe_dt, :]
                index_constituent_div_df = self.constituent_div_prediction(self.dt, index_type, mode='csv')
                after_exe_div_df = index_constituent_div_df.loc[(index_constituent_div_df['派息日'] >= exe_dt) |
                                                                (index_constituent_div_df['派息日'].isna()), :]
                after_exe_div_df.to_csv(index_type + '_after.csv', encoding='gbk')
                index_constituent_div_df = pd.concat([before_exe_div_df, after_exe_div_df], ignore_index=True)
            filename = code_dict[index_type] + 'bonus' + self.last_trading_date.strftime('%Y%m%d') + '.csv'
            index_constituent_div_df.to_csv(local_prefix + 'raw_utf8/' + filename, encoding='utf-8')
            index_constituent_div_df.to_csv(local_prefix + 'raw_gbk/' + filename, encoding='gbk')

            # 修订
            modified_df = self.modify(index_constituent_div_df, index_type)
            modified_df.to_csv(local_prefix + 'modified_utf8/' + filename, encoding='utf-8')
            self.ftp_client.upload_file(remote_utf8_prefix + filename, local_prefix + 'modified_utf8/' + filename)
            modified_df.to_csv(local_prefix + 'modified_gbk/' + filename, encoding='gbk')
            self.ftp_client.upload_file(remote_gbk_prefix + filename, local_prefix + 'modified_gbk/' + filename)

            # 编制summary
            summary_dict[index_type] = self.get_summary_date_dict(modified_df)

        summary_df = pd.DataFrame(summary_dict)
        summary_df = summary_df.loc[:, ['IH', 'IF', 'IC']]
        summary_df.index = summary_df.index.strftime('%Y-%m-%d')
        summary_filename = 'bonusSummary' + self.last_trading_date.strftime('%Y%m%d') + '.xls'
        summary_df.to_excel(local_prefix + 'summary/' + summary_filename)
        print('dividend is done!')
        print('*********************************************************')

        right_issue_dt = self.dt - relativedelta(months=2)
        right_issue_df = self.rdf.get_right_issue_df(right_issue_dt)
        filename = 'right_issue' + self.dt.strftime('%Y%m%d') + '.xls'
        writer = pd.ExcelWriter(local_prefix + 'right_issue/' + filename)
        for index_type in to_do_list:
            index_constituent_df = self.rdf.get_index_constituent_df(index_type, self.dt)
            sheet_name = index_type + '配股'
            right_issue_df.loc[right_issue_df['股票代码'].isin(index_constituent_df['constituent_code']), :] \
                .reset_index(drop=True).to_excel(writer, sheet_name=sheet_name)
        print('right issue is done!')
        writer.save()
        self.ftp_client.upload_file(remote_gbk_prefix + filename, local_prefix + 'right_issue/' + filename)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pd.set_option('display.max_columns', None)

    div = div_pred_statistic()
    dt_today = dtparser.parse(datetime.datetime.now().strftime('%Y%m%d'))
    if div.trading_calendar[div.trading_calendar == dt_today].empty:
        print('Not Trade Day...')
    else:
        div.run_daily()
