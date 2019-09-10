from WindPy import w
import numpy as np
import pandas as pd
import dateutil.parser as dtparser

class windAPI_data:
    def __init__(self):
        w.start()

    def date_preprocess(self, date_input):
        if isinstance(date_input, str):
            str_date = dtparser.parse(date_input).strftime("%Y-%m-%d")
        elif isinstance(date_input, int):
            str_date = dtparser.parse(str(date_input)).strftime("%Y-%m-%d")
        elif isinstance(date_input, datetime.datetime):
            str_date = date_input.strftime("%Y-%m-%d")
        return str_date

    def get_index_futures_min_data(self,symbol,start_input,end_input):
        w.wsi(symbol,self.date_preprocess(start_input),self.date_preprocess(end_input))
