from __future__ import annotations

import numpy as np
import pandas as pd

#from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor as Alpaca
#from finrl.meta.data_processors.processor_wrds import WrdsProcessor as Wrds
#from finrl.meta.data_processors.processor_yahoofinance import (
#    YahooFinanceProcessor as YahooFinance,
#)
#from finrl.meta.data_processors.processor_local import LocalProcessor as Local

from finrl.meta.paper_trading.utilities import timeit, memoize, keycompute
#from diskcache import Cache

#cache = Cache('/mnt/diskcache/cs221_finrl', EVICTION_POLICY='none')

class DataProcessor:
    # Alfred implemented hash function to help with memoization
    def __hash__(self):
        # go through all self variables and compute the hash
        return hash(self.processor)
    #@cache.memoize()
    def __init__(self, data_source, tech_indicator=None, vix=None, **kwargs):
        a=2
        #if data_source == "alpaca":
        #    try:
        #        API_KEY = kwargs.get("API_KEY")
        #        API_SECRET = kwargs.get("API_SECRET")
        #        API_BASE_URL = kwargs.get("API_BASE_URL")
        #        self.processor = Alpaca(API_KEY, API_SECRET, API_BASE_URL)
        #        print("Alpaca successfully connected")
        #    except BaseException:
        #        raise ValueError("Please input correct account info for alpaca!")

        #elif data_source == "wrds":
        #    self.processor = Wrds()

        #elif data_source == "yahoofinance":
        #    self.processor = YahooFinance(start_date, end_date, time_interval)

        #elif data_source == "local":
        #    self.processor = Local()

        #else:
        #    raise ValueError("Data source input is NOT supported yet.")

        # Initialize variable in case it is using cache and does not use download_data() method
        #self.tech_indicator_list = list(tech_indicator)
        #self.vix = vix


    @timeit
    #@memoize()
    def download_data(
        self, ticker_list, start_date, end_date, time_interval
    ) -> pd.DataFrame:
        df = self.processor.download_data(
            ticker_list=ticker_list,
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval,
        )
        return df

    @timeit
    #@cache.memoize(hash=keycompute)
    #@memoize()
    # Alfred needs time_interval to properly clean and memoize
    def clean_data(self, df, start, end, time_interval) -> pd.DataFrame:
        df = self.processor.clean_data(df, start, end, time_interval)

        return df

    @timeit
    #@cache.memoize(hash=keycompute)
    #@memoize()
    def add_technical_indicator(self, df, tech_indicator_list) -> pd.DataFrame:
        self.tech_indicator_list = list(tech_indicator_list)
        df = self.processor.add_technical_indicator(df, tech_indicator_list)

        return df

    @timeit
    #@cache.memoize(hash=keycompute)
    #@memoize()
    def add_turbulence(self, df) -> pd.DataFrame:
        df = self.processor.add_turbulence(df)

        return df

    @timeit
    #@cache.memoize(hash=keycompute)
    #@memoize()
    def add_vix(self, df, start, end, time_interval) -> pd.DataFrame:
        df = self.processor.add_vix(df, start, end, time_interval)

        return df

    @timeit
    #@cache.memoize(hash=keycompute)
    #@memoize()
    def add_turbulence(self, df) -> pd.DataFrame:
        df = self.processor.add_turbulence(df)

        return df


    def add_vixor(self, df) -> pd.DataFrame:
        df = self.processor.add_vixor(df)

        return df

    def df_to_array(self, df) -> np.array:
        price_array, tech_array = self.processor.df_to_array(
            df, self.tech_indicator_list
        )
        # fill nan and inf values with 0 for technical indicators
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0
        tech_inf_positions = np.isinf(tech_array)
        tech_array[tech_inf_positions] = 0

        return price_array, tech_array
