# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 22:28:35 2016

@author: jasmeet.gujral
"""
import pandas as pd
import numpy as np

def date_series(df_input_px,sr_trading_days, data_incep_date, end_date):
    """index Publishing dates"""
    index_calc_days = pd.Series(df_input_px.index)
    
    """Monthly rebalance and reset dates"""
    wom = pd.datetools.WeekOfMonth(week=1,weekday=2)
    theo_dates = pd.Series(pd.date_range(start=data_incep_date, end=end_date, freq = wom))
    comp_dates = theo_dates.apply(lambda x: sr_trading_days[sr_trading_days>x].iloc[1])

    return index_calc_days, comp_dates