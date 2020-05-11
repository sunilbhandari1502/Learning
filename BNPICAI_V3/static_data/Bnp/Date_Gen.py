# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 22:28:35 2016

@author: jasmeet.gujral
"""
import pandas as pd

def date_series(df_input_px, incep_date, end_date):
    """index Publishing dates"""
    index_calc_days = pd.Series(df_input_px.index)


    """Monthly rebalance and reset dates"""
    wom = pd.datetools.WeekOfMonth(week=2,weekday=4)
    theo_rebal_dates = pd.Series(pd.date_range(start=incep_date, end=end_date, freq = wom))
    index_rebal_dates = pd.Series(incep_date).append(theo_rebal_dates.apply(lambda x: index_calc_days[index_calc_days<=x].iloc[-1]))

    return index_rebal_dates