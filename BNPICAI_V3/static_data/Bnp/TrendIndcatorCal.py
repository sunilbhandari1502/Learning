# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 00:48:59 2016

@author: sunilbhandari1
"""
def trndCalculator(calDate,df_input_clean):
    startDate = df_input_clean[:calDate].index[-252]
    df_trend_asset =  df_input_clean[(df_input_clean.index >= startDate) & (df_input_clean.index <= calDate)]
    def f(x):
        return sum(x<x[-1])
    
    df_trend_asset = df_trend_asset.apply(f, axis=0)/252    

    return df_trend_asset
