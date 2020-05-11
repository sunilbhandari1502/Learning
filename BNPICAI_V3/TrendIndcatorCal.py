# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 00:48:59 2016

@author: sunilbhandari1
"""

def trndCalculator(calDate,df_asset):
    startDate = df_asset[:calDate].index[0]
    df_trend_asset =  df_asset[(df_asset.index >= startDate) & (df_asset.index <= calDate)]
    no_Observation = len(df_trend_asset.index)
    if no_Observation > 252:
       no_Observation = 252
       startDate = df_asset[:calDate].index[-252]
       df_trend_asset =  df_asset[(df_asset.index >= startDate) & (df_asset.index <= calDate)]
    #print("Calculating for date startDate",startDate," Cal Date ",calDate," no_Observation ",no_Observation ) 
    def f(x):
      return sum(x <= (x[-1]+0.00000000000001))    
    df_trend = df_trend_asset.apply(f, axis=0)/no_Observation
    #print("Trend Indicator ", df_trend)
    return df_trend
