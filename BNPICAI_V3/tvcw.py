# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 20:40:39 2016

@author: Sunil.Bhandari1
"""

import pandas as pd
import math as mt
import calBasket
import calVol

def tvcwFunc(Optweight,date,df_asset,days):
    startDate = df_asset[:date].index[-20]   
    df_Vol_Asset = df_asset[(df_asset.index >= startDate) & (df_asset.index <= date)]
    Volmax = 0
    VolTarget = 0.1
    for dates in df_Vol_Asset.index:
        df_basket = calBasket.calBasketFunc(Optweight,dates,df_asset,20)
    Vol = calVol.calVolFunct(df_basket)

    if Volmax < Vol:
        Volmax = Vol
    VolmaxInt = mt.ceil(Volmax*100)/100      
    if VolmaxInt < VolTarget:
        tvcw = 1
    else:
        tvcw =   VolTarget/  VolmaxInt        
    return tvcw