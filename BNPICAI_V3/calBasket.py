# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 12:48:45 2016

@author: Sunil.Bhandari1
"""
import pandas as pd

def calBasketFunc(towLastReb,calDate,df_asset,daysCal):
    startDate = df_asset[:calDate].index[-daysCal-21]
    df_basket_Asset = df_asset[(df_asset.index >= startDate) & (df_asset.index <= calDate)]
    df_basket = pd.Series(index = df_basket_Asset.index)
    
    for dates in df_basket.index:
        tempBasket = 0
        columnNum = 0
        for columns in df_basket_Asset:
            if dates == startDate:
                df_basket.loc[dates] = 100
            else:
                tempBasket = tempBasket + towLastReb.iloc[columnNum]*(df_basket_Asset[:dates].iloc[-1,columnNum]/df_basket_Asset[:dates].iloc[-2,columnNum])
            columnNum = columnNum +1    
        if  dates > startDate:   
            df_basket.loc[dates] = tempBasket*df_basket[:dates][-2]                           
    return df_basket
