# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 17:05:55 2016

@author: Sunil.Bhandari1
"""
import pandas as pd
import numpy as np

def calVolFunct(df_basket):

    df_basket_return = np.log(np.divide(df_basket.iloc[1:],df_basket.iloc[:-1]))
    df_vol = pd.Series()
    for dates in df_basket.index[21:]:
        df_vol.loc[dates] = np.sqrt(252)*np.std(df_basket_return[:dates][-20:])
#        Vol1 = Vol1 + mt.log(mt.pow((df_basket[:dates][-1]/df_basket[:dates][-2]),2))
#        Vol2 = Vol2 + mt.log(df_basket[:dates][-1]/df_basket[:dates][-2])
#    Vol = mt.pow(252*((Vol1/Vper)-mt.pow(Vol2/Vper,2)),0.5)
    Vol = np.max(df_vol)    
    return Vol    
