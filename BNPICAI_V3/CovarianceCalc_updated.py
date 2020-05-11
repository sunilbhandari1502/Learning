# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 12:35:34 2016

@author: sunilbhandari1
"""

import pandas as pd
import numpy as np
import datetime as dt
import pickle as pic

def covarianceCalculator(df_asset,calDate,data_incep_date):
    
    df_Cov_Prev =  pd.DataFrame(index = df_asset.columns,columns = df_asset.columns)
    df_Cov =  pd.DataFrame(index = df_asset.columns,columns = df_asset.columns)
    DF = np.power(.5,(1/252))
     
    try: 
        df_Cov_Prev = pd.read_pickle('df_cov_prev_Serialized')        
        with open('calDate.pickle', 'rb') as handle:
            dt_Cov_Date_pickled = pd.to_datetime(pic.load(handle))
            pre_dates = dt_Cov_Date_pickled    
        print("inside try")
        print("dt_Cov_Date_pickled ",dt_Cov_Date_pickled)
        
        for dates in df_asset.index[(df_asset.index <= calDate) & (df_asset.index> dt_Cov_Date_pickled)]:
           print("Calculating for date ",dates)                  
           for column in df_Cov.columns:
                for index in df_Cov.index:                 
                    assetrow = df_asset.loc[dates,index]/df_asset.loc[pre_dates,index]
                    assetCol = df_asset.loc[dates,column]/df_asset.loc[pre_dates,column]
                    df_Cov.loc[index,column] = DF*df_Cov_Prev.loc[index,column]+(1-DF)*252*(assetrow-1)*(assetCol-1)
           pre_dates = dates
           df_Cov_Prev = df_Cov           
    
                   
    except:
        print("inside except")
        df_Cov_Prev.loc[:,:] = 0
        df_Cov_Prev.values[[np.arange(len(df_Cov_Prev.columns))]*2] = .01
        pre_dates = data_incep_date
        for dates in df_asset.index[(df_asset.index <= calDate) & (df_asset.index > pre_dates)]:
            print("Running for Date ",dates)          
#            print("pre_dates ",pre_dates)             
            for column in df_Cov.columns:
                for index in df_Cov.index: 
                    assetrow = df_asset.loc[dates,index]/df_asset.loc[pre_dates,index]
                    assetCol = df_asset.loc[dates,column]/df_asset.loc[pre_dates,column]
#                    if dates == pd.to_datetime('2013-05-09'):
#                        print("Prev Value",df_Cov_Prev.loc[index,column])
#                        print("assetrow ",assetrow)
#                        print("assetCol ",assetCol)

                    df_Cov.loc[index,column] = DF*df_Cov_Prev.loc[index,column] + (1-DF)*252*(assetrow-1)*(assetCol-1)
            pre_dates = dates
            df_Cov_Prev = df_Cov
                    
    df_Cov_Prev.to_pickle('df_Cov_Prev_Serialized')
#    pd.to_datetime(calDate).to_pickle('dt_cov_date_Serialized')
    with open('calDate.pickle', 'wb') as handle:
        pic.dump(dt.datetime.date(calDate), handle)

  
    return df_Cov  
        
