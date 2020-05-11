# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 17:11:50 2016

@author: sunil.bhandari1
"""
import pandas as pd
import numpy as np


def covarianceCalculator(df_asset,calDate,data_incep_date):

    df_Cov_Prev =  pd.DataFrame(index = df_asset.columns,columns = df_asset.columns)
    df_Cov =  pd.DataFrame(index = df_asset.columns,columns = df_asset.columns)
    
    
    if calDate == pd.to_datetime('2013-06-14'):   
            print("running for first date")
            """initialising the initial Cov"""
            startDate = df_asset[:calDate].index[0]
            df_Cov_Initial = pd.DataFrame(index = df_asset.columns,columns = df_asset.columns)
            df_Cov_Initial.loc[:,:] = 0
            df_Cov_Initial.values[[np.arange(len(df_Cov_Prev.columns))]*2] = .01
            """asset series required for cov cal"""
            df_cov_asset =  df_asset[(df_asset.index >= startDate) & (df_asset.index <= calDate)]
            DF = np.power(.5,(1/252))
            df_cov_asset_prev = None
       
    else:
        print("inside else")
        df_Cov_Prev = pd.read_pickle('df_Cov_Prev_Serialized')
        df_cov_asset_prev = pd.read_pickle('df_cov_asset_prev_Serialized')
        df_cov_asset_prevRun = pd.read_pickle('df_cov_asset_previous_run')
        lastDate = df_cov_asset_prevRun.index[-1]
        df_cov_asset =  df_asset[(df_asset.index > lastDate) & (df_asset.index <= calDate)]
        startDate = calDate#as during the last rebal cov is being claculated till last date we have to start from cal date
    
    """looping through all the date in the asset series for covariance to get the covariane on the last date"""   
    for index in df_cov_asset.index:
        print("running for date : ",index)
        if index == startDate:
            df_Cov = df_Cov_Initial.copy()
    #        print("inside first date")
        else:
            for column in df_Cov.columns:
                for index1 in df_Cov.index:                 
                    assetrow =  df_cov_asset.loc[index,index1]/df_cov_asset_prev[index1]
                    assetCol = df_cov_asset.loc[index,column]/df_cov_asset_prev[column]
                    df_Cov.loc[index1,column] = DF*df_Cov_Prev.loc[index1,column]+(1-DF)*252*(assetrow-1)*(assetCol-1)   
    #                print("Col : ",column,"Row(index1): ",index1,"assetrow: ",assetrow,"assetCol: ",assetCol," df_Cov.loc[index1,column] :", df_Cov.loc[index1,column])
        df_Cov_Prev = df_Cov
        df_cov_asset_prev = df_cov_asset.loc[index]
     
    df_Cov_Prev.to_pickle('df_Cov_Prev_Serialized')
    df_cov_asset_prev.to_pickle('df_cov_asset_prev_Serialized')
    df_cov_asset.to_pickle('df_cov_asset_previous_run')
    return df_Cov    