# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 23:54:03 2016

@author: Sunil.Bhandari1
"""

def regFactor(calDate,df_reg_factor):
    print("inside regFactor")
    import pandas as pd
    import numpy as np

    df_reg_factor = df_reg_factor.replace(['neutral', 'underweight', 'overweight'], 
                     [1, 0.5, 1.5])
    columnDate = np.sum(df_reg_factor.columns.values <= calDate)-1

    regionalFac = np.array([df_reg_factor.iloc[21-1,columnDate],
                            df_reg_factor.iloc[21-1,columnDate],
                            .45*df_reg_factor.iloc[23-1,columnDate]+.55*df_reg_factor.iloc[24-1,columnDate],
                            .53*df_reg_factor.iloc[23-1,columnDate]+.47*df_reg_factor.iloc[24-1,columnDate],
                            .75*df_reg_factor.iloc[23-1,columnDate]+.25*df_reg_factor.iloc[24-1,columnDate],
                            df_reg_factor.iloc[17-1,columnDate],
                            df_reg_factor.iloc[26-1,columnDate],
                            df_reg_factor.iloc[16-1,columnDate],
                            df_reg_factor.iloc[25-1,columnDate],
                            df_reg_factor.iloc[15-1,columnDate],  
                            df_reg_factor.iloc[15-1,columnDate],
                            df_reg_factor.iloc[27-1,columnDate],     
                            df_reg_factor.iloc[20-1,columnDate],                         
                            df_reg_factor.iloc[1-1,columnDate],
                            .54*df_reg_factor.iloc[3-1,columnDate]+.3*df_reg_factor.iloc[5-1,columnDate]+.16*df_reg_factor.iloc[4-1,columnDate],  
                            df_reg_factor.iloc[6-1,columnDate],  
                            df_reg_factor.iloc[7-1,columnDate],
                            .7*df_reg_factor.iloc[11-1,columnDate]+.3*df_reg_factor.iloc[12-1,columnDate],
                            df_reg_factor.iloc[8-1,columnDate],    
                            df_reg_factor.iloc[10-1,columnDate],
                            df_reg_factor.iloc[9-1,columnDate],
                            df_reg_factor.iloc[14-1,columnDate],
                            ])
    return (regionalFac)