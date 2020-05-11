# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 23:54:03 2016

@author: Sunil.Bhandari1
"""

def regFactor(calDate):
    print("inside regFactor")
    import pandas as pd
    import numpy as np
    """ Reading data from Input_data file"""
    input_file = pd.ExcelFile("C:/Users/sunilbhandari1/Documents/Python Scripts/Bnp/Blackrock.xlsx")
    df_reg_factor = input_file.parse("macro views").set_index(['Category'])
    df_reg_factor = df_reg_factor.replace(['neutral', 'underweight', 'overweight'], 
                     [1, 0.5, 1.5])
    columnDate = np.sum(df_reg_factor.columns.values <= calDate)
    
    regionalFac = np.array([df_reg_factor.iloc[21,columnDate],
                            df_reg_factor.iloc[21,columnDate],
                            .45*df_reg_factor.iloc[23,columnDate]+.55*df_reg_factor.iloc[24,columnDate],
                            .53*df_reg_factor.iloc[23,columnDate]+.47*df_reg_factor.iloc[24,columnDate],
                            .75*df_reg_factor.iloc[23,columnDate]+.25*df_reg_factor.iloc[24,columnDate],
                            df_reg_factor.iloc[17,columnDate],
                            df_reg_factor.iloc[26,columnDate],
                            df_reg_factor.iloc[16,columnDate],
                            df_reg_factor.iloc[25,columnDate],
                            df_reg_factor.iloc[15,columnDate],  
                            df_reg_factor.iloc[15,columnDate],
                            df_reg_factor.iloc[27,columnDate],     
                            df_reg_factor.iloc[20,columnDate],                         
                            df_reg_factor.iloc[1,columnDate],
                            .54*df_reg_factor.iloc[3,columnDate]+.3*df_reg_factor.iloc[17,columnDate]+.16*df_reg_factor.iloc[17,columnDate],  
                            df_reg_factor.iloc[6,columnDate],  
                            df_reg_factor.iloc[7,columnDate],
                            .7*df_reg_factor.iloc[11,columnDate]+.3*df_reg_factor.iloc[12,columnDate],
                            df_reg_factor.iloc[8,columnDate],    
                            df_reg_factor.iloc[10,columnDate],
                            df_reg_factor.iloc[9,columnDate],
                            df_reg_factor.iloc[14,columnDate],
                            ])
    return (regionalFac)