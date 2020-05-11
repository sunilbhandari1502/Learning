# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 12:47:20 2017

@author: siddhant.madan
"""
import pandas as pd
import numpy as np

def sector_capping_function(w):
    
    path_capping="C:\\Users\\sunilbhandari1\\Desktop\\Python\\BNPICAI_V3\\static_data\\Input_Files_Cap.xlsx"
    sector_file = pd.ExcelFile(path_capping)
    capping_data = sector_file.parse("Optimization")

    """Stock Table"""
    stock_table=capping_data.loc[:,:"Stock Max"]

    """Sector Table"""
    sector_table=capping_data.loc[:,"Sector Unique List":"Max_Sec"]
    sector_table=sector_table.dropna()

    """Geography Table"""
    geography_table=capping_data.loc[:,"Geography Unique List":"Max_Geo"]
    geography_table=geography_table.dropna()

    """
    w=[0.015385,.015385,0.015385,0.015385,0.015385,0.015385,0.015385,0.015385,0.015385,.023077,.007692,.015385,.015385,.167373,.073314,.024147,.013729,.005269,.00563,.005118,.002619,.0028]
    """
    
    stock_table["weights"] = w

    sector_maximum=capping_data['Max_Sec']
    sector_maximum=sector_maximum.dropna()
   
    
    """group_sector_maximum=sector_maximum["Max_Sec"].groupby(sector_maximum["Sector Unique List"])
    group_sector_maximum["Sector Unique List"]"""

    group_sector=stock_table["weights"].groupby(stock_table["Sector"]).sum()
    
    sector_constraint=sector_maximum-group_sector.values
    
    return sector_constraint.as_matrix()
    