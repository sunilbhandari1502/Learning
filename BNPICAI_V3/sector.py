# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 15:36:44 2017

@author: siddhant.madan
"""
import pandas as pd
def sector(x,y):
    sector_weight = 0
    path_sectors = "C:\\Users\\sunil.bhandari1\\Desktop\\Python\\BNPICAI_V3\\static_data\\input_sectors.xlsx"
    sector_file = pd.ExcelFile(path_sectors)
    sector_data = sector_file.parse("Sheet1")
    sector_data["weights"] = x
    print (sector_data)

    for  item in range(len(sector_data)):
        
        if sector_data.iloc[item]["Sector"] == y:
            sector_weight = sector_weight + sector_data.iloc[item]["weights"]
    return sector_weight