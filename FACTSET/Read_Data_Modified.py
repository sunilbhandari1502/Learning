# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 22:18:50 2016

@author: Jasmeet.Gujral
"""

import pandas as pd

"""Reading Px and Des data from Excel stored under Static_data folder"""

def read_data(incep_date,end_date):
    
    """ Reading Px data from excle file"""
    i_path = "C:\\Users\\sunilbhandari1\\Desktop\\Python\FACTSET\\Staticdata\\Input_Px.csv"
    df_input_px = pd.read_csv(i_path, header = 0, index_col = 0)    
    df_input_px.index = pd.to_datetime(df_input_px.index)
    df_input_px = df_input_px[(df_input_px.index>=incep_date) & (df_input_px.index<=end_date)]
    
    input_file_path = "C:\\Users\\sunilbhandari1\\Desktop\\Python\\FACTSET\\Staticdata\\Ticker Mapping.xlsx"
    input_file = pd.ExcelFile(input_file_path)
#    df_input_px = input_file.parse("Px_Clean").set_index("Date")
    """ Reading Ticker mapping"""
    df_mapping = input_file.parse("Sheet1").set_index("SEDOL")    
    
    """Reading Rebalancing file"""
    rebal_details_path = "C:\\Users\\sunilbhandari1\\Desktop\\Python\\FACTSET\\Staticdata\\FinTech_Const.xlsx"
    rfile = pd.ExcelFile(rebal_details_path)    
    df_rebal_details = rfile.parse("Index Constituents").set_index("Rebal Date")
    
    """Reading CA file"""
    input_CA_path = "C:\\Users\\sunilbhandari1\\Desktop\\Python\\FACTSET\\Staticdata\\Input_CA.xlsx"
    input_CA_file = pd.ExcelFile(input_CA_path)    
    df_input_CD = input_CA_file.parse("CASH_details")
    df_input_CAo = input_CA_file.parse("Others")
    return df_input_px, df_mapping, df_rebal_details, df_input_CD, df_input_CAo