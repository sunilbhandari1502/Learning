# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 20:49:18 2016

@author: jasmeet.gujral
"""
import pandas as pd

"""Reading data from Excel"""
def read_data(data_incep_date, end_date):
    """Reading price and div data"""
    input_file_path = "C:\\Users\\sunilbhandari1\\Desktop\\Python\\BNPICAI_V3\\static_data\\Input_Data_Client_v2.xlsx"
    input_file = pd.ExcelFile(input_file_path)
    df_input_px = input_file.parse("Final_Input").set_index("Date")
    df_input_px = df_input_px[(df_input_px.index>=data_incep_date) & (df_input_px.index<=end_date)]
    df_input_div = input_file.parse("Div").set_index("Date")
    df_trading_days = input_file.parse("Trading_days")["Trading_days"]
    sr_trading_days = pd.Series(df_trading_days[(df_trading_days>=data_incep_date) & (df_trading_days<=end_date)])    
    
    """Reading view matrix"""
    input_file = pd.ExcelFile("C:\\Users\\sunilbhandari1\\Desktop\\Python\\BNPICAI_V3\\static_data\\Blackrock.xlsx")
    df_reg_factor = input_file.parse("macro views").set_index("Category")
    
    """Rule bookk details"""
    des_file = pd.ExcelFile("C:\\Users\\sunilbhandari1\\Desktop\\Python\\BNPICAI_V3\\static_data\\Comp_Des.xlsx")
    df_descrip = des_file.parse("Descrip").set_index("BBG")
    """reading tax details"""
    df_tax = des_file.parse("Tax").set_index("Domicile")
         
    return df_input_px, df_input_div, df_reg_factor, df_descrip, df_tax, sr_trading_days
    
    
    