# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 21:55:08 2018

@author: sunilbhandari1
"""

import pandas as pd

#Read instrument code file
input_file_path = ("C:\\Users\\sunilbhandari1\\Desktop\\Kiteconnect\\Code\\instrumentsId.xlsx")
input_file = pd.ExcelFile(input_file_path)
df_instruCode = input_file.parse("InstrumentsId")

#Set property for history data
month = 'DEC'
Strike = 9900
optionType = 'CE'
Year = 18
instrumentName = 'NIFTY'+str(Year)+month+str(Strike)+optionType     #18DEC10250PE
instrument_token = df_instruCode[df_instruCode['tradingsymbol']==instrumentName]
instrument_token = instrument_token.iloc[0][0]
from_date = '2018-11-30'
to_date = '2018-12-20'
interval='minute'  #3minute
continuous = False
TickData_Nifty50 = kite.historical_data(instrument_token='256265',from_date=from_date,to_date=to_date,interval=interval,continuous=continuous)
TickData_Nifty50 = pd.DataFrame(TickDataCall_9900)
TickDataCall_9900.to_csv('TickDataPut_11900.csv')
#Set property for history data
month = 'DEC'
Strike = 11900
optionType = 'PE'
Year = 18
instrumentName = 'NIFTY'+str(Year)+month+str(Strike)+optionType     #18DEC10250PE
instrument_token = df_instruCode[df_instruCode['tradingsymbol']==instrumentName]
instrument_token = instrument_token.iloc[0][0]
from_date = '2018-11-30'
to_date = '2018-12-20'
interval='minute'
continuous = False
TickDataPut_11900 = kite.historical_data(instrument_token=instrument_token,from_date=from_date,to_date=to_date,interval=interval,continuous=continuous)

#Writing to CSV
pd.DataFrame(TickDataPut_11900).to_csv('TickDataPut_11900.csv')