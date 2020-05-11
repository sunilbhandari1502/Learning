# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:26:18 2017

@author: garima.madan
"""

import pickle as pic
import pandas as pd
import numpy as np
import datetime as dt
import win32com.client as win32   
import os

data = {}

"""Reading Pickeled file"""

p_name = 'C:/Users/garima.madan/Desktop/FACTSET_Fintech_Index/Pickeled_File.pkl'
fileObject = open(p_name,'rb')
data = pic.load(fileObject)

"""Reading Mapping file"""

mapping_path = "C:/Users/garima.madan/Desktop/FACTSET_Fintech_Index/Staticdata/Ticker Mapping.xlsx"
input_file = pd.ExcelFile(mapping_path)
df_mapping = input_file.parse("Sheet1").set_index("SEDOL") 


"""Reading Rebalancing file"""
rebal_details_path = "C:/Users/garima.madan/Desktop/FACTSET_Fintech_Index/Staticdata/FinTech_Const.xlsx"
rfile = pd.ExcelFile(rebal_details_path)    
df_rebal_details = rfile.parse("Index Constituents").set_index("Rebal Date")
    
"""Reading CA file"""
input_CA_path = "C:/Users/garima.madan/Desktop/FACTSET_Fintech_Index/Staticdata/Input_CA.xlsx"
input_CA_file = pd.ExcelFile(input_CA_path)    
df_input_CD = input_CA_file.parse("CASH_details")
df_input_CAo = input_CA_file.parse("Others")


"""Conversion methodology"""

if len(np.unique(df_mapping["Ccy_mapping"])) > len(data["Conversion_methodology"]):
    for curncy in np.unique(df_mapping["Currency"]):
        if curncy not in data["Conversion_methodology"].index:
            if curncy[-1].islower():
                data["Conversion_methodology"].loc[curncy,"Factor"] = 100
            else:
                data["Conversion_methodology"].loc[curncy,"Factor"] = 1

            if data["Conversion_methodology"].loc[curncy,"Ticker"][:3] == "USD":
                data["Conversion_methodology"].loc[curncy,"Methodology"] = "inverse"
            else:
                df_rebal_details.loc[curncy,"Methodology"] = "multiply"

        
"""Closing_date"""

last_closing = data["Divisor_Series"].index[-2]

current_date = data["Divisor_Series"].index[-1]

next_opendate = current_date + dt.timedelta(1)

if (next_opendate.weekday() == 5):
    next_opendate = next_opendate + dt.timedelta(2)
elif (next_opendate.weekday() == 6):
    next_opendate = next_opendate + dt.timedelta(1)

file_date = current_date + dt.timedelta(1)

#date_str = str(file_date.day) + "_" + str(file_date.month) + "_" + str(file_date.year)


"""Next opendate holiday check"""

h_path = "C:/Users/garima.madan/Desktop/FACTSET_Fintech_Index/Staticdata/Holiday_List.xlsx"
holidays_file = pd.ExcelFile(h_path) 
df_holidays = holidays_file.parse("Holiday_List")

relevant_exchange = []

#for ticker in prices.index:
#    if prices.loc[ticker, "Close"] != 0:
#        exchange = np.asscalar(df_mapping.loc[df_mapping["RIC"]==ticker,"Exchange"].values)
#        relevant_exchange.append(exchange)    

for ticker in data["Adj_units"].columns:
    if data["Adj_units"].loc[current_date, ticker] != 0 and ~(np.isnan(data["Adj_units"].loc[current_date, ticker])):
        exchange = np.asscalar(df_mapping.loc[df_mapping["RIC"]==ticker,"Exchange"].values)
        relevant_exchange.append(exchange)    

relevant_exchange = list(set(relevant_exchange))

df_rel_hol = df_holidays.loc[:,relevant_exchange]

holiday = []

col_len = pd.DataFrame()

for columns in df_rel_hol.columns:
    col_len.loc[columns,"Col_len"] = df_rel_hol[columns].count()
    
min_len = min(col_len["Col_len"])

rel_index = []

for index in col_len.index:
    if col_len.loc[index,"Col_len"] == min_len:
        rel_index.append(index)
   
holiday = df_rel_hol.loc[:,rel_index[0]]

holiday = holiday.dropna()

for i in rel_index:
    holiday = list(set(holiday).intersection(set(df_rel_hol.loc[:,i])))
    
    
#    holiday = set(holiday).intersection(df_rel_hol.iloc[:,i])

for columns in df_rel_hol.columns:
    holiday = list(set(holiday).intersection(set(df_rel_hol.loc[:,columns])))        

    
while (next_opendate in holiday) or ((next_opendate.weekday() == 5) or (next_opendate.weekday() == 6)):
    next_opendate = next_opendate + dt.timedelta(1)


    
"""Reading daily prices and FX Rate"""

i_path = "C:/Users/garima.madan/Desktop/FACTSET_Fintech_Index/Staticdata/New Data" + file_date.to_pydatetime().strftime('%d_%m_%Y') + "/Prices" + file_date.to_pydatetime().strftime('%d_%m_%Y')  +".csv"
prices = pd.read_csv(i_path, header = 0, index_col = 0)
    
    
for index in prices.index:
    if prices.loc[index, "Close"] == 0:
       prices.loc[index, "Close"] = data["df_input_px"].loc[last_closing, index]

for index in prices.index:
    data["df_input_px"].loc[current_date, index] = prices.loc[index,"Close"]

fx_path = "C:/Users/garima.madan/Desktop/FACTSET_Fintech_Index/Staticdata/New Data" + file_date.to_pydatetime().strftime('%d_%m_%Y')  + "/WMCO rates_FactSet_" + file_date.to_pydatetime().strftime('%d_%m_%Y')  + ".xlsx"
input_file = pd.ExcelFile(fx_path)
fx_rate = input_file.parse("FactSet").set_index("Tickers")

fx_rate.loc["USDUSD WMCO Curncy","Spot Rate"] = 1

rebal_dates = df_rebal_details.index.drop_duplicates()
prerebaldate = rebal_dates[rebal_dates<current_date][-1]
preprerebaldate = prerebaldate


"""Business day check"""

if np.nansum(data["df_input_px"].loc[current_date,:][df_mapping["RIC"][df_mapping.index.isin(df_rebal_details["SEDOL-CHK"][df_rebal_details.index==prerebaldate])]])==0:
    data["df_input_px"] = data["df_input_px"][data["df_input_px"].index != current_date]
    
else:
    data["df_input_px"] = data["df_input_px"].fillna(method ='pad')


        

df_px_fx_adj = pd.DataFrame(columns = df_mapping["RIC"])



for column in df_px_fx_adj.columns:
    crncy = df_mapping["Currency"][df_mapping["RIC"]==column]
    mthd = data["Conversion_methodology"].loc[crncy,:]

    if mthd.iloc[0,1] == "inverse":
        df_px_fx_adj.loc[current_date,column] =  np.asscalar(data["df_input_px"].loc[current_date,column] / fx_rate.loc[mthd["Ticker"],"Spot Rate"].values/mthd.iloc[0,2])
    elif mthd.iloc[0,1] == "multiply":
        df_px_fx_adj.loc[current_date,column] =  np.asscalar(data["df_input_px"].loc[current_date,column] * fx_rate.loc[mthd["Ticker"],"Spot Rate"].values/mthd.iloc[0,2])



""" CA adjustment function"""


def CA_adjust(current_date, next_opendate, prerebaldate):
    global df_input_CAo_required1
    global df_mapping
    global df_rebal_details
    global data
    
    
    
    for columns in df_rebal_details["SEDOL-CHK"][df_rebal_details.index == prerebaldate]:
        
                if (df_mapping.loc[columns,"RIC"] in np.array(df_input_CAo_required1["RIC"])):
                    df_input_CAo_required = df_input_CAo_required1[df_input_CAo_required1["RIC"]==df_mapping.loc[columns,"RIC"]]
                    
                    if not len(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Delisting"]) == 0:
                        data["Adj_units"].loc[next_opendate,df_mapping.loc[columns,"RIC"]] = 0
                        
                    if not len(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Cash Dividend"]) == 0:
                        #Get Currency of CA
                        curncy = df_input_CAo_required[df_input_CAo_required["Action Type"]=="Cash Dividend"]["Currency"]
                        mthd = data["Conversion_methodology"].loc[curncy,:]
                        if mthd.iloc[0,1] == "inverse":
                            data["Adj_px"].loc[next_opendate,df_mapping.loc[columns,"RIC"]] = np.asscalar(data["Adj_px"].loc[next_opendate,df_mapping.loc[columns,"RIC"]] - (df_input_CAo_required[df_input_CAo_required["Action Type"]=="Cash Dividend"]["Gross Amount"] / fx_rate.loc[mthd["Ticker"],"Spot Rate"].values/mthd.iloc[0,2]).values)
                        elif mthd.iloc[0,1] == "multiply":
                            data["Adj_px"].loc[next_opendate,df_mapping.loc[columns,"RIC"]] = np.asscalar(data["Adj_px"].loc[next_opendate,df_mapping.loc[columns,"RIC"]] - (df_input_CAo_required[df_input_CAo_required["Action Type"]=="Cash Dividend"]["Gross Amount"] * fx_rate.loc[mthd["Ticker"],"Spot Rate"].values/mthd.iloc[0,2]).values)

                    
                    if not len(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Stock Dividend"]) == 0:
                        data["Adj_px"].loc[next_opendate,df_mapping.loc[columns,"RIC"]] = data["Adj_px"].loc[next_opendate,df_mapping.loc[columns,"RIC"]]/np.array(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Stock Dividend"]["Gross Amount"])
                        data["Adj_units"].loc[next_opendate,df_mapping.loc[columns,"RIC"]] = data["Adj_units"].loc[next_opendate,df_mapping.loc[columns,"RIC"]]*np.array(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Stock Dividend"]["Gross Amount"])
                    
                    if not len(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Stock Split"]) == 0:
                        data["Adj_px"].loc[next_opendate,df_mapping.loc[columns,"RIC"]] = data["Adj_px"].loc[next_opendate,df_mapping.loc[columns,"RIC"]]/np.array(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Stock Split"]["Gross Amount"])
                        data["Adj_units"].loc[next_opendate,df_mapping.loc[columns,"RIC"]] = data["Adj_units"].loc[next_opendate,df_mapping.loc[columns,"RIC"]]*np.array(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Stock Split"]["Gross Amount"])
                    
                    if not len(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Rights Offerings"]) == 0:
                        curncy = df_input_CAo_required[df_input_CAo_required["Action Type"]=="Rights Offerings"]["Currency"]
                        mthd = data["Conversion_methodology"].loc[curncy,:]                                                  
                        if mthd.iloc[0,1] == "inverse":
                            data["Adj_px"].loc[next_opendate,df_mapping.loc[columns,"RIC"]] = float(np.array((data["Adj_px"].loc[next_opendate,df_mapping.loc[columns,"RIC"]]+(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Rights Offerings"]["Subscription Price"]/mthd.iloc[0,2]/fx_rate.loc[mthd["Ticker"],"Spot Rate"].values*df_input_CAo_required[df_input_CAo_required["Action Type"]=="Rights Offerings"]["Gross Amount"]))/(1+df_input_CAo_required[df_input_CAo_required["Action Type"]=="Rights Offerings"]["Gross Amount"])))
                        elif mthd.iloc[0,1] == "multiply":
                            data["Adj_px"].loc[next_opendate,df_mapping.loc[columns,"RIC"]] = float(np.array((data["Adj_px"].loc[next_opendate,df_mapping.loc[columns,"RIC"]]+(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Rights Offerings"]["Subscription Price"]/mthd.iloc[0,2]*fx_rate.loc[mthd["Ticker"],"Spot Rate"].values*df_input_CAo_required[df_input_CAo_required["Action Type"]=="Rights Offerings"]["Gross Amount"]))/(1+df_input_CAo_required[df_input_CAo_required["Action Type"]=="Rights Offerings"]["Gross Amount"])))
                        
                        data["Adj_units"].loc[next_opendate,df_mapping.loc[columns,"RIC"]] = float(np.array(data["Adj_units"].loc[next_opendate,df_mapping.loc[columns,"RIC"]]*(1+df_input_CAo_required[df_input_CAo_required["Action Type"]=="Rights Offerings"]["Gross Amount"])))
                    
                    if not len(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Spin-off"]) == 0:
                        curncy = df_input_CAo_required[df_input_CAo_required["Action Type"]=="Spin-off"]["Currency"]
                        mthd = data["Conversion_methodology"].loc[curncy,:]                                                  
                        if mthd.iloc[0,1] == "inverse":
                            data["Adj_px"].loc[next_opendate,df_mapping.loc[columns,"RIC"]] = float(np.array((data["Adj_px"].loc[next_opendate,df_mapping.loc[columns,"RIC"]]-(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Spin-off"]["Subscription Price"]*df_input_CAo_required[df_input_CAo_required["Action Type"]=="Spin-off"]["Gross Amount"] / fx_rate.loc[mthd["Ticker"],"Spot Rate"].values /mthd.iloc[0,2]))))    
                        elif mthd.iloc[0,1] == "multiply":
                            data["Adj_px"].loc[next_opendate,df_mapping.loc[columns,"RIC"]] = float(np.array((data["Adj_px"].loc[next_opendate,df_mapping.loc[columns,"RIC"]]-(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Spin-off"]["Subscription Price"]*df_input_CAo_required[df_input_CAo_required["Action Type"]=="Spin-off"]["Gross Amount"] * fx_rate.loc[mthd["Ticker"],"Spot Rate"].values /mthd.iloc[0,2]))))

"""Function end"""


sumpdt_px_units = 0

for columns in df_rebal_details["SEDOL-CHK"][df_rebal_details.index == prerebaldate]:
    sumpdt_px_units += data["Adj_units"].loc[current_date,df_mapping.loc[columns,"RIC"]]*df_px_fx_adj.loc[current_date,df_mapping.loc[columns,"RIC"]]

                        
                        
data["Index_Series"][current_date] = np.asscalar(sumpdt_px_units/data["Divisor_Series"].loc[current_date])

df_input_CAo_required1 = df_input_CAo[(df_input_CAo["Effective Date"]>current_date) & (df_input_CAo["Effective Date"]<=next_opendate)]   

    
    
if current_date in rebal_dates:
    print("Rebalancing on "+ str(current_date))
    prerebaldate = current_date
    for columns in df_rebal_details["SEDOL-CHK"][df_rebal_details.index == prerebaldate]:
        data["Adj_px"].loc[next_opendate,df_mapping.loc[columns,"RIC"]] = df_px_fx_adj.loc[current_date,df_mapping.loc[columns,"RIC"]]    
        data["Adj_units"].loc[next_opendate,df_mapping.loc[columns,"RIC"]] = df_rebal_details[df_rebal_details["SEDOL-CHK"] == columns].loc[prerebaldate,"Index Constituent Weights "]*data["Index_Series"].loc[current_date]/df_px_fx_adj.loc[current_date,df_mapping.loc[columns,"RIC"]]    
    
    """CA Adjustment"""
    if not len(df_input_CAo_required1) == 0:
        print("CA on " + str(current_date))
            
        CA_adjust(current_date, next_opendate, prerebaldate)


else:
    for columns in df_rebal_details["SEDOL-CHK"][df_rebal_details.index == prerebaldate]:
        data["Adj_px"].loc[next_opendate,df_mapping.loc[columns,"RIC"]] = df_px_fx_adj.loc[current_date,df_mapping.loc[columns,"RIC"]]    
    data["Adj_units"].loc[next_opendate,:] = data["Adj_units"].loc[current_date,:]
    
    """CA Adjustment"""
    if not len(df_input_CAo_required1) == 0:
        print("CA on " + str(current_date))
            
        CA_adjust(current_date, next_opendate, prerebaldate)


"""Divisor calculation"""
sumpdt_adj_px_units = 0
    
for columns in df_rebal_details["SEDOL-CHK"][df_rebal_details.index == prerebaldate]:
    sumpdt_adj_px_units += data["Adj_units"].loc[next_opendate,df_mapping.loc[columns,"RIC"]]*data["Adj_px"].loc[next_opendate,df_mapping.loc[columns,"RIC"]]

data["Divisor_Series"].loc[next_opendate] = np.asscalar(data["Divisor_Series"].loc[current_date] * sumpdt_adj_px_units / sumpdt_px_units)
        


"""Total Return Series Calculation"""

totaldiv = 0
df_input_CD_required = df_input_CD[(df_input_CD["Effective Date"]>last_closing) & (df_input_CD["Effective Date"]<=current_date)]

df_cash_div = pd.DataFrame(columns = df_px_fx_adj.columns)

if not (len(df_input_CD_required) == 0):
    for rindex in df_input_CD_required.index:
        curncy = df_input_CD_required.loc[rindex,"Currency"]
        mthd = data["Conversion_methodology"].loc[curncy,:]
        if not np.isnan(data["Adj_units"].loc[current_date,df_input_CD_required.loc[rindex,"RIC"]]):
            if mthd.iloc[1] == "inverse":
                df_cash_div.loc[current_date,df_input_CD_required.loc[rindex,"RIC"]] = df_input_CD_required.loc[rindex,"Gross Amount"] / fx_rate.loc[mthd["Ticker"],"Spot Rate"] /mthd.iloc[2]
                totaldiv = totaldiv + data["Adj_units"].loc[current_date,df_input_CD_required.loc[rindex,"RIC"]]*df_input_CD_required.loc[rindex,"Gross Amount"] / fx_rate.loc[mthd["Ticker"],"Spot Rate"] /mthd.iloc[2]
            elif mthd.iloc[1] == "multiply":
                df_cash_div.loc[current_date,df_input_CD_required.loc[rindex,"RIC"]] = df_input_CD_required.loc[rindex,"Gross Amount"] * fx_rate.loc[mthd["Ticker"],"Spot Rate"] /mthd.iloc[2]
                totaldiv = totaldiv + data["Adj_units"].loc[current_date,df_input_CD_required.loc[rindex,"RIC"]]*df_input_CD_required.loc[rindex,"Gross Amount"] * fx_rate.loc[mthd["Ticker"],"Spot Rate"] /mthd.iloc[2]
        
data["Total_Index_Series"].loc[current_date] = np.asscalar(data["Total_Index_Series"].loc[last_closing]*(data["Index_Series"].loc[current_date]+(totaldiv/data["Divisor_Series"].loc[current_date]))/data["Index_Series"].loc[last_closing])

weight_denom = 0
        
for columns in df_rebal_details["SEDOL-CHK"][df_rebal_details.index == preprerebaldate]:
    weight_denom += data["Adj_units"].loc[current_date,df_mapping.loc[columns,"RIC"]]*data["Adj_px"].loc[current_date,df_mapping.loc[columns,"RIC"]]


f_name = 'C:/Users/garima.madan/Desktop/FACTSET_Fintech_Index/Pickeled_File.pkl'             
fileObject = open(f_name, 'wb')
pic.dump(data, fileObject)
fileObject.close() 




""" Index Composition File"""

pre_rebal1 = rebal_dates[rebal_dates<current_date][-1]


df_header_PR =pd.DataFrame(columns = ("Index","FDSTPR"))
df_header_PR.loc[0,"Index"] = "Date"
df_header_PR.loc[0,"FDSTPR"] = current_date.to_pydatetime().strftime('%m/%d/%y')
df_header_PR.loc[1,"Index"] = "Index Value"
df_header_PR.loc[1,"FDSTPR"] = data["Index_Series"].loc[current_date]
df_header_PR.loc[2,"Index"] = ""
df_header_PR.loc[2,"FDSTPR"] = ""


df_main_PR =pd.DataFrame(columns = ("Tickers","ISIN","RIC","SEDOL","Component Currency","Component Weight","Component Units","Component 1 day Performance in USD","Component Closing Price (Local)","FX rate"))
i=0
for columns in df_rebal_details["SEDOL-CHK"][df_rebal_details.index == pre_rebal1]:

    if not (data["Adj_units"].loc[current_date,df_mapping.loc[columns,"RIC"]]==0):    
        df_main_PR.loc[i,"Tickers"] =  df_mapping.loc[columns,"BBG Ticker"]
        df_main_PR.loc[i,"ISIN"] =  df_mapping.loc[columns,"ISIN"]
        df_main_PR.loc[i,"RIC"] =  df_mapping.loc[columns,"RIC"]
        df_main_PR.loc[i,"SEDOL"] =  columns
        df_main_PR.loc[i,"Component Currency"] =  df_mapping.loc[columns,"Currency"]
        df_main_PR.loc[i,"Component Weight"] =  data["Adj_units"].loc[current_date,df_mapping.loc[columns,"RIC"]]*data["Adj_px"].loc[current_date,df_mapping.loc[columns,"RIC"]]/weight_denom
        df_main_PR.loc[i,"Component Units"] =  data["Adj_units"].loc[current_date,df_mapping.loc[columns,"RIC"]]
        df_main_PR.loc[i,"Component 1 day Performance in USD"] =  float((df_px_fx_adj.loc[current_date,df_mapping.loc[columns,"RIC"]]/data["Adj_px"].loc[current_date,df_mapping.loc[columns,"RIC"]])-1)
        df_main_PR.loc[i,"Component Closing Price (Local)"] =  data["df_input_px"].loc[current_date,df_mapping.loc[columns,"RIC"]]
        df_main_PR.loc[i,"FX rate"] =  float(df_px_fx_adj.loc[current_date,df_mapping.loc[columns,"RIC"]]/data["df_input_px"].loc[current_date,df_mapping.loc[columns,"RIC"]])

    i=i+1

directory ="C:/Users/garima.madan/Desktop/FACTSET_Fintech_Index/Output/Publish " +current_date.to_pydatetime().strftime('%Y%m%d')+ "/"
if not os.path.exists(directory):
    os.makedirs(directory) 
filename = directory + "Index_composition_FDSFTPR_" + current_date.to_pydatetime().strftime('%Y%m%d') + ".csv"
df_header_PR.to_csv(filename, sep=',', index = False)
with open(filename, 'a') as f:
    f.write("")
    df_main_PR.to_csv(f, index=False)

   
df_header_TR =pd.DataFrame(columns = ("Index","FDSTTR"))
df_header_TR.loc[0,"Index"] = "Date"
df_header_TR.loc[0,"FDSTTR"] = current_date.to_pydatetime().strftime('%m/%d/%y')
df_header_TR.loc[1,"Index"] = "Index Value"
df_header_TR.loc[1,"FDSTTR"] = data["Total_Index_Series"].loc[current_date]
df_header_TR.loc[2,"Index"] = ""
df_header_TR.loc[2,"FDSTTR"] = ""

df_main_TR =pd.DataFrame(columns = ("Tickers","ISIN","RIC","SEDOL","Component Currency","Component Weight","Component Units","Component 1 day Performance in USD","Component Closing Price (Local)","FX rate"))

i=0
for columns in df_rebal_details["SEDOL-CHK"][df_rebal_details.index == pre_rebal1]:

    if not (data["Adj_units"].loc[current_date,df_mapping.loc[columns,"RIC"]]==0):    
        df_main_TR.loc[i,"Tickers"] =  df_mapping.loc[columns,"BBG Ticker"]
        df_main_TR.loc[i,"ISIN"] =  df_mapping.loc[columns,"ISIN"]
        df_main_TR.loc[i,"RIC"] =  df_mapping.loc[columns,"RIC"]
        df_main_TR.loc[i,"SEDOL"] =  columns
        df_main_TR.loc[i,"Component Currency"] =  df_mapping.loc[columns,"Currency"]
        df_main_TR.loc[i,"Component Weight"] =  data["Adj_units"].loc[current_date,df_mapping.loc[columns,"RIC"]]*data["Adj_px"].loc[current_date,df_mapping.loc[columns,"RIC"]]/weight_denom
        df_main_TR.loc[i,"Component Units"] =  data["Adj_units"].loc[current_date,df_mapping.loc[columns,"RIC"]]
        if not len(df_cash_div)==0:
            if not np.isnan(df_cash_div.loc[current_date,df_mapping.loc[columns,"RIC"]]):
                df_main_TR.loc[i,"Component 1 day Performance in USD"] =  float(((df_px_fx_adj.loc[current_date,df_mapping.loc[columns,"RIC"]]+df_cash_div.loc[current_date,df_mapping.loc[columns,"RIC"]])/data["Adj_px"].loc[current_date,df_mapping.loc[columns,"RIC"]])-1)
        else:
            df_main_TR.loc[i,"Component 1 day Performance in USD"] =  float((df_px_fx_adj.loc[current_date,df_mapping.loc[columns,"RIC"]]/data["Adj_px"].loc[current_date,df_mapping.loc[columns,"RIC"]])-1)
        df_main_TR.loc[i,"Component Closing Price (Local)"] =  data["df_input_px"].loc[current_date,df_mapping.loc[columns,"RIC"]]
        df_main_TR.loc[i,"FX rate"] =  float(df_px_fx_adj.loc[current_date,df_mapping.loc[columns,"RIC"]]/data["df_input_px"].loc[current_date,df_mapping.loc[columns,"RIC"]])

    i=i+1

filename = directory + "Index_composition_FDSFTTR_" + current_date.to_pydatetime().strftime('%Y%m%d') + ".csv"
df_header_TR.to_csv(filename, sep=',', index = False)
with open(filename, 'a') as f:
    f.write("")
    df_main_TR.to_csv(f, index=False)
#date = pd.to_datetime('2016-12-06')

"""Open-Close files"""
df_OCheader_PR =pd.DataFrame(columns = ("Cal Date",current_date.to_pydatetime().strftime('%m/%d/%y')))
df_OCheader_PR.loc[0,"Cal Date"] = "FDSTPR"
df_OCheader_PR.loc[1,"Cal Date"] = "FDSTTR"
df_OCheader_PR.loc[2,"Cal Date"] = "Index Currency"
df_OCheader_PR.loc[3,"Cal Date"] = "Rebal Date"
df_OCheader_PR.loc[4,"Cal Date"] = "Close Divisor"
df_OCheader_PR.loc[5,"Cal Date"] = "Open Date"
df_OCheader_PR.loc[6,"Cal Date"] = "Open Divisor"

df_OCheader_PR.loc[0,current_date.to_pydatetime().strftime('%m/%d/%y')] = data["Index_Series"].loc[current_date]
df_OCheader_PR.loc[1,current_date.to_pydatetime().strftime('%m/%d/%y')] = data["Total_Index_Series"].loc[current_date]
df_OCheader_PR.loc[2,current_date.to_pydatetime().strftime('%m/%d/%y')] = "USD"
df_OCheader_PR.loc[3,current_date.to_pydatetime().strftime('%m/%d/%y')] = prerebaldate.to_pydatetime().strftime('%m/%d/%y')
df_OCheader_PR.loc[4,current_date.to_pydatetime().strftime('%m/%d/%y')] = data["Divisor_Series"].loc[current_date]
df_OCheader_PR.loc[5,current_date.to_pydatetime().strftime('%m/%d/%y')] = next_opendate.to_pydatetime().strftime('%m/%d/%y')
df_OCheader_PR.loc[6,current_date.to_pydatetime().strftime('%m/%d/%y')] = data["Divisor_Series"].loc[next_opendate]

df_OCmain_PR =pd.DataFrame(columns = ("S.No","BBG Ticker","Security Name","ISIN","RIC","Main Exchange","Sedol","Currency of ticker(local)","Fx Rate for ticker","Closing Price (local)","Closing Price[USD]", "Closing Units","Weight","Open Price[USD]","Open Units [AN]","Cash Dividend [USD]","Dividend times shares","Other CA"))

#For non rebal day

#for rebalancing

comp_list = np.union1d(df_rebal_details["SEDOL-CHK"][df_rebal_details.index == pre_rebal1], df_rebal_details["SEDOL-CHK"][df_rebal_details.index == prerebaldate])


i=0
for columns in comp_list:
    i=i+1
    if not (data["Adj_units"].loc[current_date, df_mapping.loc[columns,"RIC"]] == 0): 
        df_OCmain_PR.loc[i,"S.No"] =  i
        df_OCmain_PR.loc[i,"BBG Ticker"] =  df_mapping.loc[columns,"BBG Ticker"]
        df_OCmain_PR.loc[i,"Security Name"] =  df_mapping.loc[columns,"Company Name"]
        df_OCmain_PR.loc[i,"ISIN"] =  df_mapping.loc[columns,"ISIN"]
        df_OCmain_PR.loc[i,"RIC"] =  df_mapping.loc[columns,"RIC"]
        df_OCmain_PR.loc[i,"Main Exchange"] =  df_mapping.loc[columns,"Exchange"] 
        df_OCmain_PR.loc[i,"Sedol"] =  columns
        df_OCmain_PR.loc[i,"Currency of ticker(local)"] =  df_mapping.loc[columns,"Currency"]
        df_OCmain_PR.loc[i,"Fx Rate for ticker"] =  df_px_fx_adj.loc[current_date,df_mapping.loc[columns,"RIC"]]/data["df_input_px"].loc[current_date,df_mapping.loc[columns,"RIC"]]
        df_OCmain_PR.loc[i,"Closing Price (local)"] =  data["df_input_px"].loc[current_date,df_mapping.loc[columns,"RIC"]]
        df_OCmain_PR.loc[i,"Closing Price[USD]"] =  df_px_fx_adj.loc[current_date,df_mapping.loc[columns,"RIC"]]
        df_OCmain_PR.loc[i,"Closing Units"] =  data["Adj_units"].loc[current_date,df_mapping.loc[columns,"RIC"]]
        df_OCmain_PR.loc[i,"Weight"] =  data["Adj_units"].loc[current_date,df_mapping.loc[columns,"RIC"]]*data["Adj_px"].loc[current_date,df_mapping.loc[columns,"RIC"]]/weight_denom
        df_OCmain_PR.loc[i,"Closing Units"] =  data["Adj_units"].loc[current_date,df_mapping.loc[columns,"RIC"]]
        df_OCmain_PR.loc[i,"Open Price[USD]"] =  data["Adj_px"].loc[next_opendate,df_mapping.loc[columns,"RIC"]]
        df_OCmain_PR.loc[i,"Open Units [AN]"] =  data["Adj_units"].loc[next_opendate,df_mapping.loc[columns,"RIC"]] 
        
        if not len(df_cash_div) == 0:
            df_OCmain_PR.loc[i,"Cash Dividend [USD]"] =  df_cash_div.loc[current_date,df_mapping.loc[columns,"RIC"]]
            df_OCmain_PR.loc[i,"Dividend times shares"] =  df_cash_div.loc[current_date,df_mapping.loc[columns,"RIC"]]*data["Adj_units"].loc[current_date,df_mapping.loc[columns,"RIC"]]
        else:
            df_OCmain_PR.loc[i,"Cash Dividend [USD]"] =  ""
            df_OCmain_PR.loc[i,"Dividend times shares"] = "" 
    
        if (df_mapping.loc[columns,"RIC"] in np.array(df_input_CAo_required1["RIC"])):
            df_input_CAo_required = df_input_CAo_required1[df_input_CAo_required1["RIC"]==df_mapping.loc[columns,"RIC"]]
            df_OCmain_PR.loc[i,"Other CA"] = df_input_CAo_required.iloc[0,1]

   
filename = directory + "Open-Close_FDSFTPR_" + current_date.to_pydatetime().strftime('%Y%m%d') + ".csv"
df_OCheader_PR.to_csv(filename, sep=',', index = False)
with open(filename, 'a') as fo:
    fo.write('\n')
    df_OCmain_PR.to_csv(fo, index=False)

filename_nqd = directory + "FactSet_" + next_opendate.to_pydatetime().strftime('%Y%m%d') + ".txt"
with open(filename_nqd, 'w') as f_nqd:
    f_nqd.write("--START FILE\n")
    f_nqd.write("--START INDEX_LEVEL\n")
    f_nqd.write("CALC_DATE\tIDENT\tPX\tOldPX\n")      
    
    f_nqd.write(current_date.to_pydatetime().strftime('%Y%m%d')+"\t"+"FDSFTPR\t"+str(round(data["Index_Series"].loc[current_date],9))+"\t"+str(round(data["Index_Series"].loc[last_closing],9))+"\n")    
    f_nqd.write(current_date.to_pydatetime().strftime('%Y%m%d')+"\t"+"FDSFTTR\t"+str(round(data["Total_Index_Series"].loc[current_date],9))+"\t"+str(round(data["Total_Index_Series"].loc[last_closing],9))+"\n")    

    f_nqd.write("--END INDEX_LEVEL\n")      
    f_nqd.write("--END FILE")

         
"""Outlook mail generation"""

outlook = win32.Dispatch('outlook.application')
mail = outlook.CreateItem(0)
mail.To = "jezhou@factset.com; AOsawa@factset.com; nazhang@factset.com; ynatsume@factset.com; vramos@factset.com; HSambongi@factset.com;Neil.Wardley@evalueserve.com;"
mail.Cc = "Anu.bhayana@evalueserve.com; jasmeet.gujral@evalueserve.com;"
mail.Subject = "FDSFTPR and FDSFTTR for " + current_date.to_pydatetime().strftime('%Y%m%d')
mail.HtmlBody = "Hi all, <br /> <br /> Please find below index levels.<br />FDSFTPR " + str(round(data["Index_Series"].loc[current_date],10)) + "<br />FDSFTTR " + str(round(data["Total_Index_Series"].loc[current_date],10)) + "<br /> <br />Regards<br />Evalueserve Team"
mail.display(True)