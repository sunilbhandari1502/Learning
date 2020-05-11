# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:49:01 2017

@author: garima.madan
"""

import pandas as pd
import numpy as np
import Read_Data_Modified
import pickle as pic
import win32com.client as win32   
import os

""" Defining BT period """
incep_date = pd.to_datetime('2011-01-05') #inceptiondate for index is fixed
#end_date = pd.to_datetime(input("Enter date in backtest end date in YYYY-MM-DD format: ")) #Taking end date of BT from user
#next_opendate = pd.to_datetime(input("Enter date next Opendate in YYYY-MM-DD format: "))
end_date = pd.to_datetime('2012-08-20') #Taking end date of BT from user
next_opendate = pd.to_datetime('2012-08-23')


print ("Start and End Date defined") #printing status on console

"""Reading data"""
df_input_px, df_mapping, df_rebal_details, df_input_CD, df_input_CAo = Read_Data_Modified.read_data(incep_date,end_date)

"""FX conversion methodology"""
df_FX_conv_method = pd.DataFrame(columns = ["FX", "Ticker", "Methodology", "Factor"])
df_FX_conv_method["FX"] = df_mapping["Currency"]
df_FX_conv_method["Ticker"] = df_mapping["Ccy_mapping"]
df_FX_conv_method.set_index("FX", inplace = True)   
df_FX_conv_method.drop_duplicates(inplace = True)

for curncy in df_FX_conv_method.index:
    if curncy[-1].islower():
        df_FX_conv_method.loc[curncy,"Factor"] = 100
    else:
        df_FX_conv_method.loc[curncy,"Factor"] = 1

    if df_FX_conv_method.loc[curncy,"Ticker"][:3] == "USD":
        df_FX_conv_method.loc[curncy,"Methodology"] = "inverse"
    else:
        df_FX_conv_method.loc[curncy,"Methodology"] = "multiply"

"""Setting rebal contituent and weight data set"""
rebal_dates = df_rebal_details.index.drop_duplicates()

"""Finding Index business days"""
lt_index_bus_day = [incep_date]
for date in df_input_px.index[1:]:
    pre_rebal = rebal_dates[rebal_dates<date][-1]
    if np.nansum(df_input_px.loc[date,:][df_mapping["RIC"][df_mapping.index.isin(df_rebal_details["SEDOL-CHK"][df_rebal_details.index==pre_rebal])]])==0:
        pass
    else:
        lt_index_bus_day.append(date)
        

"""Original px_data with pad and limited to index business days"""
df_input_index_days = df_input_px.fillna(method ='pad').loc[lt_index_bus_day,:]

"""FX adjusted prices"""
df_px_fx_adj = pd.DataFrame(columns = df_mapping["RIC"], index = df_input_index_days.index )

for column in df_px_fx_adj.columns:
    crncy = df_mapping["Currency"][df_mapping["RIC"]==column]
    mthd = df_FX_conv_method.loc[crncy,:]  
    #Currency of underlying from mapping table
    #Conversion methodology from 
    #factor 
    if mthd.iloc[0,1] == "inverse":
        df_px_fx_adj[column] =  (df_input_index_days[column]/df_input_index_days[mthd["Ticker"]].transpose()/mthd.iloc[0,2]).transpose()
    elif mthd.iloc[0,1] == "multiply":
        df_px_fx_adj[column] =  (df_input_index_days[column]*df_input_index_days[mthd["Ticker"]].transpose()/mthd.iloc[0,2]).transpose()

""" Adj Px and Adj N Calculation """
df_adjpx = pd.DataFrame(columns = df_px_fx_adj.columns, index = df_px_fx_adj.index )
df_adjunits = pd.DataFrame(columns = df_px_fx_adj.columns, index = df_px_fx_adj.index )
Index_Series = pd.Series(index = lt_index_bus_day) #Price return
Divisor = pd.Series(index = lt_index_bus_day)
Total_Index_Series = pd.Series(index = lt_index_bus_day)
totaldiv = pd.Series(index = lt_index_bus_day)
df_cash_div = pd.DataFrame(columns = df_px_fx_adj.columns, index = df_px_fx_adj.index )


""" CA adjustment function"""

def CA_adjust(date, Opendate, prerebaldate):
    global df_input_CAo_required1
    global df_mapping
    global df_rebal_details
    global df_FX_conv_method
    global df_input_index_days
    global df_adjunits
    global df_adjpx    
    
    
    for columns in df_rebal_details["SEDOL-CHK"][df_rebal_details.index == prerebaldate]:
        
                if (df_mapping.loc[columns,"RIC"] in np.array(df_input_CAo_required1["RIC"])):
                    df_input_CAo_required = df_input_CAo_required1[df_input_CAo_required1["RIC"]==df_mapping.loc[columns,"RIC"]]
                    
                    if not len(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Delisting"]) == 0:
                        df_adjunits.loc[Opendate,df_mapping.loc[columns,"RIC"]] = 0
                        
                    if not len(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Cash Dividend"]) == 0:
                        #Get Currency of CA
                        curncy = df_input_CAo_required[df_input_CAo_required["Action Type"]=="Cash Dividend"]["Currency"]
                        mthd = df_FX_conv_method.loc[curncy,:]
                        if mthd.iloc[0,1] == "inverse":
                            df_adjpx.loc[Opendate,df_mapping.loc[columns,"RIC"]] = df_adjpx.loc[Opendate,df_mapping.loc[columns,"RIC"]] - (df_input_CAo_required[df_input_CAo_required["Action Type"]=="Cash Dividend"]["Gross Amount"] / df_input_index_days.loc[date,mthd["Ticker"]].values/mthd.iloc[0,2]).values
                        elif mthd.iloc[0,1] == "multiply":
                            df_adjpx.loc[Opendate,df_mapping.loc[columns,"RIC"]] = df_adjpx.loc[Opendate,df_mapping.loc[columns,"RIC"]] - (df_input_CAo_required[df_input_CAo_required["Action Type"]=="Cash Dividend"]["Gross Amount"] * df_input_index_days.loc[date,mthd["Ticker"]].values/mthd.iloc[0,2]).values

                    
                    if not len(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Stock Dividend"]) == 0:
                        df_adjpx.loc[Opendate,df_mapping.loc[columns,"RIC"]] = df_adjpx.loc[Opendate,df_mapping.loc[columns,"RIC"]]/np.array(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Stock Dividend"]["Gross Amount"])
                        df_adjunits.loc[Opendate,df_mapping.loc[columns,"RIC"]] = df_adjunits.loc[Opendate,df_mapping.loc[columns,"RIC"]]*np.array(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Stock Dividend"]["Gross Amount"])
                    
                    if not len(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Stock Split"]) == 0:
                        df_adjpx.loc[Opendate,df_mapping.loc[columns,"RIC"]] = df_adjpx.loc[Opendate,df_mapping.loc[columns,"RIC"]]/np.array(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Stock Split"]["Gross Amount"])
                        df_adjunits.loc[Opendate,df_mapping.loc[columns,"RIC"]] = df_adjunits.loc[Opendate,df_mapping.loc[columns,"RIC"]]*np.array(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Stock Split"]["Gross Amount"])
                    
                    if not len(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Rights Offerings"]) == 0:
                        curncy = df_input_CAo_required[df_input_CAo_required["Action Type"]=="Rights Offerings"]["Currency"]
                        mthd = df_FX_conv_method.loc[curncy,:]                                                  
                        if mthd.iloc[0,1] == "inverse":
                            df_adjpx.loc[Opendate,df_mapping.loc[columns,"RIC"]] = float(np.array((df_adjpx.loc[Opendate,df_mapping.loc[columns,"RIC"]]+(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Rights Offerings"]["Subscription Price"]/mthd.iloc[0,2]/df_input_index_days.loc[date,mthd["Ticker"]].values*df_input_CAo_required[df_input_CAo_required["Action Type"]=="Rights Offerings"]["Gross Amount"]))/(1+df_input_CAo_required[df_input_CAo_required["Action Type"]=="Rights Offerings"]["Gross Amount"])))
                        elif mthd.iloc[0,1] == "multiply":
                            df_adjpx.loc[Opendate,df_mapping.loc[columns,"RIC"]] = float(np.array((df_adjpx.loc[Opendate,df_mapping.loc[columns,"RIC"]]+(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Rights Offerings"]["Subscription Price"]/mthd.iloc[0,2]*df_input_index_days.loc[date,mthd["Ticker"]].values*df_input_CAo_required[df_input_CAo_required["Action Type"]=="Rights Offerings"]["Gross Amount"]))/(1+df_input_CAo_required[df_input_CAo_required["Action Type"]=="Rights Offerings"]["Gross Amount"])))
                        
                        df_adjunits.loc[Opendate,df_mapping.loc[columns,"RIC"]] = float(np.array(df_adjunits.loc[Opendate,df_mapping.loc[columns,"RIC"]]*(1+df_input_CAo_required[df_input_CAo_required["Action Type"]=="Rights Offerings"]["Gross Amount"])))
                    
                    if not len(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Spin-off"]) == 0:
                        curncy = df_input_CAo_required[df_input_CAo_required["Action Type"]=="Spin-off"]["Currency"]
                        mthd = df_FX_conv_method.loc[curncy,:]                                                  
                        if mthd.iloc[0,1] == "inverse":
                            df_adjpx.loc[Opendate,df_mapping.loc[columns,"RIC"]] = float(np.array((df_adjpx.loc[Opendate,df_mapping.loc[columns,"RIC"]]-(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Spin-off"]["Subscription Price"]*df_input_CAo_required[df_input_CAo_required["Action Type"]=="Spin-off"]["Gross Amount"] / np.array(df_input_index_days.loc[date,mthd["Ticker"]]) /mthd.iloc[0,2]))))    
                        elif mthd.iloc[0,1] == "multiply":
                            df_adjpx.loc[Opendate,df_mapping.loc[columns,"RIC"]] = float(np.array((df_adjpx.loc[Opendate,df_mapping.loc[columns,"RIC"]]-(df_input_CAo_required[df_input_CAo_required["Action Type"]=="Spin-off"]["Subscription Price"]*df_input_CAo_required[df_input_CAo_required["Action Type"]=="Spin-off"]["Gross Amount"] * np.array(df_input_index_days.loc[date,mthd["Ticker"]]) /mthd.iloc[0,2]))))

"""Function end"""

for date in df_adjpx.index:
    
    
    if date == incep_date:
        prerebaldate = date
        preindex = 100
        Index_Series[date] = 100
        Divisor[date] = 1
        
        for columns in df_rebal_details["SEDOL-CHK"][df_rebal_details.index == prerebaldate]:       
            df_adjunits.loc[date,df_mapping.loc[columns,"RIC"]] = df_rebal_details[df_rebal_details["SEDOL-CHK"] == columns].loc[prerebaldate,"Index Constituent Weights "]*preindex/df_px_fx_adj.loc[prerebaldate,df_mapping.loc[columns,"RIC"]]
            df_adjpx.loc[date,df_mapping.loc[columns,"RIC"]] = df_px_fx_adj.loc[date,df_mapping.loc[columns,"RIC"]]    

        print("Unit calculation for inception day completed")  
    preprerebaldate = prerebaldate
    
    if date == end_date:
        Opendate = next_opendate
    else:    
        Opendate = df_adjpx.index[df_adjpx.index > date][0] #Next Open date

    """CA for divisor adjustment"""    
    df_input_CAo_required1 = df_input_CAo[(df_input_CAo["Effective Date"]>date) & (df_input_CAo["Effective Date"]<=Opendate)]   
    
    """Index level calculation"""                                      
    sumpdt_px_units = 0
    for columns in df_rebal_details["SEDOL-CHK"][df_rebal_details.index == prerebaldate]:
        sumpdt_px_units += (df_adjunits.loc[date,df_mapping.loc[columns,"RIC"]]*df_px_fx_adj.loc[date,df_mapping.loc[columns,"RIC"]])


    Index_Series[date] = sumpdt_px_units/Divisor[date]
    
    """Price calculation"""

    """Adj units calculation"""
    
    """Rebalancing date check"""
    if date in rebal_dates:
        print("Rebalancing on "+ str(date))
        prerebaldate = date
        for columns in df_rebal_details["SEDOL-CHK"][df_rebal_details.index == prerebaldate]:
            df_adjunits.loc[Opendate,df_mapping.loc[columns,"RIC"]] = df_rebal_details[df_rebal_details["SEDOL-CHK"] == columns].loc[prerebaldate,"Index Constituent Weights "]*Index_Series[date]/df_px_fx_adj.loc[prerebaldate,df_mapping.loc[columns,"RIC"]]
            df_adjpx.loc[Opendate,df_mapping.loc[columns,"RIC"]] = df_px_fx_adj.loc[date,df_mapping.loc[columns,"RIC"]]    

        """CA check"""

        if not len(df_input_CAo_required1) == 0:
            print("CA on " + str(date))
            
            CA_adjust(date, Opendate, prerebaldate)
        
        

    else:
        df_adjunits.loc[Opendate,:] = df_adjunits.loc[date,:]
        for columns in df_rebal_details["SEDOL-CHK"][df_rebal_details.index == prerebaldate]:
            df_adjpx.loc[Opendate,df_mapping.loc[columns,"RIC"]] = df_px_fx_adj.loc[date,df_mapping.loc[columns,"RIC"]]    
                
        """CA check"""

        if not len(df_input_CAo_required1) == 0:
            print("CA on " + str(date))
            
            CA_adjust(date, Opendate, prerebaldate)
    
    sumpdt_adj_px_units = 0
    
    for columns in df_rebal_details["SEDOL-CHK"][df_rebal_details.index == prerebaldate]:
        sumpdt_adj_px_units += df_adjunits.loc[Opendate,df_mapping.loc[columns,"RIC"]]*df_adjpx.loc[Opendate,df_mapping.loc[columns,"RIC"]]
    Divisor[Opendate] = np.asscalar(Divisor[date] * sumpdt_adj_px_units / sumpdt_px_units)
    
    weight_denom = 0                  
    for columns in df_rebal_details["SEDOL-CHK"][df_rebal_details.index == preprerebaldate]:
        weight_denom = weight_denom + df_adjunits.loc[date,df_mapping.loc[columns,"RIC"]]*df_adjpx.loc[date,df_mapping.loc[columns,"RIC"]]
    
    
    
    """Total Return Series Calculation"""
    if date == incep_date:
        Total_Index_Series[date] = 100
        pre_date = date
        totaldiv[date] = 0

    totaldiv[date] = 0
    df_input_CD_required = df_input_CD[(df_input_CD["Effective Date"]>pre_date) & (df_input_CD["Effective Date"]<=date)]
    if not (len(df_input_CD_required) == 0):
        for rindex in df_input_CD_required.index:
            curncy = df_input_CD_required.loc[rindex,"Currency"]
            mthd = df_FX_conv_method.loc[curncy,:]
            if not np.isnan(df_adjunits.loc[date,df_input_CD_required.loc[rindex,"RIC"]]):
                if mthd.iloc[1] == "inverse":
                    df_cash_div.loc[date,df_input_CD_required.loc[rindex,"RIC"]] = float(df_input_CD_required.loc[rindex,"Gross Amount"] / df_input_index_days.loc[date,mthd["Ticker"]] /mthd.iloc[2])
                    totaldiv[date] = totaldiv[date] + float(df_adjunits.loc[date,df_input_CD_required.loc[rindex,"RIC"]]*df_input_CD_required.loc[rindex,"Gross Amount"] / df_input_index_days.loc[date,mthd["Ticker"]] /mthd.iloc[2])
                elif mthd.iloc[1] == "multiply":
                    df_cash_div.loc[date,df_input_CD_required.loc[rindex,"RIC"]] = float(df_input_CD_required.loc[rindex,"Gross Amount"] * df_input_index_days.loc[date,mthd["Ticker"]] /mthd.iloc[2])
                    totaldiv[date] = totaldiv[date] + float(df_adjunits.loc[date,df_input_CD_required.loc[rindex,"RIC"]]*df_input_CD_required.loc[rindex,"Gross Amount"] * df_input_index_days.loc[date,mthd["Ticker"]] /mthd.iloc[2])
        
    Total_Index_Series[date] = Total_Index_Series[pre_date]*(Index_Series[date]+(totaldiv[date]/Divisor[date]))/Index_Series[pre_date]
        
    pre_date = date
    

"""Serialisation of files"""

df_input_index_days.drop(df_input_index_days.columns[-18:], axis = 1, inplace = True)
    
file_Name = {'Divisor_Series' : Divisor, 
             'Index_Series' : Index_Series,
             'Total_Index_Series' : Total_Index_Series,
             'Adj_px' : df_adjpx, 
             'Adj_units' : df_adjunits, 
             'Conversion_methodology' : df_FX_conv_method,
             'df_input_px' : df_input_index_days
             }

f_name = 'Pickeled_File.pkl'             
fileObject = open(f_name, 'wb')
pic.dump(file_Name, fileObject)
fileObject.close() 
             

""" Index Composition File"""
preceding_date = np.array(lt_index_bus_day)[np.array(lt_index_bus_day)<end_date][-1]

if(date == incep_date):
    pre_rebal1 = incep_date   
else:
    pre_rebal1 = rebal_dates[rebal_dates<date][-1]

df_header_PR =pd.DataFrame(columns = ("Index","FDSTPR"))
df_header_PR.loc[0,"Index"] = "Date"
df_header_PR.loc[0,"FDSTPR"] = date.to_pydatetime().strftime('%m/%d/%y')
df_header_PR.loc[1,"Index"] = "Index Value"
df_header_PR.loc[1,"FDSTPR"] = Index_Series[date]
df_header_PR.loc[2,"Index"] = ""
df_header_PR.loc[2,"FDSTPR"] = ""


df_main_PR =pd.DataFrame(columns = ("Tickers","ISIN","RIC","SEDOL","Component Currency","Component Weight","Component Units","Component 1 day Performance in USD","Component Closing Price (Local)","FX rate"))
i=0
for columns in df_rebal_details["SEDOL-CHK"][df_rebal_details.index == pre_rebal1]:

    if not (df_adjunits.loc[date,df_mapping.loc[columns,"RIC"]]==0):    
        df_main_PR.loc[i,"Tickers"] =  df_mapping.loc[columns,"BBG Ticker"]
        df_main_PR.loc[i,"ISIN"] =  df_mapping.loc[columns,"ISIN"]
        df_main_PR.loc[i,"RIC"] =  df_mapping.loc[columns,"RIC"]
        df_main_PR.loc[i,"SEDOL"] =  columns
        df_main_PR.loc[i,"Component Currency"] =  df_mapping.loc[columns,"Currency"]
        df_main_PR.loc[i,"Component Weight"] =  df_adjunits.loc[date,df_mapping.loc[columns,"RIC"]]*df_adjpx.loc[date,df_mapping.loc[columns,"RIC"]]/weight_denom
        df_main_PR.loc[i,"Component Units"] =  df_adjunits.loc[date,df_mapping.loc[columns,"RIC"]]
        df_main_PR.loc[i,"Component 1 day Performance in USD"] =  float((df_px_fx_adj.loc[date,df_mapping.loc[columns,"RIC"]]/df_adjpx.loc[date,df_mapping.loc[columns,"RIC"]])-1)
        df_main_PR.loc[i,"Component Closing Price (Local)"] =  df_input_index_days.loc[date,df_mapping.loc[columns,"RIC"]]
        df_main_PR.loc[i,"FX rate"] =  float(df_px_fx_adj.loc[date,df_mapping.loc[columns,"RIC"]]/df_input_index_days.loc[date,df_mapping.loc[columns,"RIC"]])

    i=i+1

directory ="C:/Users/garima.madan/Desktop/FACTSET_Fintech_Index/Output/Publish " +date.to_pydatetime().strftime('%Y%m%d')+ "/"
if not os.path.exists(directory):
    os.makedirs(directory) 
filename = directory + "Index_composition_FDSFTPR_" + date.to_pydatetime().strftime('%Y%m%d') + ".csv"
df_header_PR.to_csv(filename, sep=',', index = False)
with open(filename, 'a') as f:
    f.write("")
    df_main_PR.to_csv(f, index=False)

   
df_header_TR =pd.DataFrame(columns = ("Index","FDSTTR"))
df_header_TR.loc[0,"Index"] = "Date"
df_header_TR.loc[0,"FDSTTR"] = date.to_pydatetime().strftime('%m/%d/%y')
df_header_TR.loc[1,"Index"] = "Index Value"
df_header_TR.loc[1,"FDSTTR"] = Total_Index_Series[date]
df_header_TR.loc[2,"Index"] = ""
df_header_TR.loc[2,"FDSTTR"] = ""

df_main_TR =pd.DataFrame(columns = ("Tickers","ISIN","RIC","SEDOL","Component Currency","Component Weight","Component Units","Component 1 day Performance in USD","Component Closing Price (Local)","FX rate"))

i=0
for columns in df_rebal_details["SEDOL-CHK"][df_rebal_details.index == pre_rebal1]:

    if not (df_adjunits.loc[date,df_mapping.loc[columns,"RIC"]]==0):    
        df_main_TR.loc[i,"Tickers"] =  df_mapping.loc[columns,"BBG Ticker"]
        df_main_TR.loc[i,"ISIN"] =  df_mapping.loc[columns,"ISIN"]
        df_main_TR.loc[i,"RIC"] =  df_mapping.loc[columns,"RIC"]
        df_main_TR.loc[i,"SEDOL"] =  columns
        df_main_TR.loc[i,"Component Currency"] =  df_mapping.loc[columns,"Currency"]
        df_main_TR.loc[i,"Component Weight"] =  df_adjunits.loc[date,df_mapping.loc[columns,"RIC"]]*df_adjpx.loc[date,df_mapping.loc[columns,"RIC"]]/weight_denom
        df_main_TR.loc[i,"Component Units"] =  df_adjunits.loc[date,df_mapping.loc[columns,"RIC"]]
        if not np.isnan(df_cash_div.loc[date,df_mapping.loc[columns,"RIC"]]):
            df_main_TR.loc[i,"Component 1 day Performance in USD"] =  float(((df_px_fx_adj.loc[date,df_mapping.loc[columns,"RIC"]]+df_cash_div.loc[date,df_mapping.loc[columns,"RIC"]])/df_adjpx.loc[date,df_mapping.loc[columns,"RIC"]])-1)
        else:
            df_main_TR.loc[i,"Component 1 day Performance in USD"] =  float((df_px_fx_adj.loc[date,df_mapping.loc[columns,"RIC"]]/df_adjpx.loc[date,df_mapping.loc[columns,"RIC"]])-1)
        df_main_TR.loc[i,"Component Closing Price (Local)"] =  df_input_index_days.loc[date,df_mapping.loc[columns,"RIC"]]
        df_main_TR.loc[i,"FX rate"] =  float(df_px_fx_adj.loc[date,df_mapping.loc[columns,"RIC"]]/df_input_index_days.loc[date,df_mapping.loc[columns,"RIC"]])

    i=i+1

filename = directory + "Index_composition_FDSFTTR_" + date.to_pydatetime().strftime('%Y%m%d') + ".csv"
df_header_TR.to_csv(filename, sep=',', index = False)
with open(filename, 'a') as f:
    f.write("")
    df_main_TR.to_csv(f, index=False)
#date = pd.to_datetime('2016-12-06')

"""Open-Close files"""
df_OCheader_PR =pd.DataFrame(columns = ("Cal Date",date.to_pydatetime().strftime('%m/%d/%y')))
df_OCheader_PR.loc[0,"Cal Date"] = "FDSTPR"
df_OCheader_PR.loc[1,"Cal Date"] = "FDSTTR"
df_OCheader_PR.loc[2,"Cal Date"] = "Index Currency"
df_OCheader_PR.loc[3,"Cal Date"] = "Rebal Date"
df_OCheader_PR.loc[4,"Cal Date"] = "Close Divisor"
df_OCheader_PR.loc[5,"Cal Date"] = "Open Date"
df_OCheader_PR.loc[6,"Cal Date"] = "Open Divisor"

df_OCheader_PR.loc[0,date.to_pydatetime().strftime('%m/%d/%y')] = Index_Series[date]
df_OCheader_PR.loc[1,date.to_pydatetime().strftime('%m/%d/%y')] = Total_Index_Series[date]
df_OCheader_PR.loc[2,date.to_pydatetime().strftime('%m/%d/%y')] = "USD"
df_OCheader_PR.loc[3,date.to_pydatetime().strftime('%m/%d/%y')] = prerebaldate.to_pydatetime().strftime('%m/%d/%y')
df_OCheader_PR.loc[4,date.to_pydatetime().strftime('%m/%d/%y')] = Divisor[date]
df_OCheader_PR.loc[5,date.to_pydatetime().strftime('%m/%d/%y')] = Opendate.to_pydatetime().strftime('%m/%d/%y')
df_OCheader_PR.loc[6,date.to_pydatetime().strftime('%m/%d/%y')] = Divisor[Opendate]

df_OCmain_PR =pd.DataFrame(columns = ("S.No","BBG Ticker","Security Name","ISIN","RIC","Main Exchange","Sedol","Currency of ticker(local)","Fx Rate for ticker","Closing Price (local)","Closing Price[USD]", "Closing Units","Weight","Open Price[USD]","Open Units [AN]","Cash Dividend [USD]","Dividend times shares","Other CA"))

#For non rebal day

#for rebalancing

comp_list = np.union1d(df_rebal_details["SEDOL-CHK"][df_rebal_details.index == pre_rebal1], df_rebal_details["SEDOL-CHK"][df_rebal_details.index == prerebaldate])


i=0
for columns in comp_list:
    i=i+1
    if not (df_adjunits.loc[date,df_mapping.loc[columns,"RIC"]]==0):    
        df_OCmain_PR.loc[i,"S.No"] =  i
        df_OCmain_PR.loc[i,"BBG Ticker"] =  df_mapping.loc[columns,"BBG Ticker"]
        df_OCmain_PR.loc[i,"Security Name"] =  df_mapping.loc[columns,"Company Name"]
        df_OCmain_PR.loc[i,"ISIN"] =  df_mapping.loc[columns,"ISIN"]
        df_OCmain_PR.loc[i,"RIC"] =  df_mapping.loc[columns,"RIC"]
        df_OCmain_PR.loc[i,"Main Exchange"] =  df_mapping.loc[columns,"Exchange"] 
        df_OCmain_PR.loc[i,"Sedol"] =  columns
        df_OCmain_PR.loc[i,"Currency of ticker(local)"] =  df_mapping.loc[columns,"Currency"]
        df_OCmain_PR.loc[i,"Fx Rate for ticker"] =  float(df_px_fx_adj.loc[date,df_mapping.loc[columns,"RIC"]]/df_input_index_days.loc[date,df_mapping.loc[columns,"RIC"]])
        df_OCmain_PR.loc[i,"Closing Price (local)"] =  df_input_index_days.loc[date,df_mapping.loc[columns,"RIC"]]
        df_OCmain_PR.loc[i,"Closing Price[USD]"] =  df_px_fx_adj.loc[date,df_mapping.loc[columns,"RIC"]]
        df_OCmain_PR.loc[i,"Closing Units"] =  df_adjunits.loc[date,df_mapping.loc[columns,"RIC"]]
        df_OCmain_PR.loc[i,"Weight"] =  df_adjunits.loc[date,df_mapping.loc[columns,"RIC"]]*df_adjpx.loc[date,df_mapping.loc[columns,"RIC"]]/weight_denom
        df_OCmain_PR.loc[i,"Closing Units"] =  df_adjunits.loc[date,df_mapping.loc[columns,"RIC"]]
        df_OCmain_PR.loc[i,"Open Price[USD]"] =  df_adjpx.loc[Opendate,df_mapping.loc[columns,"RIC"]]
        df_OCmain_PR.loc[i,"Open Units [AN]"] =  df_adjunits.loc[Opendate,df_mapping.loc[columns,"RIC"]] 
        df_OCmain_PR.loc[i,"Cash Dividend [USD]"] =  df_cash_div.loc[date,df_mapping.loc[columns,"RIC"]]
        df_OCmain_PR.loc[i,"Dividend times shares"] =  df_cash_div.loc[date,df_mapping.loc[columns,"RIC"]]*df_adjunits.loc[date,df_mapping.loc[columns,"RIC"]]
    
    
        if (df_mapping.loc[columns,"RIC"] in np.array(df_input_CAo_required1["RIC"])):
            df_input_CAo_required = df_input_CAo_required1[df_input_CAo_required1["RIC"]==df_mapping.loc[columns,"RIC"]]
            df_OCmain_PR.loc[i,"Other CA"] = df_input_CAo_required.iloc[0,1]

   
filename = directory + "Open-Close_FDSFTPR_" + date.to_pydatetime().strftime('%Y%m%d') + ".csv"
df_OCheader_PR.to_csv(filename, sep=',', index = False)
with open(filename, 'a') as fo:
    fo.write('\n')
    df_OCmain_PR.to_csv(fo, index=False)

filename_nqd = directory + "FactSet_" + Opendate.to_pydatetime().strftime('%Y%m%d') + ".txt"
with open(filename_nqd, 'w') as f_nqd:
    f_nqd.write("--START FILE\n")
    f_nqd.write("--START INDEX_LEVEL\n")
    f_nqd.write("CALC_DATE\tIDENT\tPX\tOldPX\n")      
    
    f_nqd.write(date.to_pydatetime().strftime('%Y%m%d')+"\t"+"FDSFTPR\t"+str(round(Index_Series[date],9))+"\t"+str(round(Index_Series[preceding_date],9))+"\n")    
    f_nqd.write(date.to_pydatetime().strftime('%Y%m%d')+"\t"+"FDSFTTR\t"+str(round(Total_Index_Series[date],9))+"\t"+str(round(Total_Index_Series[preceding_date],9))+"\n")    

    f_nqd.write("--END INDEX_LEVEL\n")      
    f_nqd.write("--END FILE")

         
"""Outlook mail generation"""

outlook = win32.Dispatch('outlook.application')
mail = outlook.CreateItem(0)
mail.To = "jezhou@factset.com; AOsawa@factset.com; nazhang@factset.com; ynatsume@factset.com; vramos@factset.com; HSambongi@factset.com;Neil.Wardley@evalueserve.com;"
mail.Cc = "Anu.bhayana@evalueserve.com; jasmeet.gujral@evalueserve.com;"
mail.Subject = "FDSFTPR and FDSFTTR for " + date.to_pydatetime().strftime('%Y%m%d')
mail.HtmlBody = "Hi all, <br /> <br /> Please find below index levels.<br />FDSFTPR " + str(round(Index_Series[date],6)) + "<br />FDSFTTR " + str(round(Total_Index_Series[date],6)) + "<br /> <br />Regards<br />Evalueserve Team"
mail.display(True)
