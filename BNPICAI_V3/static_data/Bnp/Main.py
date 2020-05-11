# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 17:01:18 2016

@author: Jasmeet.Gujral
"""
import pandas as pd
import numpy as np
import OptimizationFile
import Date_Gen
import CovarianceCalc


incep_date = pd.to_datetime('2014-05-20')
end_date = pd.to_datetime('2014-07-31')

"""Input Data"""

""" Reading data from Input_data file"""
input_file = pd.ExcelFile("C:/Users/sunilbhandari1/Documents/Python Scripts/Bnp/Input_Data.xlsx")
df_input_px = input_file.parse("Final_Input").set_index(['Date'])

""" Cleaning data to replace nan with previous day value"""
df_input_clean_all = df_input_px.fillna(method ='pad')
""" excluding the currency exhange rate and forward index"""
df_input_clean = df_input_clean_all.iloc[:,0:22]

"""Generating date series"""
index_rebal_dates = Date_Gen.date_series(df_input_px, incep_date, end_date)

"""Reading Div data from CashDividend file"""
div_file = pd.ExcelFile("C:/Users/sunilbhandari1/Documents/Python Scripts/Bnp/CashDividend.xlsx")
df_div = div_file.parse("Cash_Div")
df_div = df_div.set_index(["Date"])

"""Reading Des file from comp details"""
des_file = pd.ExcelFile("C:/Users/sunilbhandari1/Documents/Python Scripts/Bnp/Comp_Des.xlsx")
df_descrip = des_file.parse("Descrip")
df_descrip = df_descrip.set_index(["BBG"])
"""reading tax details"""
df_tax = des_file.parse("Tax")
df_tax = df_tax.set_index(["Domicile"])
"""reading long term Vol"""
LongTermVol = df_descrip.loc[:,"Long_Term_Volatility"]
LongTermVol = np.transpose(np.array(LongTermVol))
"""reading Gap"""
Gap = df_descrip.loc[:,"Gap"]
Gap = np.transpose(np.array(Gap))
"""reading max and minimum"""
minWeight = df_descrip.loc[:,"MinWeight_EF"]
maxWeight = df_descrip.loc[:,"MaxWeight_EF"]
weigthTuple = []
for k in range(0,len(minWeight)):
    a = minWeight.iloc[k]
    b = maxWeight.iloc[k]
    weigthTuple.append((a,b))    
weigthTuple = tuple(weigthTuple)
""" Processing Data"""
"""Finding Computation dates and Rebalancing Dates"""
#wom = pd.datetools.WeekOfMonth(week=1,weekday = 2)      
#print(incep_date + wom)


"""Calculating Asset(Curr) values in local currency"""
df_asset_curr = pd.DataFrame(index = df_input_clean.index[df_input_clean.index>=incep_date],columns = df_input_clean.columns)
"""print(df_asset_curr.loc[incep_date,"IBTS LN Equity"])
"""
asset_ini = 1
ETF_ini = 1

for column in df_asset_curr:
    for index in df_asset_curr[df_asset_curr.index>=incep_date].index:
        if index == incep_date: 
            df_asset_curr.loc[index,column] = df_input_clean.loc[index,column]
        else:
            """ div calculation """
            if (index in df_div[df_div["Underlying"]==column].index):
                div_temp = df_div[df_div["Underlying"]==column]
                div = div_temp[div_temp.index == index].loc[index,"Amount"]
            else:
                div = 0
            #if ETF_ini == 0:
            """Tax inclusion"""
            temp_dom = df_descrip.loc[column,"Domicile"]  
            temp_ReinRate = df_tax.loc[temp_dom,"Rate"]
            
            """Asset Curr calculation"""
            df_asset_curr.loc[index,column] = asset_ini * ((df_input_clean.loc[index,column] + (div*temp_ReinRate))/ETF_ini)
                  
        asset_ini = df_asset_curr.loc[index,column]       
        ETF_ini = df_input_clean.loc[index,column]
        


"""Calculating Asser value in Base currency"""
df_asset =  pd.DataFrame(index = df_input_clean.index[df_input_clean.index>=incep_date],columns = df_input_clean.columns)
asset_ini = 1
assetccy_ini = 1
index_pre = incep_date


for column in df_asset:
    for index in df_asset[df_asset.index>=incep_date].index:
        cond = df_descrip.loc[column,"BC Asset Formulae"]
        under_FX = df_descrip.loc[column,"Curcy"] + "EUR CURNCY"
        if cond == 1 :
            if index == incep_date:
                df_asset.loc[index,column] = 100
            else:    
                tempFx = df_input_clean_all.loc[index,under_FX]/df_input_clean_all.loc[index_pre,under_FX]
                df_asset.loc[index,column] = asset_ini*(df_asset_curr.loc[index,column]/assetccy_ini)/tempFx
        elif cond == 2:
            if index == incep_date:
                df_asset.loc[index,column] = 100
            else:    
                tempFx = df_input_clean_all.loc[index,under_FX]/df_input_clean_all.loc[index_pre,under_FX]
                tempIndxC = df_input_clean_all.loc[index,"BNPIUSEU Index"]/df_input_clean_all.loc[index_pre,"BNPIUSEU Index"]
                df_asset.loc[index,column] = asset_ini*(1 + (df_asset_curr.loc[index,column]/assetccy_ini/tempFx) - tempIndxC)
        else:
            df_asset.loc[index,column] = df_input_clean.loc[index,column]
        asset_ini = df_asset.loc[index,column]
        assetccy_ini = df_asset_curr.loc[index,column]   
        index_pre = index        


"""calling covariance calculation function"""
calDate = pd.to_datetime('2014-06-20')
df_Cov = CovarianceCalc.covarianceCalculator(df_input_clean,calDate,incep_date)

"""assigning cov to covm to call optimizer"""
#df_Cov= np.multiply(df_Cov,10000)
covm = np.array(df_Cov)

"""calling optimization"""
optimizerOutput = OptimizationFile.optimiz(covm,LongTermVol,weigthTuple,Gap,calDate,df_input_clean)







