# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 17:01:18 2016

@author: Jasmeet.Gujral
"""
import pandas as pd
import numpy as np
import OptimizationFile
import Date_Gen as dg
import CovarianceCalc_updated
import Read_Data as rd
import TrendIndcatorCal
import RegionalFactor
import tvcw
import calendar


incep_date = pd.to_datetime('2014-05-20')
data_incep_date = pd.to_datetime('2013-05-8')
end_date = pd.to_datetime('2014-05-25')
#calDate = pd.to_datetime('2014-05-16')
#calDate = pd.to_datetime('2013-05-10')
"""Input Data"""
df_input_px, df_input_div, df_reg_factor, df_descrip, df_tax, sr_trading_days = rd.read_data(data_incep_date, end_date)

"""Cleaning Data"""

""" Cleaning data to replace nan with previous day value"""
df_input_clean = df_input_px.fillna(method ='pad')
"""reading long term Vol"""
LongTermVol = np.transpose(np.array(df_descrip.loc[:,"Long_Term_Volatility"]))
"""reading Gap"""
Gap = np.transpose(np.array(df_descrip.loc[:,"Gap"]))
"""reading max and minimum"""
minWeight = df_descrip.loc[:,"MinWeight_EF"]
maxWeight = df_descrip.loc[:,"MaxWeight_EF"]
weigthTuple = []
for k in range(0,len(minWeight)):
    a = minWeight.iloc[k]
    b = maxWeight.iloc[k]
    weigthTuple.append((a,b))    
weigthTuple = tuple(weigthTuple)  



"""Date series generation"""
index_calc_days, comp_dates = dg.date_series(df_input_px,sr_trading_days, data_incep_date, end_date)


""" Processing Data"""
"""Normalized timeseries with adjustment for dividend and FX"""
df_asset_curr = pd.DataFrame(index = df_input_clean.index[df_input_clean.index>=data_incep_date],columns = df_input_clean.columns[:-5])
"""Setting initial value as 1"""
df_asset_curr.iloc[0] = 1

for column in df_asset_curr:
    temp_dom = df_descrip.loc[column,"Domicile"]  
    temp_ReinRate = df_tax.loc[temp_dom,"Rate"]
    pre_date = data_incep_date
    for date in df_asset_curr.index[1:]:
        """ div calculation """
        if (date in df_input_div[df_input_div["Underlying"]==column].index):
                div_temp = df_input_div[df_input_div["Underlying"]==column]
                div = div_temp[div_temp.index == date].loc[date,"Amount"]
        else:
             div = 0

        """Asset calculation"""
        df_asset_curr.loc[date,column] = df_asset_curr.loc[pre_date,column] * ((df_input_clean.loc[date,column] + (div*temp_ReinRate))/df_input_clean.loc[pre_date,column])
        
        if not np.isnan(df_input_px.loc[date,column]):          
            pre_date = date
        

"""Calculating Asser value in Base currency"""
df_asset =  pd.DataFrame(index = df_asset_curr.index,columns = df_asset_curr.columns)
df_asset.iloc[0] = df_asset_curr.iloc[0]

"""calculating df_Asset in base currency"""
for column in df_asset:
#column = df_asset.columns[0]    
    date_pre = data_incep_date
    for date in df_asset.index[1:]:
        cond = df_descrip.loc[column,"BC Asset Formulae"]
        under_FX = df_descrip.loc[column,"Curcy"] + "EUR CURNCY"
        tempFx = df_input_clean.loc[date,under_FX]/df_input_clean.loc[date_pre,under_FX]
        tempFXadjr = (df_asset_curr.loc[date,column]/df_asset_curr.loc[date_pre,column])/tempFx
        if cond == 1 :
            df_asset.loc[date,column] = df_asset.loc[date_pre,column]*tempFXadjr
            if not (np.isnan(df_input_px.loc[date,column])): 
                date_pre = date
        elif cond == 2:
            tempIndxC = df_input_clean.loc[date,"BNPIUSEU Index"]/df_input_clean.loc[date_pre,"BNPIUSEU Index"]
            df_asset.loc[date,column] = df_asset.loc[date_pre,column] *(1+tempFXadjr - tempIndxC)
            if not (np.isnan(df_input_px.loc[date,column]) | np.isnan(df_input_px.loc[date,"LQD UP Equity"])): 
                date_pre = date       
#            print("Date is ",date, "Date_pre is ",date_pre)
#            print("DAte Fx ",df_input_clean.loc[date,under_FX], "Pre DAte Fx ",df_input_clean.loc[date_pre,under_FX])
#            print("Date BNPIS ",df_input_clean.loc[date,"BNPIUSEU Index"], "Date_pre BNPISis ",df_input_clean.loc[date_pre,"BNPIUSEU Index"])
        else:
            df_asset.loc[date,column] = df_asset_curr.loc[date,column]
 
"""calculating df_cash"""
df_cash = pd.Series(index = df_asset.index)
predate = data_incep_date
for index in df_cash.index:    
     if index >= data_incep_date:
        if index == data_incep_date:
             df_cash.loc[index] = 100
        else:
             days = (calendar.timegm(index.utctimetuple()) - calendar.timegm(predate.utctimetuple()))/86400
             df_cash.loc[index] = df_cash[:index][-2]*(1+(((df_input_clean.loc[predate,'EONIA Equity'])/100)*(days/360)))
             calendar.timegm(date.utctimetuple()) - calendar.timegm(predate.utctimetuple())
     predate = index    
#     
#"""calling covariance calculation function"""
#calDate = pd.to_datetime('2014-05-16')
#df_Cov = CovarianceCalc_updated.covarianceCalculator(df_asset,calDate,data_incep_date)
#trendIndicator = TrendIndcatorCal.trndCalculator(calDate,df_asset)
#regFactor = RegionalFactor.regFactor(calDate,df_reg_factor)
#expReturn = trendIndicator*LongTermVol*regFactor
#
#"""assigning cov to covm to call optimizer"""
#covm = np.multiply(np.array(df_Cov),1)
#
#"""calling optimization"""
#optimizerOutput = OptimizationFile.optimiz(covm,LongTermVol,weigthTuple,Gap,calDate,df_asset,df_reg_factor,trendIndicator,regFactor,expReturn)

##For repoting purpose
#Optweight = pd.Series(data = optimizerOutput.x)
#OptVol = np.sqrt(np.dot(np.dot(np.transpose(optimizerOutput.x),covm),optimizerOutput.x)) 
#OptRet = np.sum((np.multiply(optimizerOutput.x,expReturn)))
#OptWeightSum =sum(optimizerOutput.x)
#OptGap = np.sum(np.multiply(np.transpose(optimizerOutput.x),Gap))
#zz = regFactor.tolist()
input_file_path = "C:\\Users\\sunilbhandari1\\Desktop\\Python\\BNPICAI_V3\\static_data\\Target_optimal_weights.xlsx"
input_file = pd.ExcelFile(input_file_path)
df_weight_Client = input_file.parse("tow")
#df_weight_Client = df_weight_Client.loc[:,calDate ]
#OptVolClient = np.sqrt(np.dot(np.dot(np.transpose(np.array(df_weight_Client)),covm),df_weight_Client)) 
#OptRetClient = np.sum((np.multiply(np.array(df_weight_Client),expReturn)))
#OptWeightSumClient =sum(df_weight_Client)
#OptGapClient = np.sum(np.multiply(np.transpose(np.array(df_weight_Client)),Gap))

"""Index level calculation"""
df_index = pd.Series(index = df_asset.index)
df_tvcw_series = pd.Series(index = df_asset.index)
df_guw = pd.DataFrame(index = df_asset.index, columns = df_asset.columns )
df_tow = pd.DataFrame(index = df_asset.index, columns = df_asset.columns )
df_weights = pd.DataFrame(index = df_asset.index, columns = df_asset.columns )
df_weightCash = pd.Series(index = df_asset.index)
df_FeesSum = pd.Series(index = df_asset.index)
df_cashSum = pd.Series(index = df_asset.index)
df_assetSum = pd.Series(index = df_asset.index)
df_preeffectiveDate = pd.Series(index = df_asset.index)
for date in df_asset.index:
    InitialOptcalDate = pd.to_datetime('2014-05-16')
#    date = pd.to_datetime('2014-05-23')
    if date >= InitialOptcalDate:
        if date < incep_date:
            if date == InitialOptcalDate:
                
                """calling covariance calculation function"""
                calDate = pd.to_datetime('2014-05-16')
                df_Cov=CovarianceCalc_updated.covarianceCalculator(df_asset,calDate,data_incep_date)
                trendIndicator = TrendIndcatorCal.trndCalculator(calDate,df_asset)
                regFactor = RegionalFactor.regFactor(calDate,df_reg_factor)
                expReturn = trendIndicator*LongTermVol*regFactor                
                """assigning cov to covm to call optimizer"""
                covm = np.multiply(np.array(df_Cov),1)
                
                optimizerOutput = OptimizationFile.optimiz(covm,LongTermVol,weigthTuple,Gap,InitialOptcalDate,df_asset,df_reg_factor,trendIndicator,regFactor,expReturn)
                Optweight = pd.Series(data = optimizerOutput.x)
#                """Overwriting with Cliet's tow"""
#                Optweight = pd.Series(df_weight_Client.loc[:,date ])

            tvcwCal = tvcw.tvcwFunc(Optweight,date,df_asset,20)
            df_tvcw_series.loc[date] = tvcwCal
            guw = Optweight*tvcwCal
            
            df_guw.loc[date] = np.array(guw)
            df_tow.loc[date] = np.array(Optweight)
        elif date == incep_date:
            """Check this as for 19th no index level is available"""
            preeffectiveDate = pd.to_datetime('2014-05-20')
            effectiveDate = pd.to_datetime('2014-05-20')
            tvcwCal = tvcw.tvcwFunc(Optweight,date,df_asset,20)
            df_tvcw_series.loc[date] = tvcwCal
            guw = Optweight*tvcwCal
            df_guw.loc[date] = np.array(guw)
            df_tow.loc[date] = np.array(Optweight)
            """as both index and asset will be one for first"""
            df_weights.loc[date] = np.array(guw)
            """ Check This as well"""
            df_weightCash.loc[date] = 0
            FeesSum = 0
            df_index.loc[date] = 100
        elif date > incep_date:
            print("Running Index calculation for date",date)
                       
            """checking if current date is computational date is yes calculate both tvcw and optimization weights"""
            if sum(comp_dates.isin([date])) == 1:
                
                """calling covariance calculation function"""
                calDate = date
                df_Cov = CovarianceCalc_updated.covarianceCalculator(df_asset,calDate,data_incep_date)
                trendIndicator = TrendIndcatorCal.trndCalculator(calDate,df_asset)
                regFactor = RegionalFactor.regFactor(calDate,df_reg_factor)
                expReturn = trendIndicator*LongTermVol*regFactor                
                """assigning cov to covm to call optimizer"""
                covm = np.multiply(np.array(df_Cov),1)
                
                optimizerOutput = OptimizationFile.optimiz(covm,LongTermVol,weigthTuple,Gap,calDate,df_asset,df_reg_factor,trendIndicator,regFactor,expReturn)
                OptweightNew = pd.Series(data = optimizerOutput.x)
                """overwriting with Clients Tow"""
#                OptweightNew = pd.Series(df_weight_Client.loc[:,date ])

                """getting 2 days after the tow cal where this tow will become effective"""
                df_effectiveDate = df_asset[df_asset.index >= date]
                effectiveDate = df_effectiveDate.index[3]
            if date == effectiveDate:
                """if its 2 days after the tow calculate the weight""" 
#                preeffectiveDate = df_effectiveDate.index[1]                  
                Optweight =  OptweightNew
            tvcwCal = tvcw.tvcwFunc(Optweight,date,df_asset,20)
            df_tvcw_series.loc[date] = tvcwCal
            guw = Optweight*df_tvcw_series[:date][-3]
            df_guw.loc[date] = np.array(guw)
            df_tow.loc[date] = np.array(Optweight)
            
            """Calculating weight based on guw"""
            flagUnitChanged = 0
            for assets in df_weights.columns:
                if (round(df_guw.loc[df_guw[:date].index[-3]][assets],12) == round(df_guw.loc[df_guw[:date].index[-2]][assets],12)) and date != pd.to_datetime('2014-05-21') :
                    df_weights.loc[date,assets] = df_weights.loc[df_weights[:date].index[-2]][assets]
                else:
                    flagUnitChanged = 1
                    df_weights.loc[date,assets] = df_guw.loc[df_guw[:date].index[-2]][assets]*(df_index[:date][-2]/((df_asset.loc[df_asset[:date].index[-2]][assets])))
                    
            if flagUnitChanged == 1 or date == pd.to_datetime('2014-05-21'):
                df_weightCash.loc[date] = (1-sum(df_guw.loc[df_guw[:date].index[-2],]))*(df_index[:date][-2]/df_cash[:date][-2])
                preeffectiveDate = df_index[:date].index[-2]
            else:
                preDate = df_weightCash[:date].index[-2]
                df_weightCash.loc[date] = df_weightCash.loc[preDate]
            
                        
            
            assetSum = 0
            FeesSum = 0
            for column in df_asset.columns:
                assetSum = assetSum + df_weights.loc[date,column]*((df_asset.loc[date,column])-(df_asset.loc[preeffectiveDate,column]))
                if df_index[:date].index[-2] == preeffectiveDate:
                    FeesSum  = FeesSum + df_asset.loc[preeffectiveDate,column]*abs(df_weights.loc[date,column]-df_weights.loc[preeffectiveDate,column])
                else:
                    FeesSum = df_FeesSum[:date][-2]/.0004
               
            if date ==   pd.to_datetime('2014-05-21') :
                FeesSum = 0
                
            df_FeesSum.loc[date] = FeesSum * .0004    
            df_assetSum.loc[date] = assetSum
            cashSum =  df_weightCash.loc[date]*(df_cash.loc[date]-df_cash.loc[preeffectiveDate]) 
            df_cashSum.loc[date] = cashSum      
            df_preeffectiveDate.loc[date] = preeffectiveDate
            df_index.loc[date] = df_index.loc[preeffectiveDate]+assetSum+cashSum-df_FeesSum.loc[date]






