# Importing Libraries

import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from sklearn.linear_model import LassoCV, LinearRegression  # To fit the lasso regression model
from sklearn.model_selection import RepeatedKFold # k-fold cross-validation to find optimal alpha value for penalty term
from sklearn.linear_model import Lasso # To fit the lasso regression model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV # k-fold cross-validation to find optimal alpha value for penalty term
import dcor # For Distance Correlation & Distance Variance
from sklearn.cluster import KMeans # For K-Means clustering
import matplotlib.pyplot as plt # For plotting
from sklearn.tree import DecisionTreeRegressor # For Decision Tree Regression
from fredapi import Fred

# Hyperparameters

tickers = ["XLB","XLV","XLP","XLY","XLE","XLF","XLI","XLK","XLU","XLRE","XLC"]

start_time = datetime.datetime(2018, 6, 29)
end_time = datetime.datetime(2020, 12, 1)


data = {}
for i1,ticker in enumerate(tickers):
    temp_df = web.DataReader(ticker,'yahoo', start_time, end_time).reset_index()
    temp_df.set_index('Date', inplace=True)
    temp_df.index = pd.to_datetime(temp_df.index)
    temp_df = temp_df.resample('1M').mean()
    data[tickers[i1]] = temp_df

# Visualizing downloaded data
for i2 in range(len(tickers)):
    print("Head snapshot of " + tickers[i2] + "\n")
    print(data[tickers[i2]].head(3))
    print(" ")
    print("Tail snapshot of " + tickers[i2] + "\n")
    print(data[tickers[i2]].tail(3))
    print(" ")

fred = Fred(api_key='71f503ef5a8254ace05fa64ea5cbbe78')  ### use your own api key
SP500_data = fred.get_series('SP500')
print(SP500_data.head(3))

data_weekly_production = fred.get_series('AWHMAN')
data_weekly_production = data_weekly_production.to_frame()['2018-06-29':'2020-12-01'].reset_index()
print(data_weekly_production.head(3))

data_weekly_hours = fred.get_series('AWHAECON')
data_weekly_hours = data_weekly_hours.to_frame()['2018-06-29':'2020-12-01'].reset_index()
print(data_weekly_hours.head(3))

data_emp_man = fred.get_series('MANEMP')
data_emp_man = data_emp_man.to_frame()['2018-06-29':'2020-12-01'].reset_index()
print(data_emp_man.head(3))

data_con_sen = fred.get_series('UMCSENT')
data_con_sen = data_con_sen.to_frame()['2018-06-29':'2020-12-01'].reset_index()
print(data_con_sen.head(3))

data_neworder_man = fred.get_series('DGORDER')
data_neworder_man = data_neworder_man.to_frame()['2018-06-29':'2020-12-01'].reset_index()
print(data_neworder_man.head(3))

data_nat_fincon = fred.get_series('NFCINONFINLEVERAGE')
data_nat_fincon = data_nat_fincon.to_frame()['2018-06-29':'2020-12-05']
data_nat_fincon.reset_index(inplace=True)
data_nat_fincon['index'] = data_nat_fincon.loc[:,'index'].astype('datetime64[ns]')

data_nat_fincon = data_nat_fincon[['index',0]]
data_nat_fincon.set_index('index',inplace=True)
data_nat_fincon = data_nat_fincon.resample('M').mean().reset_index()
print(data_nat_fincon.head(3))

data_snp = fred.get_series('SP500')
data_snp = data_snp.to_frame()['2018-06-29':'2020-12-01']
data_snp.reset_index(inplace=True)
data_snp['index'] = data_snp.loc[:,'index'].astype('datetime64[ns]')

data_snp = data_snp[['index',0]]
data_snp.set_index('index',inplace=True)
data_snp = data_snp.resample('M').mean().reset_index()
print(data_snp.head(3))

data_neword_nondef = fred.get_series('NEWORDER')
data_neword_nondef = data_neword_nondef.to_frame()['2018-06-29':'2020-12-01'].reset_index()
print(data_neword_nondef.head(3))

data_4wma = fred.get_series('IC4WSA')
data_4wma = data_4wma.to_frame()['2018-06-29':'2020-12-10']
data_4wma.reset_index(inplace=True)
data_4wma['index'] = data_4wma.loc[:,'index'].astype('datetime64[ns]')

data_4wma = data_4wma[['index',0]]
data_4wma.set_index('index',inplace=True)
data_4wma = data_4wma.resample('M').mean().reset_index()
print(data_4wma.head(3))

data_permit = fred.get_series('PERMIT')
data_permit = data_permit.to_frame()['2018-06-29':'2020-12-01'].reset_index()
print(data_permit.head(3))

for i3 in range(len(data)):
    temp_ticker = tickers[i3]
    temp_data = data[temp_ticker]
    temp_data["Daily Returns"] = temp_data["Adj Close"].pct_change(1)  # 1 for ONE DAY lookback
    data[temp_ticker] = temp_data
# Printing and checking for the returns column
for i4 in range(len(data)):
    print("\n" + "Head snapshot of " + tickers[i4] + "\n")
    print(data[tickers[i4]].head(2))

monthly_returns = {}
for i3 in range(len(data)):
    temp_ticker = tickers[i3]
    temp_data = data[temp_ticker]
    temp_data = temp_data.pct_change()  # 1 for ONE DAY lookback

    monthly_returns[temp_ticker] = temp_data

# Visualizing monthly returns, of High, Low, Open, Close, Volume & Adj Close columns
# For the rest of the group work, we will be mostly working with Adj Close returns data
for i4 in range(len(monthly_returns)):
    print("\n" + "Head snapshot of " + tickers[i4] + "\n")
    print(monthly_returns[tickers[i4]].head(3))


reg_data_lei = pd.concat([data_weekly_production[0],data_weekly_hours[0],data_emp_man[0],
                          data_con_sen[0],data_neworder_man[0],data_nat_fincon[0],data_snp[0],
                          data_neword_nondef[0],data_4wma[0],data_permit[0]],axis = 1)

print(reg_data_lei.head(3))

reg_data_lei.drop(reg_data_lei.tail(1).index,inplace=True)

reg_data_lei.columns = ['data_weekly_production','data_weekly_hours','data_emp_man','data_con_sen','data_neworder_man',
                        'data_nat_fincon','data_snp','data_neword_nondef','data_4wma','data_permit']

print(reg_data_lei.head(3))

# Finding returns for each of LEI
reg_data_lei_returns = {}
for i5 in range(reg_data_lei.shape[1]):
    temp_data = reg_data_lei.loc[:,reg_data_lei.columns[i5]]
    temp_data_returns = temp_data.pct_change()  # 1 for ONE period lookback
    reg_data_lei_returns[reg_data_lei.columns[i5] + '_returns'] = temp_data_returns
    print("\n" + "For " + str(reg_data_lei.columns[i5]) + ", returns structure:")
    print(temp_data_returns.head(3))

# Running a regression modeling each of the ETF returns on the 10 Leading EI (LEI)
reg_data_lei_returns_df = pd.DataFrame(reg_data_lei_returns).dropna()
predictor_data = pd.DataFrame(index=range(reg_data_lei_returns_df.shape[0]),columns=tickers)
temp_max_len = len(monthly_returns[tickers[0]].index)
temp_start = temp_max_len - reg_data_lei_returns_df.shape[0]
temp_index = monthly_returns[tickers[0]].index[temp_start:temp_max_len]
lei_predictor_data = pd.DataFrame(index=temp_index,columns=tickers)
for i6 in range(0,len(monthly_returns)):
    temp_df = monthly_returns[tickers[i6]]['Adj Close'].iloc[temp_start:monthly_returns[tickers[i6]].shape[0]]
    lei_predictor_data[tickers[i6]] = temp_df
    print("\n" + "For: " + tickers[i6])
    print(temp_df.head(3))

for i8,column in enumerate(lei_predictor_data):
    temp_list = lei_predictor_data[tickers[i8]].to_list()
    temp_df = pd.DataFrame({tickers[i8]:temp_list})
    temp_df = temp_df.set_index(reg_data_lei_returns_df.index)
    temp_lr_df = pd.concat([reg_data_lei_returns_df.dropna(), temp_df.dropna()], axis=1)
    temp_X = temp_lr_df.iloc[:, np.arange(0, temp_lr_df.shape[1]-1, 1).tolist()]
    temp_Y = temp_lr_df.iloc[:, -1]
    temp_lr_model = LinearRegression()
    temp_lr_model.fit(temp_X, temp_Y)
    temp_linear_regression = LinearRegression().fit(temp_X,temp_Y.fillna(0))
    print("\n" + "For security: " + tickers[i8] + "; R-Sqaured coefficient is: " +
          str(round(temp_linear_regression.score(temp_X,temp_Y),2)) +
          ", coefficients of multiple regression are: " + str(temp_linear_regression.coef_))

data_snp = fred.get_series('SP500')
data_snp = data_snp.to_frame()['2018-06-29':'2020-12-01']
data_snp.reset_index(inplace=True)
data_snp['index'] = data_snp.loc[:,'index'].astype('datetime64[ns]')

data_snp = data_snp[['index',0]]
data_snp.set_index('index',inplace=True)
data_snp = data_snp.resample('M').mean().reset_index()
print(data_snp.head(3))

data_rsafs = fred.get_series('RSAFS')
data_rsafs = data_rsafs.to_frame()['2018-06-29':'2020-12-01']
data_rsafs.reset_index(inplace=True)
data_rsafs['index'] = data_rsafs.loc[:,'index'].astype('datetime64[ns]')

data_rsafs = data_rsafs[['index',0]]
data_rsafs.set_index('index',inplace=True)
data_rsafs = data_rsafs.resample('M').mean().reset_index()
print(data_snp.head(3))

data_payems = fred.get_series('PAYEMS')
data_payems = data_payems.to_frame()['2018-06-29':'2020-12-01']
data_payems.reset_index(inplace=True)
data_payems['index'] = data_payems.loc[:,'index'].astype('datetime64[ns]')

data_payems = data_payems[['index',0]]
data_payems.set_index('index',inplace=True)
data_payems = data_payems.resample('M').mean().reset_index()
print(data_payems.head(3))

data_pi = fred.get_series('PI')
data_pi = data_pi.to_frame()['2018-06-29':'2020-12-01']
data_pi.reset_index(inplace=True)
data_pi['index'] = data_pi.loc[:,'index'].astype('datetime64[ns]')

data_pi = data_pi[['index',0]]
data_pi.set_index('index',inplace=True)
data_pi = data_pi.resample('M').mean().reset_index()
print(data_pi.head(3))

data_indpro = fred.get_series('INDPRO')
data_indpro = data_indpro.to_frame()['2018-06-29':'2020-12-01']
data_indpro.reset_index(inplace=True)
data_indpro['index'] = data_indpro.loc[:,'index'].astype('datetime64[ns]')

data_indpro = data_indpro[['index',0]]
data_indpro.set_index('index',inplace=True)
data_indpro = data_indpro.resample('M').mean().reset_index()
print(data_indpro.head(3))

reg_data_cei = pd.concat([data_rsafs[0],data_payems[0],data_pi[0],data_indpro[0]],axis = 1)

reg_data_cei.columns = ['data_rsafs','data_payems','data_pi','data_indpro']

# Finding returns for each of CEI
reg_data_cei_returns = {}
for i9 in range(reg_data_cei.shape[1]):
    temp_data = reg_data_cei.loc[:,reg_data_cei.columns[i9]]
    temp_data_returns = temp_data.pct_change()  # 1 for ONE period lookback
    reg_data_cei_returns[reg_data_cei.columns[i9] + '_returns'] = temp_data_returns
    print("\n" + "For " + str(reg_data_cei.columns[i9]) + ", returns structure:")
    print(temp_data_returns.head(3))

# Running a regression modeling each of the ETF returns on the 4 Co-incidental EI (CEI)
reg_data_cei_returns_df = pd.DataFrame(reg_data_cei_returns).dropna()
cei_predictor_data = pd.DataFrame(index=range(reg_data_lei_returns_df.shape[0]),columns=tickers)
temp_max_len = len(monthly_returns[tickers[0]].index)
temp_start = temp_max_len - reg_data_cei_returns_df.shape[0]
temp_index = monthly_returns[tickers[0]].index[temp_start:temp_max_len]
cei_predictor_data = pd.DataFrame(index=temp_index,columns=tickers)
for i10 in range(0,len(monthly_returns)):
    temp_df = monthly_returns[tickers[i10]]['Adj Close'].iloc[temp_start:monthly_returns[tickers[i10]].shape[0]]
    cei_predictor_data[tickers[i10]] = temp_df
    print("\n" + "For: " + tickers[i10])
    print(temp_df.head(3))

for i11,column in enumerate(cei_predictor_data):
    temp_list = cei_predictor_data[tickers[i11]].to_list()
    temp_df = pd.DataFrame({tickers[i11]:temp_list})
    temp_df = temp_df.set_index(reg_data_cei_returns_df.index)
    temp_lr_df = pd.concat([reg_data_cei_returns_df.dropna(), temp_df.dropna()], axis=1)
    temp_X = temp_lr_df.iloc[:, np.arange(0, temp_lr_df.shape[1]-1, 1).tolist()]
    temp_Y = temp_lr_df.iloc[:, -1]
    temp_lr_model = LinearRegression()
    temp_lr_model.fit(temp_X, temp_Y)
    temp_linear_regression = LinearRegression().fit(temp_X,temp_Y.fillna(0))
    print("\n" + "For security: " + tickers[i11] + "; R-Sqaured coefficient is: " +
          str(round(temp_linear_regression.score(temp_X,temp_Y),2)) +
          ", coefficients of multiple regression are: " + str(temp_linear_regression.coef_))

data_mprime = fred.get_series('MPRIME')
data_mprime = data_mprime.to_frame()['2018-06-29':'2020-12-01']
data_mprime.reset_index(inplace=True)
data_mprime['index'] = data_mprime.loc[:,'index'].astype('datetime64[ns]')

data_mprime = data_mprime[['index',0]]
data_mprime.set_index('index',inplace=True)
data_mprime = data_mprime.resample('M').mean().reset_index()
print(data_mprime.head(3))

data_busloans = fred.get_series('BUSLOANS')
data_busloans = data_busloans.to_frame()['2018-06-29':'2020-12-01']
data_busloans.reset_index(inplace=True)
data_busloans['index'] = data_busloans.loc[:,'index'].astype('datetime64[ns]')

data_busloans = data_busloans[['index',0]]
data_busloans.set_index('index',inplace=True)
data_busloans = data_busloans.resample('M').mean().reset_index()
print(data_busloans.head(3))

data_totalsl = fred.get_series('TOTALSL')
data_totalsl = data_totalsl.to_frame()['2018-06-29':'2020-12-01']
data_totalsl.reset_index(inplace=True)
data_totalsl['index'] = data_totalsl.loc[:,'index'].astype('datetime64[ns]')

data_totalsl = data_totalsl[['index',0]]
data_totalsl.set_index('index',inplace=True)
data_totalsl = data_totalsl.resample('M').mean().reset_index()
print(data_totalsl.head(3))

data_cpiogsns = fred.get_series('CPIOGSNS')
data_cpiogsns = data_cpiogsns.to_frame()['2018-06-29':'2020-12-01']
data_cpiogsns.reset_index(inplace=True)
data_cpiogsns['index'] = data_cpiogsns.loc[:,'index'].astype('datetime64[ns]')

data_cpiogsns = data_cpiogsns[['index',0]]
data_cpiogsns.set_index('index',inplace=True)
data_cpiogsns = data_cpiogsns.resample('M').mean().reset_index()
print(data_cpiogsns.head(3))

data_isratio = fred.get_series('ISRATIO')
data_isratio = data_isratio.to_frame()['2018-06-29':'2020-12-01']
data_isratio.reset_index(inplace=True)
data_isratio['index'] = data_isratio.loc[:,'index'].astype('datetime64[ns]')

data_isratio = data_isratio[['index',0]]
data_isratio.set_index('index',inplace=True)
data_isratio = data_isratio.resample('M').mean().reset_index()
print(data_isratio.head(3))

data_civpart = fred.get_series('CIVPART')
data_civpart = data_civpart.to_frame()['2018-06-29':'2020-12-01']
data_civpart.reset_index(inplace=True)
data_civpart['index'] = data_civpart.loc[:,'index'].astype('datetime64[ns]')

data_civpart = data_civpart[['index',0]]
data_civpart.set_index('index',inplace=True)
data_civpart = data_civpart.resample('M').mean().reset_index()
print(data_civpart.head(3))

data_uempmed = fred.get_series('UEMPMED')
data_uempmed = data_uempmed.to_frame()['2018-06-29':'2020-12-01']
data_uempmed.reset_index(inplace=True)
data_uempmed['index'] = data_uempmed.loc[:,'index'].astype('datetime64[ns]')

data_uempmed = data_uempmed[['index',0]]
data_uempmed.set_index('index',inplace=True)
data_uempmed = data_uempmed.resample('M').mean().reset_index()
print(data_civpart.head(3))

reg_data_lag = pd.concat([data_mprime[0],data_busloans[0],data_totalsl[0],data_cpiogsns[0],
                          data_isratio[0],data_civpart[0],data_uempmed[0]],axis = 1)

reg_data_lag.columns= ['data_mprime','data_busloans','data_totalsl','data_cpiogsns',
                       'data_isratio','data_civpart','data_uempmed']

# Finding returns for each of LAG
reg_data_lag_returns = {}
for i12 in range(reg_data_lag.shape[1]):
    temp_data = reg_data_lag.loc[:,reg_data_lag.columns[i12]]
    temp_data_returns = temp_data.pct_change()  # 1 for ONE period lookback
    reg_data_lag_returns[reg_data_lag.columns[i12] + '_returns'] = temp_data_returns
    print("\n" + "For " + str(reg_data_lag.columns[i12]) + ", returns structure:")
    print(temp_data_returns.head(3))

# Running a regression modeling each of the ETF returns on the 7 Lagging EI (LAG)
reg_data_lag_returns_df = pd.DataFrame(reg_data_lag_returns).dropna()
lag_predictor_data = pd.DataFrame(index=range(reg_data_lag_returns_df.shape[0]),columns=tickers)
temp_max_len = len(monthly_returns[tickers[0]].index)
temp_start = temp_max_len - reg_data_lag_returns_df.shape[0]
temp_index = monthly_returns[tickers[0]].index[temp_start:temp_max_len]
lag_predictor_data = pd.DataFrame(index=temp_index,columns=tickers)
for i13 in range(0,len(monthly_returns)):
    temp_df = monthly_returns[tickers[i13]]['Adj Close'].iloc[temp_start:monthly_returns[tickers[i13]].shape[0]]
    lag_predictor_data[tickers[i13]] = temp_df
    print("\n" + "For: " + tickers[i13])
    print(temp_df.head(3))

for i14,column in enumerate(lag_predictor_data):
    temp_list = lag_predictor_data[tickers[i14]].to_list()
    temp_df = pd.DataFrame({tickers[i14]:temp_list})
    temp_df = temp_df.set_index(reg_data_lag_returns_df.index)
    temp_lr_df = pd.concat([reg_data_lag_returns_df.dropna(), temp_df.dropna()], axis=1)
    temp_X = temp_lr_df.iloc[:, np.arange(0, temp_lr_df.shape[1]-1, 1).tolist()]
    temp_Y = temp_lr_df.iloc[:, -1]
    temp_lr_model = LinearRegression()
    temp_lr_model.fit(temp_X, temp_Y)
    temp_linear_regression = LinearRegression().fit(temp_X,temp_Y.fillna(0))
    print("\n" + "For security: " + tickers[i14] + "; R-Sqaured coefficient is: " +
          str(round(temp_linear_regression.score(temp_X,temp_Y),2)) +
          ", coefficients of multiple regression are: " + str(temp_linear_regression.coef_))

lei_cei_lag_data = pd.concat([reg_data_lei_returns_df,reg_data_cei_returns_df,reg_data_lag_returns_df],axis = 1)
print(lei_cei_lag_data.head(2))

# Running a regression modeling each of the ETF returns on the 21 EIs
lei_cei_lag_data_returns_df = pd.DataFrame(lei_cei_lag_data).dropna()
lei_cei_lag_predictor_data = pd.DataFrame(index=range(lei_cei_lag_data_returns_df.shape[0]),columns=tickers)
temp_max_len = len(monthly_returns[tickers[0]].index)
temp_start = temp_max_len - lei_cei_lag_data_returns_df.shape[0]
temp_index = monthly_returns[tickers[0]].index[temp_start:temp_max_len]
lei_cei_lag_predictor_data = pd.DataFrame(index=temp_index,columns=tickers)
for i15 in range(0,len(monthly_returns)):
    temp_df = monthly_returns[tickers[i15]]['Adj Close'].iloc[temp_start:monthly_returns[tickers[i15]].shape[0]]
    lei_cei_lag_predictor_data[tickers[i15]] = temp_df
    print("\n" + "For: " + tickers[i15])
    print(temp_df.head(3))

for i16,column in enumerate(lei_cei_lag_predictor_data):
    temp_list = lei_cei_lag_predictor_data[tickers[i16]].to_list()
    temp_df = pd.DataFrame({tickers[i16]:temp_list})
    temp_df = temp_df.set_index(lei_cei_lag_data_returns_df.index)
    temp_lr_df = pd.concat([lei_cei_lag_data_returns_df.dropna(), temp_df.dropna()], axis=1)
    temp_X = temp_lr_df.iloc[:, np.arange(0, temp_lr_df.shape[1]-1, 1).tolist()]
    temp_Y = temp_lr_df.iloc[:, -1]
    temp_lr_model = LinearRegression()
    temp_lr_model.fit(temp_X, temp_Y)
    temp_linear_regression = LinearRegression().fit(temp_X,temp_Y.fillna(0))
    print("\n" + "For security: " + tickers[i16] + "; R-Sqaured coefficient is: " +
          str(round(temp_linear_regression.score(temp_X,temp_Y),2)) +
          ", coefficients of multiple regression are: " + str(temp_linear_regression.coef_))

for i17 in range(lei_cei_lag_data.shape[1]):
    print("Factor #" + str(i17+1) + ": " + lei_cei_lag_data.columns[i17])

# Running a LASSO model on each of the ETF returns using 21 EIs
for i18, column in enumerate(lei_cei_lag_predictor_data):
    temp_list = lei_cei_lag_predictor_data[tickers[i18]].to_list()
    temp_df = pd.DataFrame({tickers[i18]: temp_list})
    temp_df = temp_df.set_index(lei_cei_lag_data.index)
    temp_lasso_df = pd.concat([lei_cei_lag_data.dropna(), temp_df.dropna()], axis=1)
    temp_X = temp_lasso_df.iloc[:, np.arange(0, temp_lasso_df.shape[1] - 1, 1).tolist()]
    temp_Y = temp_lasso_df.iloc[:, -1]

    temp_Lasso_model = Lasso()

    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    # define grid
    grid = dict()
    grid['alpha'] = np.arange(0, 1, 0.01)

    # define search
    search = GridSearchCV(temp_Lasso_model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

    # evaluate model
    scores = cross_val_score(temp_Lasso_model, temp_X, temp_Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

    # perform the search
    results = search.fit(temp_X, temp_Y)

    # summarize
    # print('MAE: %.3f' % results.best_score_)
    print('\n' + 'For: ' + tickers[i18] + ', Lasso Config: %s' % results.best_params_)

    scores = np.absolute(scores)
    print('For: ' + tickers[i18] + ', Lasso Mean(STD) MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

for i19,column in enumerate(lei_cei_lag_predictor_data):
    temp_list = lei_cei_lag_predictor_data[tickers[i19]].to_list()
    temp_df = pd.DataFrame({tickers[i19]:temp_list})
    temp_df = temp_df.set_index(lei_cei_lag_data_returns_df.index)
    temp_dcor_df = pd.concat([lei_cei_lag_data_returns_df.dropna(), temp_df.dropna()], axis=1)
    temp_X = temp_dcor_df.iloc[:, np.arange(0, temp_dcor_df.shape[1]-1, 1).tolist()]
    temp_Y = temp_dcor_df.iloc[:, -1]
    print("For security: " + tickers[i19] + ", distance coerelation is: " +
          str(round(dcor.distance_correlation(temp_X,temp_Y),2)))

for i20,column in enumerate(lei_cei_lag_predictor_data):
    temp_list = lei_cei_lag_predictor_data[tickers[i20]].to_list()
    temp_df = pd.DataFrame({tickers[i20]:temp_list})
    temp_df = temp_df.set_index(lei_cei_lag_data_returns_df.index)
    temp_dcov_df = pd.concat([lei_cei_lag_data_returns_df.dropna(), temp_df.dropna()], axis=1)
    temp_X = temp_dcov_df.iloc[:, np.arange(0, temp_dcov_df.shape[1]-1, 1).tolist()]
    temp_Y = temp_dcov_df.iloc[:, -1]
    print("For security: " + tickers[i20] + ", distance covariance is: " +
          str(round(dcor.distance_covariance(temp_X,temp_Y),2)))

# K-Means of 11 ETF returns
kmeans = KMeans(n_clusters=3).fit(lei_cei_lag_predictor_data)
centroids = kmeans.cluster_centers_
print(centroids)

# Decision Tree Regression
for i21,column in enumerate(lei_cei_lag_predictor_data):
    temp_list = lei_cei_lag_predictor_data[tickers[i21]].to_list()
    temp_df = pd.DataFrame({tickers[i21]:temp_list})
    temp_df = temp_df.set_index(lei_cei_lag_data_returns_df.index)
    temp_tree_df = pd.concat([lei_cei_lag_data_returns_df.dropna(), temp_df.dropna()], axis=1)
    temp_X = temp_tree_df.iloc[:, np.arange(0, temp_tree_df.shape[1]-1, 1).tolist()]
    temp_Y = temp_tree_df.iloc[:, -1]
    regression_model = DecisionTreeRegressor(criterion="mse",min_samples_leaf=5)
    regression_model = regression_model.fit(temp_X,temp_Y)
    print("\n" + "For security: " + tickers[i21] + ", tree impurity is: " +
          str(regression_model.tree_.impurity))