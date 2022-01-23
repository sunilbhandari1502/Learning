# Importing Libraries
import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from itertools import combinations
from pypfopt import CLA

from PyPortfolioOpt.pypfopt.efficient_frontier import efficient_frontier
from PyPortfolioOpt.pypfopt import risk_models
from PyPortfolioOpt.pypfopt import expected_returns

# Hyperparameters

tickers = ["XLB","XLV","XLP","XLY","XLE","XLF","XLI","XLK","XLU","XLRE","XLC"]

start_time = datetime.datetime(2018, 6, 29)
end_time = datetime.datetime(2020, 12, 1)

data = {}

for i1,ticker in enumerate(tickers):
    data[tickers[i1]] = web.DataReader(ticker,'yahoo', start_time, end_time).reset_index()

# Adding returns to the data; visualizing
for i3 in range(len(data)):
    temp_ticker = tickers[i3]
    temp_data = data[temp_ticker]
    temp_data["Daily Returns"] = temp_data["Adj Close"].pct_change(1) # 1 for 1 day lookback
    temp_data = temp_data.dropna()
    print("\n" + "Head snapshot of " + tickers[i3])
    print(temp_data.head(3))
    data[temp_ticker] = temp_data

# Adj_Close_price has only adjusted close price of all the tickers
XLB_XLV_Adj_Close_price = pd.concat([data[tickers[0]]['Adj Close'],data[tickers[1]]['Adj Close']],axis = 1)
XLB_XLV_Adj_Close_price.columns = ['XLB','XLV']
XLB_XLV_returns = XLB_XLV_Adj_Close_price.pct_change().dropna()
# Computing row-wise mean
XLB_XLV_returns['Equi_Weighted_Portfolio_Returns'] = XLB_XLV_returns.mean(axis = 1)
# Return of an equi weighted portfolio
XLB_XLV_returns.head(5)

for i4 in range(0,2):
    print("\n" + "Brief Overview of 'Daily Returns' of " + tickers[i4] + "\n")
    temp_data = data[tickers[i4]]["Daily Returns"]
    temp_data_description = temp_data.describe()
    print(temp_data_description)

XLB_XLV_returns.iloc[:,:2].corr()


# First input is dataframe with Adjusted Close prices of all securities
# Second input is list of weights, by default equally weighted
def WeightedReturn(df, weights=0):
    no_assets = len(df.columns)
    if weights == 0:
        weights = [1 / no_assets for i5 in range(no_assets)]

    ret = df.pct_change().dropna()
    ret['Weighted_Returns'] = ret.mul(weights, axis='columns').sum(axis=1)

    return ret['Weighted_Returns']

# Returns of the portfolio visualised
weighted_returns_example = WeightedReturn(XLB_XLV_returns.iloc[:,:2])
print(weighted_returns_example.head(3))

# Input is dataframe (single column) with Returns of a securities
def PortfolioStdDev(df):
    return df[np.isfinite(df)].describe().to_frame().T['std']

# STD of the portfolio visualised
PortfolioStdDev(weighted_returns_example)

# Simulating different portfolios to eventually get efficient frontier
# Note: Only pass the portfolio with the adjusted close prices of your assets - everthing else will be computed
def EfficientFrontier(weighted_portfolio):
    log_ret = np.log(weighted_portfolio/weighted_portfolio.shift(1))
    log_ret.dropna(inplace = True)
    np.random.seed(10)
    num_ports = 1000
    all_weights = np.zeros((num_ports,len(weighted_portfolio.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)
    for i5 in range(num_ports):
        # Weights
        weights = np.array(np.random.random(weighted_portfolio.shape[1]))
        # Normalize weights
        weights = weights/np.sum(weights)
        # Record weights
        all_weights[i5,:] = weights
        # Expected return
        ret_arr[i5] = np.sum((log_ret.mean()*weights*252))
        # Expected volatility
        vol_arr[i5] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*252, weights)))
        # Sharpe Ratio
        sharpe_arr[i5] = ret_arr[i5]/vol_arr[i5]
    # End of 'for i5 in range(num_ports):'
    return sharpe_arr,vol_arr,ret_arr,all_weights

# Passing the two ETFs and trying differeent weight combinations
sharpe_arr,vol_arr,ret_arr,EF_weights = EfficientFrontier(XLB_XLV_Adj_Close_price)
print("Max Sharp Ratio: {}".format(sharpe_arr.max()))

print(EF_weights[0,:])

max_sr_ret = ret_arr[sharpe_arr.argmax()]
max_sr_vol = vol_arr[sharpe_arr.argmax()]

plt.figure(figsize=(12,8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(max_sr_vol, max_sr_ret,c='red', s=50) # Optimal point
plt.show()

weighted_portfolio = XLB_XLV_Adj_Close_price
log_ret = np.log(weighted_portfolio/weighted_portfolio.shift(1))
log_ret.dropna(inplace = True)
np.random.seed(10)
num_ports = 1000
all_weights = np.zeros((num_ports,len(weighted_portfolio.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)
correl_matrix = matrix = np.array([1,-1,-1,1]).reshape(2,2)

for i31 in range(num_ports):
    # Weights
    weights = np.array(np.random.random(weighted_portfolio.shape[1]))
    # Normalize weights
    weights = weights/np.sum(weights)
    # Record weights
    all_weights[i31,:] = weights
    # Expected return
    ret_arr[i31] = np.sum((log_ret.mean()*weights*252))
    # Expected volatility
    vol_arr[i31] = np.sqrt(np.dot(weights.T, np.dot(correl_matrix*252, weights)))
    # Sharpe Ratio
    sharpe_arr[i31] = ret_arr[i31]/vol_arr[i31]

print("Max Sharp Ratio: {}".format(sharpe_arr.max()))

max_sr_ret = ret_arr[sharpe_arr.argmax()]
max_sr_vol = vol_arr[sharpe_arr.argmax()]

plt.figure(figsize=(12,8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio, Correl: -1')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(max_sr_vol, max_sr_ret,c='red', s=50) # Optimal point
plt.show()

weighted_portfolio = XLB_XLV_Adj_Close_price
log_ret = np.log(weighted_portfolio/weighted_portfolio.shift(1))
log_ret.dropna(inplace = True)
np.random.seed(10)
num_ports = 1000
all_weights = np.zeros((num_ports,len(weighted_portfolio.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)
correl_matrix = np.array([1,0,0,1]).reshape(2,2)

for i32 in range(num_ports):
    # Weights
    weights = np.array(np.random.random(weighted_portfolio.shape[1]))
    # Normalize weights
    weights = weights/np.sum(weights)
    # Record weights
    all_weights[i32,:] = weights
    # Expected return
    ret_arr[i32] = np.sum((log_ret.mean()*weights*252))
    # Expected volatility
    vol_arr[i32] = np.sqrt(np.dot(weights.T, np.dot(correl_matrix*252, weights)))
    # Sharpe Ratio
    sharpe_arr[i32] = ret_arr[i32]/vol_arr[i32]

print("Max Sharp Ratio: {}".format(sharpe_arr.max()))

max_sr_ret = ret_arr[sharpe_arr.argmax()]
max_sr_vol = vol_arr[sharpe_arr.argmax()]

plt.figure(figsize=(12,8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio, Correl: -1')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(max_sr_vol, max_sr_ret,c='red', s=50) # Optimal point
plt.show()

percent_trim = 0.05 # 0.1 means 10% person

trimmed_data = {}

for i_trim in range(len(data)):
    temp_non_trimmed_data_returns = data[tickers[i_trim]]['Daily Returns'].dropna().tolist()
    temp_sorted_returns = sorted(temp_non_trimmed_data_returns)
    temp_how_many_to_trim = round(percent_trim * len(temp_sorted_returns))
    trimmed_daily_returns = temp_sorted_returns[temp_how_many_to_trim:-temp_how_many_to_trim]
    trimmed_data[tickers[i_trim]] = trimmed_daily_returns
    print("Non trimmed returns for '" + tickers[i_trim] + "': " + str(temp_non_trimmed_data_returns[0:3]))
    print("Trimmed returns for '" + tickers[i_trim] + "': " + str(trimmed_daily_returns[0:3]) + "\n")

log_ret_trimmed = pd.DataFrame({'XLB': trimmed_data['XLB'],'XLV': trimmed_data['XLV']})
log_ret_trimmed.dropna(inplace = True)
np.random.seed(10)
num_ports = 1000
all_weights = np.zeros((num_ports,log_ret_trimmed.shape[1]))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for i35 in range(num_ports):
    # Weights
    weights = np.array(np.random.random(weighted_portfolio.shape[1]))
    # Normalize weights
    weights = weights/np.sum(weights)
    # Record weights
    all_weights[i35,:] = weights
    # Expected return
    ret_arr[i35] = np.sum((log_ret_trimmed.mean()*weights*252))
    # Expected volatility
    vol_arr[i35] = np.sqrt(np.dot(weights.T, np.dot(log_ret_trimmed.cov()*252, weights)))
    # Sharpe Ratio
    sharpe_arr[i35] = ret_arr[i35]/vol_arr[i35]

print("Max Sharp Ratio: {}".format(sharpe_arr.max()))

max_sr_ret = ret_arr[sharpe_arr.argmax()]
max_sr_vol = vol_arr[sharpe_arr.argmax()]
plt.figure(figsize=(12,8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(max_sr_vol, max_sr_ret,c='red', s=50) # Optimal point
plt.show()

## Adj_Close_price has only adjusted close price of all the tickers
XLB_XLV_XLP_Adj_Close_price = pd.concat([data[tickers[0]]['Adj Close'],data[tickers[1]]['Adj Close'],
                                    data[tickers[2]]['Adj Close']],axis = 1)
XLB_XLV_XLP_Adj_Close_price.columns = ['XLB','XLV','XLP']
XLB_XLV_XLP_returns = XLB_XLV_XLP_Adj_Close_price.pct_change().dropna()

# Computing Correlation
XLB_XLV_XLP_returns.corr()

# Passing the three ETFs and trying differeent weight combinations
sharpe_arr_3_ETF,vol_arr_3_ETF,ret_arr_3_ETF,EF_weights_3_ETF = EfficientFrontier(XLB_XLV_XLP_Adj_Close_price)
print("Max Sharp Ratio: {}".format(sharpe_arr_3_ETF.max()))

max_sr_ret_3_ETF = ret_arr_3_ETF[sharpe_arr_3_ETF.argmax()]
max_sr_vol_3_ETF = vol_arr_3_ETF[sharpe_arr_3_ETF.argmax()]
plt.figure(figsize=(12,8))
plt.scatter(vol_arr_3_ETF, ret_arr_3_ETF, c=sharpe_arr_3_ETF, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(max_sr_vol_3_ETF, max_sr_ret_3_ETF,c='red', s=50) # Optimal point
plt.show()

start_time_2019 = datetime.datetime(2018, 12, 31)
end_time_2019 = datetime.datetime(2019, 12, 31)

data_2019 = {}

for i6,ticker in enumerate(tickers):
    data_2019[tickers[i6]] = web.DataReader(ticker,'yahoo', start_time_2019, end_time_2019).reset_index()

# Adding returns to the data; visualizing
for i7 in range(len(data)):
    temp_ticker = tickers[i7]
    temp_data = data_2019[temp_ticker]
    temp_data["Daily Returns"] = temp_data["Adj Close"].pct_change(1) # 1 for 1 day lookback
    temp_data = temp_data.dropna()
    print("\n" + "Head snapshot of " + tickers[i7])
    print(temp_data.head(3))
    data_2019[temp_ticker] = temp_data

# Making all the combinations
ticker_indices = np.arange(0, len(tickers)).tolist()
all_portfolio_of_3_combinations = combinations(ticker_indices, 3)
ETF_Combinations_Performance = pd.DataFrame(columns=['Security_1', 'Security_2', 'Security_3',
                                                     'Max_Sharpe', 'Security_1_weight',
                                                     'Security_2_weight', 'Security_3_weight'])
# Calculating Sharpe of these portfolio
for i8 in list(all_portfolio_of_3_combinations):
    temp_security_1 = tickers[i8[0]]
    temp_security_2 = tickers[i8[1]]
    temp_security_3 = tickers[i8[2]]

    ## Adj_Close_price has only adjusted close price of all the tickers
    temp_Adj_Close_price = pd.concat([data_2019[temp_security_1]['Adj Close'],
                                      data_2019[temp_security_2]['Adj Close'],
                                      data_2019[temp_security_3]['Adj Close']], axis=1)
    temp_Adj_Close_price.columns = [temp_security_1, temp_security_2, temp_security_3]

    # Passing the three ETFs and trying different weight combinations
    sharpe_arr_temp, vol_arr_temp, ret_arr_temp, EF_weights_temp = EfficientFrontier(temp_Adj_Close_price)

    # Max Sharpe
    temp_Max_Sharpe = sharpe_arr_temp.max()

    # Optimal Weights
    temp_best_weights = EF_weights_temp[sharpe_arr_temp.argmax()]
    temp_best_weight_1 = temp_best_weights[0]
    temp_best_weight_2 = temp_best_weights[1]
    temp_best_weight_3 = temp_best_weights[2]

    print("Max Sharp Ratio of portfolio of {}({}%), {}({}%), {}({}%): {}.".format(temp_security_1,
                                                                                  round(100 * temp_best_weight_1, 2),
                                                                                  temp_security_2,
                                                                                  round(100 * temp_best_weight_2, 2),
                                                                                  temp_security_3,
                                                                                  round(100 * temp_best_weight_3, 2),
                                                                                  round(temp_Max_Sharpe, 2)))

    # Appending analysis data to 'ETF_Combinations_Performance'
    ETF_Combinations_Performance = ETF_Combinations_Performance.append({'Security_1': temp_security_1,
                                                                        'Security_2': temp_security_2,
                                                                        'Security_3': temp_security_3,
                                                                        'Max_Sharpe': temp_Max_Sharpe,
                                                                        'Security_1_weight': temp_best_weight_1,
                                                                        'Security_2_weight': temp_best_weight_2,
                                                                        'Security_3_weight': temp_best_weight_3},
                                                                       ignore_index=True)

Fixed_Risk = 0.10  # Nomenclature: 0.10 for 10% risk

# Making all the combinations for Fixed Risk
ticker_indices = np.arange(0, len(tickers)).tolist()
all_portfolio_of_3_combinations = combinations(ticker_indices, 3)
ETF_Combinations_Performance_Fixed_Risk = pd.DataFrame(columns=['Security_1', 'Security_2', 'Security_3',
                                                                'Fixed Risk', 'Sharpe', 'Security_1_weight',
                                                                'Security_2_weight', 'Security_3_weight'])

# Calculating Sharpe of these portfolio
for i9 in list(all_portfolio_of_3_combinations):
    temp_security_1 = tickers[i9[0]]
    temp_security_2 = tickers[i9[1]]
    temp_security_3 = tickers[i9[2]]

    ## Adj_Close_price has only adjusted close price of all the tickers
    temp_Adj_Close_price = pd.concat([data_2019[temp_security_1]['Adj Close'],
                                      data_2019[temp_security_2]['Adj Close'],
                                      data_2019[temp_security_3]['Adj Close']], axis=1)
    temp_Adj_Close_price.columns = [temp_security_1, temp_security_2, temp_security_3]

    # Passing the three ETFs and trying different weight combinations
    sharpe_arr_temp, vol_arr_temp, ret_arr_temp, EF_weights_temp = EfficientFrontier(temp_Adj_Close_price)

    temp_Closest_Risk = min(vol_arr_temp, key=lambda x: abs(x - Fixed_Risk))
    temp_Closest_Risk_index = vol_arr_temp.tolist().index(temp_Closest_Risk)

    # Sharpe
    temp_Sharpe = sharpe_arr_temp[temp_Closest_Risk_index]

    # Optimal Weights
    temp_weights = EF_weights_temp[temp_Closest_Risk_index]
    temp_weight_1 = temp_best_weights[0]
    temp_weight_2 = temp_best_weights[1]
    temp_weight_3 = temp_best_weights[2]

    print("At {}% Risk, Sharp Ratio of portfolio of {}({}%), {}({}%), {}({}%): {}.".format(round(100 * Fixed_Risk, 2),
                                                                                           temp_security_1,
                                                                                           round(
                                                                                               100 * temp_best_weight_1,
                                                                                               2),
                                                                                           temp_security_2,
                                                                                           round(
                                                                                               100 * temp_best_weight_2,
                                                                                               2),
                                                                                           temp_security_3,
                                                                                           round(
                                                                                               100 * temp_best_weight_3,
                                                                                               2),
                                                                                           round(temp_Sharpe, 2)))

    # Appending analysis data to 'ETF_Combinations_Performance_Fixed_Risk'
    ETF_Combinations_Performance_Fixed_Risk = ETF_Combinations_Performance_Fixed_Risk.append(
        {'Security_1': temp_security_1,
         'Security_2': temp_security_2,
         'Security_3': temp_security_3,
         'Fixed Risk': Fixed_Risk,
         'Sharpe': temp_Sharpe,
         'Security_1_weight': temp_best_weight_1,
         'Security_2_weight': temp_best_weight_2,
         'Security_3_weight': temp_best_weight_3},
        ignore_index=True)

print(ETF_Combinations_Performance.sort_values(by='Max_Sharpe',ascending=False))

start_time_2020 = datetime.datetime(2019, 12, 31)
end_time_2020 = datetime.datetime(2020, 12, 31)

data_2020 = {}

for i10,ticker in enumerate(tickers):
    data_2020[tickers[i10]] = web.DataReader(ticker,'yahoo', start_time_2020, end_time_2020).reset_index()

# Adding returns to the data; visualizing
for i11 in range(len(data)):
    temp_ticker = tickers[i11]
    temp_data = data_2020[temp_ticker]
    temp_data["Daily Returns"] = temp_data["Adj Close"].pct_change(1) # 1 for 1 day lookback
    temp_data = temp_data.dropna()
    print("\n" + "Head snapshot of " + tickers[i7])
    print(temp_data.head(3))
    data_2020[temp_ticker] = temp_data

# Making all the combinations
ticker_indices = np.arange(0, len(tickers)).tolist()
all_portfolio_of_3_combinations_2020 = combinations(ticker_indices, 3)
ETF_Combinations_Performance_2020 = pd.DataFrame(columns=['Security_1', 'Security_2', 'Security_3',
                                                          'Max_Sharpe', 'Security_1_weight',
                                                          'Security_2_weight', 'Security_3_weight'])
# Calculating Sharpe of these portfolio
for i11 in list(all_portfolio_of_3_combinations_2020):
    temp_security_1 = tickers[i11[0]]
    temp_security_2 = tickers[i11[1]]
    temp_security_3 = tickers[i11[2]]

    ## Adj_Close_price has only adjusted close price of all the tickers
    temp_Adj_Close_price = pd.concat([data_2020[temp_security_1]['Adj Close'],
                                      data_2020[temp_security_2]['Adj Close'],
                                      data_2020[temp_security_3]['Adj Close']], axis=1)
    temp_Adj_Close_price.columns = [temp_security_1, temp_security_2, temp_security_3]

    # Passing the three ETFs and trying different weight combinations
    sharpe_arr_temp, vol_arr_temp, ret_arr_temp, EF_weights_temp = EfficientFrontier(temp_Adj_Close_price)

    # Max Sharpe
    temp_Max_Sharpe = sharpe_arr_temp.max()

    # Optimal Weights
    temp_best_weights = EF_weights_temp[sharpe_arr_temp.argmax()]
    temp_best_weight_1 = temp_best_weights[0]
    temp_best_weight_2 = temp_best_weights[1]
    temp_best_weight_3 = temp_best_weights[2]

    print("Max Sharp Ratio of portfolio of {}({}%), {}({}%), {}({}%): {}.".format(temp_security_1,
                                                                                  round(100 * temp_best_weight_1, 2),
                                                                                  temp_security_2,
                                                                                  round(100 * temp_best_weight_2, 2),
                                                                                  temp_security_3,
                                                                                  round(100 * temp_best_weight_3, 2),
                                                                                  round(temp_Max_Sharpe, 2)))

    # Appending analysis data to 'ETF_Combinations_Performance'
    ETF_Combinations_Performance_2020 = ETF_Combinations_Performance_2020.append({'Security_1': temp_security_1,
                                                                                  'Security_2': temp_security_2,
                                                                                  'Security_3': temp_security_3,
                                                                                  'Max_Sharpe': temp_Max_Sharpe,
                                                                                  'Security_1_weight': temp_best_weight_1,
                                                                                  'Security_2_weight': temp_best_weight_2,
                                                                                  'Security_3_weight': temp_best_weight_3},
                                                                                 ignore_index=True)

print(ETF_Combinations_Performance_2020.sort_values(by='Max_Sharpe',ascending=False))