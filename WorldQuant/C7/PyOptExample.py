from PyPortfolioOpt.pypfopt.expected_returns import mean_historical_return
from PyPortfolioOpt.pypfopt.risk_models import CovarianceShrinkage
from PyPortfolioOpt.pypfopt import plotting
from PyPortfolioOpt.pypfopt import CLA
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from itertools import combinations

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

# Reading in the data; preparing expected returns and a risk model
df = pd.read_csv("C:/Users/sunil/PycharmProjects/Learning/PyPortfolioOpt/tests/resources/stock_prices.csv", parse_dates=True, index_col="date")
df = XLB_XLV_Adj_Close_price
returns = df.pct_change().dropna()

mu = mean_historical_return(df)
S = CovarianceShrinkage(df).ledoit_wolf()

cla = CLA(mu, S,weight_bounds=(0, 1))
print(cla.max_sharpe())
cla.portfolio_performance(verbose=True)
plotting.plot_efficient_frontier(cla)


