# Import Libraries
import os
import requests

import pandas as pd
import numpy as np

import holoviews as hv
import hvplot.pandas

# Import Plotting Backend
hv.extension('bokeh')

date_ranges = [[1970, 1979, 'dat'],
               [1960, 1969, 'dat'],
               [1950, 1959, 'dat'],
               [1940, 1949, 'dat'],
               [1930, 1939, 'dat'],
               [1920, 1929, 'prn'],
               [1900, 1919, 'dat'],
               [1888, 1899, 'dat']][::-1]


# Read and format the data
def load_data(start=1920, end=1929, extension='prn'):
    path = os.path.join("E:\\WQMsc\\Course8-CasesinFinance\\M1\\Case Studies in Risk Management Module 1 extra documents\\", "Data", f"Daily_Share_Volume_{start}-{end}.{extension}")
    if extension == 'prn':
        data = pd.read_csv(path, sep='   ', parse_dates=['Date'], engine='python').iloc[2:, 0:2]
        data.loc[:, "  Stock U.S Gov't"] = pd.to_numeric(data.loc[:, "  Stock U.S Gov't"], errors='coerce')
        data.Date = pd.to_datetime(data.Date, format='%Y%m%d', errors='coerce')
        data.columns = ['Date', 'Volume']
        return data
    else:
        data = pd.read_csv(path)
        data.iloc[:, 0] = data.iloc[:, 0].apply(lambda x: str(x).strip(' '))
        data = data.iloc[:, 0].str.split(' ', 1, expand=True)
        data.columns = ['Date', 'Volume']
        data.loc[:, "Volume"] = pd.to_numeric(data.loc[:, "Volume"], errors='coerce')
        data.Date = pd.to_datetime(data.Date, format='%Y%m%d', errors='coerce')
        return data

data = pd.concat([load_data(decade[0], decade[1], decade[2]) for decade in date_ranges], axis=0)

# Create plotting object
plot_data = hv.Dataset(data, kdims=['Date'], vdims=['Volume'])

# Create scatter plot

black_tuesday = pd.to_datetime('1929-10-29')

vline = hv.VLine(black_tuesday).options(color='#FF7E47')

m = hv.Scatter(plot_data).options(width=700, height=400).redim('NYSE Share Trading Volume').hist() * vline * \
    hv.Text(black_tuesday + pd.DateOffset(months=10), 4e7, "Black Tuesday", halign='left').options(color='#FF7E47')
print(m)

# Create plotting object
plot_data_zoom = hv.Dataset(data.loc[((data.Date >= pd.to_datetime("1920-01-01"))&(data.Date <= pd.to_datetime("1940-01-01"))),:], kdims=['Date'], vdims=['Volume'])

# Create scatter plot

black_tuesday = pd.to_datetime('1929-10-29')

vline = hv.VLine(black_tuesday).options(color='#FF7E47')

m = hv.Scatter(plot_data_zoom).options(width=700, height=400).redim('NYSE Share Trading Volume').hist() * vline * \
    hv.Text(black_tuesday + pd.DateOffset(months=10), 4e7, "Black Tuesday", halign='left').options(color='#FF7E47')
m

# %%opts Scatter [width = 400 height = 200]

data['Quarter'] = data.Date.dt.quarter


def second_order(days_window):
    data_imputed = data
    data_imputed.Volume = data_imputed.Volume.interpolate()

    return hv.Scatter(pd.concat([data_imputed.Date, data_imputed.Volume.rolling(days_window).mean()],
                                names=['Date', 'Volumne Trend'], axis=1)
                      .dropna()).redim(Volume='Mean Trend') + \
           hv.Scatter(pd.concat([data_imputed.Date, data_imputed.Volume.rolling(days_window).cov()],
                                names=['Date', 'Volumne Variance'], axis=1)
                      .dropna()).redim(Volume='Volume Variance').options(color='#FF7E47')


hv.DynamicMap(second_order, kdims=['days_window']).redim.range(days_window=(7, 1000))

# % % opts
# Bars[width = 400
# height = 300]
from statsmodels.tsa.stattools import acf, pacf


def auto_correlations(start_year, window_years):
    start_year = pd.to_datetime(f'{start_year}-01-01')
    window_years = pd.DateOffset(years=window_years)

    data_window = data
    data_window = data_window.loc[((data_window.Date >= start_year)
                                   & (data_window.Date <= (start_year + window_years))), :]

    return hv.Bars(acf(data_window.Volume.interpolate().dropna())) \
               .redim(y='Autocorrelation', x='Lags') + \
           hv.Bars(pacf(data_window.Volume.interpolate().dropna())) \
               .redim(y='Patial Autocorrelation', x='Lags').options(color='#FF7E47')


hv.DynamicMap(auto_correlations, kdims=['start_year', 'window_years']
              ).redim.range(start_year=(data.Date.min().year, data.Date.max().year), window_years=(1, 25))


# Too much Margin!

from functools import reduce
import operator

import os
import requests

import pandas as pd
import numpy as np

import holoviews as hv
import hvplot.pandas

np.random.seed(42)
hv.extension('bokeh')

def plot(mu, sigma, samples):
    return pd.Series(np.random.normal(mu,sigma, 1000)).cumsum(
    ).hvplot(title='Random Walks', label=f'{samples}')

def prod(mu, sigma, samples):
    return reduce(operator.mul,
                  list(map(lambda x: plot(mu,sigma, x),
                           range(1,samples+1))))

hv.DynamicMap(prod,kdims=['mu', 'sigma', 'samples']).redim.range(mu=(0,5), sigma=(1,10), samples=(2,10)).options(width=900, height=400)


class Accounts:
    def __init__(self, account_mu=20, account_sigma=5, account_numbers=100, mu=0, sigma=0.05, margin_mu=0.1,
                 momentum_mu=0.025, margin_sigma=0.001):
        # We initialize paramters for our simulations
        self.account_mu = account_mu
        self.account_sigma = account_sigma
        self.account_numbers = account_numbers

        self.margin_mu = margin_mu
        self.momentum_mu = momentum_mu
        self.margin_sigma = margin_sigma

        self.accounts = np.maximum(
            np.random.normal(loc=self.account_mu, scale=self.account_sigma, size=self.account_numbers), 0)
        self.call = self.accounts * np.random.uniform(0.5, 0.7, self.account_numbers)

        self.mu = mu
        self.sigma = sigma

        self.called_accounts_factor = 0

        self.momentum = 0
        self.history = [0, 0, 0, 0, 0]

        return None

    def price(self):
        # Calculate factors
        self.called_accounts_factor = ((self.accounts <= self.call).sum()) / self.account_numbers
        self.momentum = (self.history[4] - self.history.pop(0))

        # Update paramteres
        self.mu = self.mu - self.margin_mu * self.called_accounts_factor + self.momentum_mu * self.momentum
        self.sigma = self.sigma + self.margin_sigma * self.called_accounts_factor

        # Update accounts
        self.history.append(np.random.normal(loc=self.mu, scale=self.sigma))
        self.accounts = self.accounts * (
                    1 + self.history[4] * np.random.uniform(low=0.5, high=1.5, size=self.account_numbers))

        # Reset called accounts
        reset = (np.random.rand(self.account_numbers) >= 0.3) * (self.accounts <= self.call)
        self.accounts[reset] = np.random.normal(self.account_mu, self.account_sigma)
        self.call[reset] = self.accounts[reset] * np.random.uniform(0.2, 0.5, np.sum(reset))

        return [self.history[4], self.accounts.sum(), self.called_accounts_factor, self.momentum]


def simulation_plots(days=10000, runs=5):
    run = []

    for _ in range(runs):
        a = Accounts()
        prices = pd.DataFrame([a.price() for day in range(days)],
                              columns=['Market Returns', 'Value of Accounts Trading in the Market',
                                       'The effect of Margin Calls on Supply', 'Momentum Effect'])
        run.append(prices)

    plot = reduce(operator.mul, [i.iloc[:, 0].hvplot() for i in run]) + \
           reduce(operator.mul, [i.iloc[:, 1].hvplot() for i in run]) + \
           reduce(operator.mul, [i.iloc[:, 2].hvplot() for i in run]) + \
           reduce(operator.mul, [i.iloc[:, 3].hvplot() for i in run])

    return plot.cols(2)

# %%opts Curve [width=500 height=300]
hv.DynamicMap(simulation_plots, kdims=['days', 'runs']).redim.range(days=(100,500), runs=(5,15)).options(width=900, height=400)


def simulation_prices(days=100, runs=1000, axis=0):
    run = []

    for _ in range(runs):
        a = Accounts()
        prices = pd.DataFrame([a.price()[0] for day in range(days)],
                              columns=['return'])
        run.append(prices)

    output = pd.concat(run, axis=axis)
    output.columns = [f'Run {i + 1}' for i in range(output.shape[1])]

    return output

simulations = simulation_prices()

# %%opts Overlay [show_title=True] Distribution [height=500, width=1000]
hv.Distribution(np.random.normal(simulations.mean(),simulations.std(),100000), label='Normal') * hv.Distribution(simulations.iloc[:,0], label='Simulation').options(fill_alpha=0.0)