# coding: utf-8

# # An Intra-Day trading Strategy using Support Vector Machines

# - It is widely regarded that intraday trading is the most profitable form of trading on equity markets. 
# - For this assignment, we will put to the test and intraday trading model using XG-Boost.
# - We are going to pose this problem as a classification problem. The idea is to just determine using data with the present day trading signals and analysis [x(t)] , whether we can make the model classify that the stock price goes either up or down the next day [x(t+1)].
# - The code and Idea will be explained as we go deeper and further into the analysis.

# ## Asset class
# - For this, we will be considering the top performing stocks on the Bombay Stock Exchange of India. For simplicity, we will only consider 1 stock right, now, namely Reliance Industries, due to computation power restrictions, but this program can easily be extended to many more stocks of the users choice.


# Import useful Libraries
from nsepy import get_history as gh
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stats
import math
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Get the data required using nsepy
stk1 = gh(symbol='RELIANCE', start=date(2019, 11, 1), end=date(2020, 5, 7))

# We will save the dataframe so that we can keep calling it over and over again
stk1.to_pickle('ril.pkl')
ril = pd.read_pickle('ril.pkl')

# - We will now pose this problem as a classification problem.
# - The idea is that if the adjusted close price the next day is higher than the present day, we buy the stock. This will be indicated as 1. Otherwise, we sell the stock. This will be indicated as 0.
# rilf = pd.DataFrame(index = ril.index)
# rilf['price'] = ril['Last']
ril['response'] = ril['Last'].diff()
ril['class'] = np.where(ril['response'] > 0.0, 1, 0)
ril['class_final'] = ril['class'].shift(-1)  # shift the classes to align the next day with present day
ril = ril.iloc[:len(ril) - 1]
ril['class_final'] = ril.class_final.astype(int)
del ril['class']
del ril['response']

# ## Creating trading signals based on technical Indicators

# - We will now add indicators as features to the dataset.
# - We will use 7 indicators right now but the program can be extended for more indicators and candlestick patterns as well.

# ### 20 day siple moving average

# Simple moving average, is a basic technical analysis
# indicator. The simple moving average, as you may have guessed from its name, is
# computed by adding up the price of an instrument over a certain period of time divided by
# the number of time periods.

time_period = 20
history = []  # track history of prices
sma_values = []  # track simple moving average values
for close_price in ril['Close'].values:
    history.append(close_price)
    if len(history) > time_period:
        del (history[0])
    sma_values.append(stats.mean(history))

ril = ril.assign(twentySMA=pd.Series(sma_values, index=ril.index))

close_price = ril['Close']
sma = ril['twentySMA']

fig = plt.figure(figsize=(14, 14), dpi=80)
ax1 = fig.add_subplot(111, ylabel='RIL price in Rupees')
close_price.plot(ax=ax1, color='g', lw=2., legend=True)
sma.plot(ax=ax1, color='r', lw=2., legend=True)
plt.show()

# ### Exponential Moving Average
# The EMA is similar to the simple moving average, but, instead of weighing all prices in the
# history equally, it places more weight on the most recent price observation and less weight
# on the older price observations

num_periods = 20
k = 2 / (num_periods + 1)  # smoothing constant
ema_p = 0
ema_values = []  # hold computed EMA values

for close_price in ril['Close'].values:
    if (ema_p == 0):
        ema_p = close_price
    else:
        ema_p = (close_price - ema_p) * k + ema_p
    ema_values.append(ema_p)

ril = ril.assign(twentyEMA=pd.Series(ema_values, index=ril.index))

close_price = ril['Close']
ema = ril['twentyEMA']

fig = plt.figure(figsize=(14, 14), dpi=80)
ax1 = fig.add_subplot(111, ylabel='RIL price in Rupees')
close_price.plot(ax=ax1, color='g', lw=2., legend=True)
ema.plot(ax=ax1, color='r', lw=2., legend=True)
plt.show()

# ### Absolute Price Oscillator
# he absolute price oscillator is computed by finding the difference between a fast
# exponential moving average and a slow exponential moving average. Intuitively, it is
# trying to measure how far the more reactive EMA ( Fast) is deviating from the more
# stable EMA ( Slow). A large difference is usually interpreted as one of two things:
# instrument prices are starting to trend or break out, or instrument prices are far away from
# their equilibrium prices, in other words, overbought or oversold

num_periods_fast = 10
K_fast = 2 / (num_periods_fast + 1)
ema_fast = 0
num_periods_slow = 40
K_slow = 2 / (num_periods_slow + 1)
ema_slow = 0

ema_fast_values = []
ema_slow_values = []
apo_values = []

for close_price in ril['Close'].values:
    if ema_fast == 0:
        ema_fast = close_price
        ema_slow = close_price
    else:
        ema_fast = (close_price - ema_fast) * K_fast + ema_fast
        ema_slow = (close_price - ema_slow) * K_slow + ema_slow

    ema_fast_values.append(ema_fast)
    ema_slow_values.append(ema_slow)
    apo_values.append(ema_fast - ema_slow)

ril = ril.assign(emafast=pd.Series(ema_fast_values, index=ril.index))
ril = ril.assign(emaslow=pd.Series(ema_slow_values, index=ril.index))
ril = ril.assign(apo=pd.Series(apo_values, index=ril.index))

close_price = ril['Close']
ema_fast = ril['emafast']
ema_slow = ril['emaslow']
apo_v = ril['apo']

fig = plt.figure(figsize=(14, 14), dpi=80)
ax1 = fig.add_subplot(211, ylabel="Ril price in rupees")
close_price.plot(ax=ax1, color='g', lw=2., legend=True)
ema_fast.plot(ax=ax1, color='r', lw=1.5, legend=True)
ema_slow.plot(ax=ax1, color='b', lw=1.5, legend=True)
ax2 = fig.add_subplot(212, ylabel="Absolute Price Oscillator")
apo_v.plot(ax=ax2, color='y', lw=2.5, legend=True)

# ### Moving Average Convergence Divergence (MACD)
# The moving average convergence divergence was created by Gerald Appel. It is similar in
# spirit to an absolute price oscillator in that it establishes the difference between a fast
# exponential moving average and a slow exponential moving average. However, in the case
# of MACD, we apply a smoothing exponential moving average to the MACD value itself in
# order to get the final signal output from the MACD indicator. O

num_periods_fast = 10
K_fast = 2 / (num_periods_fast + 1)
ema_fast = 0
num_periods_slow = 40
K_slow = 2 / (num_periods_slow + 1)
ema_slow = 0

num_periods_macd = 20
k_macd = 2 / (num_periods_macd + 1)
ema_macd = 0

ema_fast_values = []
ema_slow_values = []
macd_values = []
macd_signal_values = []

macd_histogram_values = []

for close_price in ril['Close'].values:
    if (ema_fast == 0):
        ema_fast = close_price
        ema_slow = close_price
    else:
        ema_fast = (close_price - ema_fast) * K_fast + ema_fast
        ema_slow = (close_price - ema_slow) * K_slow + ema_slow

    ema_fast_values.append(ema_fast)
    ema_slow_values.append(ema_slow)
    macd = ema_fast - ema_slow

    if ema_macd == 0:
        ema_macd = macd
    else:
        ema_macd = (macd - ema_macd) * K_slow + ema_macd

    macd_values.append(macd)
    macd_signal_values.append(ema_macd)
    macd_histogram_values.append(macd - ema_macd)

ril = ril.assign(macd=pd.Series(macd_signal_values, index=ril.index))
ril = ril.assign(macd_hist=pd.Series(macd_histogram_values, index=ril.index))

close_price = ril['Close']
macd = ril['macd']
macd_hist = ril['macd_hist']

fig = plt.figure(figsize=(14, 14), dpi=80)
ax1 = fig.add_subplot(311, ylabel="Ril price in rupees")
close_price.plot(ax=ax1, color='g', lw=2., legend=True)
macd.plot(ax=ax1, color='r', lw=1.5, legend=True)
macd_hist.plot(ax=ax1, color='b', lw=1.5, legend=True)
ax2 = fig.add_subplot(312, ylabel="Moving Average Convergence Divergence")
macd.plot(ax=ax2, color='y', lw=2.5, legend=True)
ax3 = fig.add_subplot(313, ylabel="MACD Histogram")
macd_hist.plot(ax=ax3, color='r', kind='bar', legend=True)
plt.show()

# ### Bollinger Bands
# Bollinger bands is a well-known technical analysis indicator developed by John Bollinger. It
# computes a moving average of the prices (you can use the simple moving average or the
# exponential moving average or any other variant). In addition, it computes the standard
# deviation of the prices in the lookback period by treating the moving average as the mean
# price. It then creates an upper band that is a moving average, plus some multiple of
# standard price deviations, and a lower band that is a moving average minus multiple
# standard price deviations. 

time_period = 20
stdev_factor = 2  # beta in bollinger band equation

history = []
sma_values = []
upper_band = []
lower_band = []

for close_price in ril['Close'].values:
    history.append(close_price)
    if len(history) > time_period:
        del (history[0])
    sma = stats.mean(history)
    sma_values.append(sma)

    variance = 0

    for hist_price in history:
        variance = variance + ((hist_price - sma) ** 2)

    stdev = math.sqrt(variance / len(history))

    upper_band.append(sma + stdev_factor * stdev)
    lower_band.append(sma - stdev_factor * stdev)

ril = ril.assign(upper_band=pd.Series(upper_band, index=ril.index))
ril = ril.assign(lower_band=pd.Series(lower_band, index=ril.index))

close = ril['Close']
sma = ril['twentySMA']
upb = ril['upper_band']
lpb = ril['lower_band']

fig = plt.figure(figsize=(14, 14), dpi=80)
ax1 = fig.add_subplot(111, ylabel='Price in rupees')
close.plot(ax=ax1, color='g', lw=2.0, legend=True)
sma.plot(ax=ax1, color='b', lw=2.0, legend=True)
upb.plot(ax=ax1, color='r', lw=2.0, legend=True)
lpb.plot(ax=ax1, color='y', lw=2.0, legend=True)
plt.show()

# ### Relative Strength Indicator
# The relative strength indicator was developed by J Welles Wilder. It comprises a lookback
# period, which it uses to compute the magnitude of the average of gains/price increases over
# that period, as well as the magnitude of the averages of losses/price decreases over that
# period. Then, it computes the RSI value that normalizes the signal value to stay between 0
# and 100, and attempts to capture if there have been many more gains relative to the losses,
# or if there have been many more losses relative to the gains

time_period = 20
gain_history = []
loss_history = []
avg_gain_values = []
avg_loss_values = []
rsi_values = []
last_price = 0

for close_price in ril['Close'].values:
    if last_price == 0:
        last_price = close_price
    gain_history.append(max(0, close_price - last_price))
    loss_history.append(max(0, last_price - close_price))
    last_price = close_price

    if len(gain_history) > time_period:
        del (gain_history[0])
        del (loss_history[0])

    avg_gain = stats.mean(gain_history)
    avg_loss = stats.mean(loss_history)

    avg_gain_values.append(avg_gain)
    avg_loss_values.append(avg_loss)

    rs = 0

    if avg_loss > 0:
        rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values.append(rsi)

ril = ril.assign(rsi=pd.Series(rsi_values, index=ril.index))
rsi_val = ril['rsi']

fig = plt.figure(figsize=(14, 14), dpi=80)
ax1 = fig.add_subplot(111, ylabel='Price in rupees')
close.plot(ax=ax1, color='g', lw=2.0, legend=True)
rsi_val.plot(ax=ax1, color='b', lw=2.0, legend=True)
plt.show()

# ### Momentum

time_period = 20

history = []
mom_values = []

for close_price in ril['Close'].values:
    history.append(close_price)
    if len(history) > time_period:
        del (history[0])

    mom = close_price - history[0]
    mom_values.append(mom)

ril = ril.assign(mom=pd.Series(mom_values, index=ril.index))

mom = ril['mom']

fig = plt.figure(figsize=(14, 14), dpi=80)
ax1 = fig.add_subplot(111, ylabel='Price in rupees')
close.plot(ax=ax1, color='g', lw=2.0, legend=True)
mom.plot(ax=ax1, color='b', lw=2.0, legend=True)
plt.show()

# ## Getting the data ready

# Deleting the non essential columns - keeping only open,high, low, close and volume
del ril['Symbol']
del ril['Series']
del ril['Prev Close']
del ril['Last']
del ril['VWAP']
del ril['Turnover']
del ril['Deliverable Volume']
del ril['%Deliverble']
del ril['Trades']

Y = ril['class_final']
ril.drop(['class_final'], axis=1, inplace=True)

X = ril

# Splitting data into train, cross validation and test

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
# X_train, X_cv, Y_train, Y_cv = train_test_split(X_train, Y_train, test_size=0.3)

# ## Running SVM

logs = []
# we will perform hyperparameter tuning for the parameter alpha
parameters = {'alpha': [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4]}
for i in parameters['alpha']:
    b = math.log(i)
    logs.append(b)
# SGD classifier with Hinge loss is SVM
neigh = SGDClassifier(loss='hinge', penalty='l2', class_weight='balanced')

clf = GridSearchCV(neigh, parameters, cv=5, scoring='roc_auc')
clf.fit(X_train, Y_train)

train_auc = clf.cv_results_['mean_train_score']
train_auc_std = clf.cv_results_['std_train_score']
cv_auc = clf.cv_results_['mean_test_score']
cv_auc_std = clf.cv_results_['std_test_score']

plt.plot(logs, train_auc, label='Train AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(logs, train_auc - train_auc_std, train_auc + train_auc_std, alpha=0.2, color='darkblue')

plt.plot(logs, cv_auc, label='CV AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(logs, cv_auc - cv_auc_std, cv_auc + cv_auc_std, alpha=0.2, color='darkorange')

plt.scatter(logs, train_auc, label='Train AUC points')
plt.scatter(logs, cv_auc, label='CV AUC points')

plt.legend()
plt.xlabel("K: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.grid()
plt.show()

# Plotting test results AUC score

neigh = SGDClassifier(loss='hinge', penalty='l2', alpha=10 ** -1, class_weight='balanced')
neigh.fit(X_train, Y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs

y_train_pred = neigh.decision_function(X_train)

y_test_pred = neigh.decision_function(X_test)

train_fpr, train_tpr, tr_thresholds = roc_curve(Y_train, y_train_pred)
test_fpr, test_tpr, te_thresholds = roc_curve(Y_test, y_test_pred)

plt.plot(train_fpr, train_tpr, label="Train AUC =" + str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="Test AUC =" + str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("True Positive Rate(TPR)")
plt.ylabel("False Positive Rate(FPR)")
plt.title("AUC")
plt.grid()
plt.show()

# ## Observations:
# - We can improve upon this by searching all the hyper - parameters of SVM which could increase the computation cost.
# - We could also add more technical indicators and we could also add candlestick chart patterns
# - We can test more models like Logistic Regression, SVM, Random forests etc
