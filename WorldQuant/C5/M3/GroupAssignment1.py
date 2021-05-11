import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import random


# Defining the relevant functions

def geometric_brownian_motion(N=200, T=1, S_0=100, mu=0.08, sigma=0.3):
    """N is the number of steps, T is the time of expiry,
    S_0 is the beginning value, sigma is standard deviation"""
    deltat = float(T) / N  # compute the size of the steps
    W = np.cumsum(np.random.standard_normal(size=N)) * np.sqrt(deltat)

    # generate the brownian motion
    t = np.linspace(0, T, N)
    X = (mu - 0.5 * sigma ** 2) * t + sigma * W
    S = S_0 * np.exp(X)
    # geometric brownian motion
    return S

def european_call_payoff(S_T, K=100):
    """S_T is the price of the underlying at expiry.
        K is the strike price"""
    return np.maximum(0, S_T - K)  # payoff for call option

def price_up_and_out_european_barrier_call(barrier, S_0, N=500, K=100):
    """barrier is the barrier level,
     S_0 is the starting value of the underlying,
     N is how many paths we will generate to take the mean.
     K is the strike price."""
    paths = [geometric_brownian_motion(S_0=S_0) for i in range(N)]
    prices = []
    for path in paths:
        if np.max(path) > barrier:  # knocked out
            prices.append(0)
        else:
            prices.append(european_call_payoff(path[-1], K))
    return np.mean(prices)

# Simulating the price depending on the current spot
# barrier is at 150
# strike price is at 100

spot = np.linspace(90, 170, 2)
prices = [price_up_and_out_european_barrier_call(150, s) for s in spot]
plt.plot(spot, prices, '--', linewidth=1)
plt.show()

#initializing
np.random.seed(0)
cva_estimates =[None]*50
cva_std = [None]*50

#market information
risk_free = 0.08
#share specific information
S0 = 100
vol = 0.3
#option specific information
strike = 100
barrier = 150
T = 1
current_time = 0
#firm specific information
V0 = 150
firm_vol = 0.25
debt = 175
recovery_rate = 0.25
corr = 0.2

#Defining function

#Define terminal share price function
def term_price(S0, Rf, vol, z, T) :
    tsp = S0*np.exp((Rf-(vol**2/2))*T+vol*np.sqrt(T)*z)
    return tsp

#Define discounted payoff function
def discounted_payoff(St, K, Rf, T, L) :
    #if (St<=L):
        #disc_pf = np.exp(-Rf*T)*np.maximum(St-K,0)
    #else:
        #disc_pf = 0
    disc_pf = np.exp(-Rf*T)*np.maximum(St-K,0)*(St<=L)
    return disc_pf

#Simulating values for CVA
for i in range (1,51) :
    norm_array = norm.rvs(size = np.array([2,50000]))
    corr_array = np.array([[1,corr],[corr,1]])
    corr_norm_matrix = np.matmul(np.linalg.cholesky(corr_array),norm_array)
    #corr_norm_matrix = norm_array*corr
    term_val_stock = term_price(S0, risk_free, vol, corr_norm_matrix, (T-current_time))
    disc_payoff = discounted_payoff(term_val_stock, strike, risk_free, (T-current_time), barrier)
    term_val_firm = term_price(V0, risk_free, firm_vol, corr_norm_matrix, (T-current_time))
    loss_amt = np.exp(-risk_free*(T-current_time))*(1-recovery_rate)*(term_val_firm<debt)*disc_payoff
    cva_estimates[i-1] = np.mean(loss_amt)
    cva_std[i-1] = np.std(loss_amt)/(np.sqrt(50000))

#Plotting Charts
plt.xlabel("Sample Size(in '000)")
plt.ylabel("Simulated Call Price")
plt.plot(cva_estimates, '.')
plt.plot(cva_estimates+3*np.array(cva_std), 'r')
plt.plot(cva_estimates-3*np.array(cva_std), 'y')
plt.show()

# Code to calculate default probability

d1_default = (np.log(V0/debt)+(risk_free+firm_vol**2/2)*(T))/(firm_vol*np.sqrt(T))
d2_default = d1_default-firm_vol*np.sqrt(T)
default_prob = norm.cdf(-d2_default)
uncorr_cva = (1-recovery_rate)*default_prob
print("Monte Carlo estimates while incorporating counterparty risk, given by the default-free price less the CVA is:",uncorr_cva)

corr = np.linspace(-1,1,21)
plt.plot([corr],[uncorr_cva]*21)
plt.plot([corr],cva_estimates,'.')
plt.plot([corr],cva_estimates+3*np.array(cva_std),'r')
plt.plot([corr],cva_estimates-3*np.array(cva_std),'r')
plt.xlabel("Correlation")
plt.ylabel("cva")
plt.show()

# temp =
# for i in range (1,51) :
#     temp[i-1] = prices[i-1] - cva_estimates[i-1]
# a = 1