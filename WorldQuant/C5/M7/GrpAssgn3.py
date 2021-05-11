#Importing all libraries

import numpy as np
from scipy.stats import norm
from scipy.stats import ncx2
import scipy.optimize as opt
import matplotlib.pyplot as plt
import random
import math
#for producing charts inline
#matplotlib inline

#market information

risk_free = 0.08

#share specific information
S0 = 100
vol = 0.3

#option specific information
strike = 100
barrier = 150
T = 12
current_time = 0

#firm specific information
V0 = 200
firm_vol = 0.3
debt = 175
recovery_rate = 0.25
corr = 0.2

#model specific information
gamma = 0.75

#Bond prices
zcb_price = [99.38, 98.76, 98.15, 97.54, 96.94, 96.34, 95.74, 95.16, 94.57, 93.99, 93.42, 92.85]

#initializing

np.random.seed(0)

shareprice = [None]*12
local_vol_share = [None]*12
shareprice_no_disc = [None]*12

firmvalue = [None]*12
local_vol_firm = [None]*12
firmvalue_no_disc = [None]*12

callprice = [None]*12
cva_estimates = [None]*12
cva_adj_estimates = [None]*12

callprice_std = [None]*12
cva_estimates_std = [None]*12
cva_adj_estimates_std = [None]*12

cont_rate = [None]*12
libor_rate = [None]*12
disc_rate = [None]*12

#continuously compund rate calculation
for i in range (0,12) :
    cont_rate[i] = np.log(100/zcb_price[i])
    libor_rate[i] = np.exp(cont_rate[i])-1
print(libor_rate)

#Defining function

#Option specific function

#Define terminal share price function
def term_price(S0, Rf, vol, z, T) :
    tsp = S0*np.exp((Rf-(vol**2/2))*T+vol*np.sqrt(T)*z)
    return tsp

#Define discounted payoff function
def discounted_payoff(St, K, Rf, T, L) :
    disc_pf = np.exp(-Rf*T)*np.maximum(St-K,0)*(St<=L)
    return disc_pf

#Share specific funtions -

#Define function for terminal shareprice
def term_price_share(S_prev, Rf, s_vol, z, T) :
    tsp = S_prev*np.exp((Rf-((s_vol*(S_prev**(gamma-1)))**2)/2)*(T)+s_vol*(S_prev**(gamma-1))*np.sqrt(T)*z)
    return tsp

#Define function for local vol for share
def localvol_share(S_prev, s_vol, gamma) :
    local_volatility = s_vol*(S_prev**(gamma-1))
    return local_volatility

#Firm specific functions -

#Define function for terminal firm value
def term_price_firm(F_prev, Rf, f_vol, z, T) :
    tsp = F_prev*np.exp((Rf-((f_vol*(F_prev**(gamma-1)))**2)/2)*(T)+f_vol*(F_prev**(gamma-1))*np.sqrt(T)*z)
    return tsp

#Define function for local vol for firm
def localvol_firm(F_prev, f_vol, gamma) :
    local_volatility = f_vol*(F_prev**(gamma-1))
    return local_volatility


term_val_share = S0
term_val_firm = V0
shareprice_no_disc[0] = S0
firmvalue_no_disc[0] = V0

for current_time in range(0, T):
    # i=1
    # for i in range (1,51) :
    norm_array = norm.rvs(size=np.array([2, 100000]))
    corr_array = np.array([[1, corr], [corr, 1]])
    corr_norm_matrix = np.matmul(np.linalg.cholesky(corr_array), norm_array)

    disc_rate[current_time] = 1
    for j in range(0, current_time + 1):
        disc_rate[current_time] = disc_rate[current_time] * libor_rate[j]

    ## function calls

    # share specific function calls
    lvol_share = localvol_share(term_val_share, vol, gamma)
    term_val_stock = term_price_share(term_val_share, risk_free, vol, corr_norm_matrix, (T - current_time) / 12)

    # firm specific funtion calls
    lvol_firm = localvol_firm(term_val_firm, firm_vol, gamma)
    term_val_f = term_price_firm(term_val_firm, risk_free, firm_vol, corr_norm_matrix, (T - current_time) / 12)

    # option specific call
    disc_payoff = discounted_payoff(term_val_stock, strike, risk_free, (T - current_time) / 12, barrier)

    ## price estimates

    # share price estimates
    sp_estimates = np.exp(-risk_free * (T - current_time) / 12) * np.mean(term_val_stock)
    # term_val_share=sp_estimates
    shareprice[current_time] = sp_estimates
    term_val_share = np.mean(term_val_stock)
    shareprice_no_disc[current_time] = np.mean(term_val_stock)

    # firm value estimate
    f_estimates = np.exp(-risk_free * (T - current_time) / 12) * np.mean(term_val_f)
    # term_val_firm = f_estimates
    term_val_firm = np.mean(term_val_f)
    firmvalue[current_time] = f_estimates
    firmvalue_no_disc[current_time] = np.mean(term_val_f)

    # call estimates
    call_estimates = np.mean(disc_payoff)
    # print("Call Estimate - ", call_estimates[0])
    callprice[current_time] = call_estimates
    callprice_std[current_time] = np.std(disc_payoff) / (np.sqrt(100000))

    # CVA estimates
    # loss_amt = np.exp(-risk_free*(T-current_time)*(1/12))*(1-recovery_rate)*(term_val_firm<debt)*disc_payoff
    loss_amt = (1 - recovery_rate) * (term_val_f < debt) * disc_payoff
    cva_estimates[current_time] = np.mean(loss_amt)
    # print("Loss Amount - ", cva_estimates[0])
    # cva_std[current_time] = np.std(loss_amt)/(np.sqrt(100000))

    # CVA adjusted price
    cva_adj_estimates[current_time] = np.mean(disc_payoff - loss_amt)
    # cva_adj_std[current_time] = np.std(disc_payoff - loss_amt)/(np.sqrt(100000))

    ## local volatility estimates
    local_vol_share[current_time] = lvol_share
    local_vol_firm[current_time] = lvol_firm

#mean values - only for testing
mean_barrier = np.mean(callprice)
print('Average Barrier Option Price (default free): ',mean_barrier)
mean_cva = np.mean(cva_estimates)
print('Avergae CVA : ',mean_cva)
mean_adj_price = np.mean(cva_adj_estimates)
print('Average CVA adjusted price : ',mean_adj_price)

#Plotting the Stock price - varying time from terminal time
plt.title("Simulated share price")
plt.xlabel("Months")
plt.ylabel("Simulated share Price")
#plt.plot(sp_estimates, 'c')
#for j in range (0,12) :
plt.plot(shareprice)
plt.plot([np.mean(shareprice)]*12)
plt.show()

#Plotting the Stock price - varying time from terminal time
plt.title("Simulated firm value")
plt.xlabel("Months")
plt.ylabel("Simulated firm value")
#plt.plot(sp_estimates, 'c')
#for j in range (0,12) :
plt.plot(firmvalue)
plt.plot([np.mean(firmvalue)]*12)
plt.show()

#Plotting the local vol
plt.title("Simulated Local Volatility - shareprice")
plt.xlabel("Sample Size(in '000)")
plt.ylabel("Simulated local vol")
#for j in range (0,12) :
plt.plot(local_vol_share)
plt.show()

#Plotting the local vol
plt.title("Simulated Local Volatility - firm value")
plt.xlabel("Months")
plt.ylabel("Simulated local vol")
#for j in range (0,12) :
plt.plot(local_vol_firm)
plt.show()

#Plotting the Barrier Value ()
plt.title("Barrier Option Value - Default Free")
plt.xlabel("Months")
plt.ylabel("Simulated Call Price")
#for j in range (0,12) :
plt.plot(callprice,'g', label = "Call Price")
plt.plot([np.mean(callprice)]*12,'.', label = "Mean Call Price")
plt.legend(loc = "upper left")
plt.show()

plt.title("Pricing dynamics")
plt.xlabel("Months")
plt.ylabel("Price")
plt.plot(callprice,'g', label = "Call Price")
plt.plot(shareprice,'r', label ="Share Price")
plt.plot([strike]*12, label = "Strike Price")
plt.plot([barrier]*12, label = "Barrier")
plt.legend(loc= "upper left")
plt.show()