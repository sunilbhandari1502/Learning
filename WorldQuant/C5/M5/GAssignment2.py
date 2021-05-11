#importing libraries
import numpy as np
from scipy.stats import norm
from scipy.stats import ncx2
import matplotlib.pyplot as plt
import random
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

# Share specific information
S0 = 100
v0 = 0.06
kappa = 9
theta = 0.06
r = 0.03
sigma = 0.5
rho = -0.4

# Call opotion related information
K=105
T=0.5
k_log=np.log(K)

#Approximate informtion
t_max =30
N=100

#Characterisitc function code
a = sigma**2/2

def b(u):
    return kappa-rho*sigma*1j*u

def c(u):
    return -(u**2+1j*u)/2

def d(u):
    return np.sqrt(b(u)**2-4*a*c(u))

def xminus(u):
    return (b(u)-d(u))/(2*a)

def xplus(u):
    return (b(u)+d(u))/(2*a)

def g(u):
    return xminus(u)/xplus(u)

def C(u):
    val1 = T*xminus(u)-np.log((1-g(u)*np.exp(-T*d(u)))/(1-g(u)))/a
    return r*T*1j*u+ theta*kappa*val1

def D(u):
    val1 = 1-np.exp(-T*d(u))
    val2 = 1-g(u)*np.exp(-T*d(u))
    return (val1/val2)*xminus(u)

def log_char(u):
    return np.exp(C(u)+D(u)*v0 + 1j*u*np.log(S0))

def adj_char(u):
    return log_char(u-1j)/log_char(-1j)

delta_t= t_max/N
from_1_to_N = np.linspace(1,N,N)
t_n=(from_1_to_N-1/2)*delta_t

first_integral = sum((((np.exp(-1j*t_n*k_log)*adj_char(t_n)).imag)/t_n)*delta_t)
second_integral = sum((((np.exp(-1j*t_n*k_log)*log_char(t_n)).imag)/t_n)*delta_t)

# Price of call option using Fourier method

fourier_call_val = S0*(1/2+first_integral/np.pi)-np.exp(-r*T)*K*(1/2+second_integral/np.pi)

print(fourier_call_val)

print(first_integral)

#initializing
np.random.seed(0) #Seeding the random generator to 0
sp_estimates =[None]*50 #Creating an array for hosting range of values for Share Price
sp_std = [None]*50      #Creating an array for holding the standard deviations
call_estimates =[None]*50  #Creating an array for holding the call estimates
call_std = [None]*50   #Creating an array for holding the call standard deviations
shareprice = np.empty((12,50))
callprice = np.empty((12,50))
local_vol = np.empty((12,50))

#market information

risk_free = 0.08

#share specific information
S0 = 100
vol = 0.3

#option specific information
strike = 100
T = 12
current_time = 0
gamma = 0.75

#Defining function

#Define terminal share price functions
def term_price(S_prev, Rf, vol, z, T) :
    tsp = S_prev*np.exp((Rf-((vol*(S_prev**(gamma-1)))**2)/2)*(T)+vol*(S_prev**(gamma-1))*np.sqrt(T)*z)
    return tsp
#Defining the discounted payoff
def discounted_payoff(St, K, Rf, T) :
    disc_pf = np.exp(-Rf*T)*np.maximum(St-K,0)
    return disc_pf

#Definigin functions for local Volatility
def localvol(S_prev, vol, gamma) :
    local_volatility = vol*(S_prev**(gamma-1))
    return local_volatility


for current_time in range (0,T) :
    term_val = S0
    for i in range (1,51) :
        norm_array = norm.rvs(size = np.array([1,i*1000]))
        # function calls
        term_val_stock = term_price(term_val, risk_free, vol, norm_array, (T-current_time)/12)
        disc_payoff = discounted_payoff(term_val_stock, strike, risk_free, (T-current_time)/12)
        lvol = localvol(term_val, vol, gamma)
        #share price estimates
        sp_estimates[i-1] = np.exp(-risk_free*(T-current_time)/12)*np.mean(term_val_stock)
        term_val=sp_estimates[i-1]
        if [current_time == 11] :
            sp_std[i-1] = np.std(np.exp(-risk_free*(T-current_time)/12)*term_val_stock)/(np.sqrt(i*1000))
        shareprice[current_time,i-1]=sp_estimates[i-1]
        #call estimates
        call_estimates[i-1] = np.mean(disc_payoff)
        if [current_time == 11] :
            call_std[i-1] = np.std(disc_payoff)/(np.sqrt(i*1000))
        callprice[current_time,i-1]=call_estimates[i-1]
        #local volatility estimates
        local_vol[current_time,i-1]=lvol

#Plotting the Stock price - varying time from terminal time
plt.title("Simulated share price")
plt.xlabel("Sample Size(in '000)")
plt.ylabel("Simulated share Price")
#plt.plot(sp_estimates, 'c')
for j in range (0,12) :
    plt.plot(shareprice[j,:])
plt.show()

#Plotting the call price - varying time to maturity
plt.title("Simulated call price")
plt.xlabel("Sample Size(in '000)")
plt.ylabel("Simulated call Price")
#plt.plot(call_estimates, 'c')
for j in range (0,12) :
    plt.plot(callprice[j,:])
plt.show()

#Plotting the local vol
plt.title("Simulated Local Volatility")
plt.xlabel("Sample Size(in '000)")
plt.ylabel("Simulated local vol")
for j in range (0,12) :
    plt.plot(local_vol[j,:])
plt.show()

#Plotting the Stock price - current_time = 0
plt.title("Simulated share price")
plt.xlabel("Sample Size(in '000)")
plt.ylabel("Simulated share Price")
plt.plot(shareprice[11,:],'.')
plt.plot(shareprice[11,:]+3*np.array(sp_std), 'r')
plt.plot(shareprice[11,:]-3*np.array(sp_std), 'y')
plt.show()

#Plotting the Call price - current_time = 0
plt.title("Simulated call price")
plt.xlabel("Sample Size(in '000)")
plt.ylabel("Simulated call Price")
plt.plot(callprice[11,:],'.')
plt.plot(callprice[11,:]+3*np.array(call_std), 'r')
plt.plot(callprice[11,:]-3*np.array(call_std), 'y')
plt.show()

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#share specific info
r = 0.08
S0 = 100
sigma = 0.3

#Call option specific
K = 100
T = 1
K_log = np.log(K)

#Volatility parts
v0 = 0.06
kappa = 9
theta = 0.06
rho = -0.4

#Approximation info
t_max = 30
N = 100

# 3. Augmented with Monte carlo calculation of Call Price and Std.

import numpy as np
from scipy.stats import uniform
from scipy.stats import norm
import matplotlib.pyplot as plt
import random

# Simulation param
gamma = 0.75
sigma = 0.3
# Setting seed
np.random.seed(0)
dT = 1 / 12


def share_path(S_0, risk_free_rate, sigma, Z, dT, gamma):
    n = len(Z)
    sharepath = [None] * n
    s_t = S0
    for i in range(n):
        z = Z[i]
        sigma_t = sigma * ((s_t) ** (gamma - 1))
        s_t = s_t * np.exp((risk_free_rate - sigma_t ** 2 / 2) * dT
                           + sigma_t * np.sqrt(dT) * z)
        sharepath[i] = s_t
    return sharepath


# function to calculate discounted call payoff = price of option
def discounted_call_payoff(S_T, K, r, T):
    return np.exp(-r * T) * np.maximum(S_T - K, 0)


# this list stores all price paths
price_paths_sims_all = [None] * 50
# These list stores call_estimates means for price and std for each sample size
mcall_est = [None] * 50
mcall_std = [None] * 50
for j in range(1, 51):
    price_paths_sims = np.zeros((j * 1000, 12))
    mcall_val = np.zeros(j * 1000)
    Z_all = norm.rvs(size=[j * 1000, 12])

    for k in range(j * 1000):
        # picks the line of Z random variables to use with this sim
        Z = Z_all[k]
        # calculate the price path according to CEV.
        price_path = share_path(S0, r, sigma, Z, dT, gamma)
        price_paths_sims[k] = price_path
        S_T = price_path[-1]
        # calculate the call price
        mcall_val[k] = discounted_call_payoff(S_T, K, r, T)

    mcall_est[j - 1] = np.mean(mcall_val)
    mcall_std[j - 1] = np.std(mcall_val) / np.sqrt(j * 1000)
    price_paths_sims_all[j - 1] = price_paths_sims
    print("Completed Sim for ", j, "Thousand samples")
print("Completed all Sims")
print(price_paths_sims_all[j - 1])