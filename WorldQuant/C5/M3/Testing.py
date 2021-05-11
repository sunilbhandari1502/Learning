import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import random

#Genral share information
S0 = np.array([[100],[95],[50]])
sigma = np.array([[0.15],[0.2],[0.3]])
cor_mat = np.array([[1,0.2,0.4],[0.2,1,0.8],[0.4,0.8,1]])
L = np.linalg.cholesky(cor_mat)
r = 0.1
T = 1

#applying Monte Carlo estimation of VaR
np.random.seed(0)
t_simulation = 10000
alpha = 0.05
#Current portfolio Value
portval_current = np.sum(S0)

#function for terminal stock value
def terminal_shareprice(S_0,rfr,sigma,Z,T):
    return S_0*np.exp((rfr-sigma**2/2)*T+sigma*np.sqrt(T)*Z)
def terminal_value(S_0,rfr,sigma,Z,T):
    return S_0*np.exp((rfr-sigma**2/2)*T+sigma*np.sqrt(T)*Z)

#Creating 10000 simulations for future portfolio values
Z = np.matmul(L,norm.rvs(size=[3,t_simulation]))
portval_future = np.sum(terminal_shareprice(S0,r,sigma,Z,T),axis = 0)

#Calculating portfoilio returns
portreturn = (portval_future-portval_current)/portval_current

#Sorting Returns
portreturn = np.sort(portreturn)

#Determining Monte Carlo VaR
mVaR_estimate = -portreturn[int(np.floor(alpha*t_simulation))-1]


#CVA in Python example

#Market information
rfr = 0.1
#Share specific information
S_0 = 100
sigma = 0.3
#Call Option specific information
strike = 110
T = 1
#Firm specifc information
V_0 = 200
sigma_firm = 0.25
debt = 180
recovery_rate = 0.2

#function to compute discounted call value
def call_payoff(S_T,K):
    return np.maximum(S_T-K,0)

np.random.seed(0)
corr_tested = np.linspace(-1,1,21)
cva_estimates = [None]*len(corr_tested)
cva_std = [None]*len(corr_tested)


for i in range(len(corr_tested)):

    correlation = corr_tested[i]
    if (correlation ==1 or correlation == -1):
        norm_vec_0 = norm.rvs(size = 50000)
        norm_vec_1 = correlation*norm_vec_0
        cor_norm_matrix = np.array([norm_vec_0,norm_vec_1])

    else:
        corr_matrix = np.array([[1,correlation],[correlation,1]])
        norm_matrix = norm.rvs(size=np.array([2,50000]))
        cor_norm_matrix = np.matmul(np.linalg.cholesky(corr_matrix),norm_matrix)

    term_stock_val = terminal_value(S_0,rfr,sigma,cor_norm_matrix[0,],T)
    call_val = call_payoff(term_stock_val,strike )
    term_firm_val = terminal_value(V_0,rfr,sigma_firm,cor_norm_matrix[1,],T)
    amount_lost = np.exp(-rfr*T)*(1-recovery_rate)*(term_firm_val < debt)*call_val
    cva_estimates[i] = np.mean(amount_lost)
    cva_std[i] = np.std(amount_lost)/np.sqrt(50000)

#Code to calculate default probablilty
d_1 = (np.log(V_0/debt)+(rfr+sigma_firm**2/2)*T)/(sigma_firm*np.sqrt(T))
d_2 = d_1 - sigma_firm*np.sqrt(T)
default_prob = norm.cdf(-d_2)

#Code for analytical solution for Vanilla European Call Option
d_1_stock = (np.log(S_0/strike)+(rfr+sigma**2/2)*T)/(sigma*np.sqrt(T))
d_2_stock = d_1_stock - sigma*np.sqrt(T)

analytical_callprice = S_0*norm.cdf(d_1_stock)-strike*np.exp(-rfr*T)*norm.cdf(d_2_stock)

uncorr_cva = (1-recovery_rate)*default_prob*analytical_callprice

plt.plot(corr_tested,[uncorr_cva]*21)
plt.plot(corr_tested,cva_estimates,'.')
plt.plot(corr_tested,cva_estimates+3*np.array(cva_std),'r')
plt.plot(corr_tested,cva_estimates-3*np.array(cva_std),'r')
plt.xlabel("Correlation")
plt.ylabel("CVA")
plt.show()
a= 1