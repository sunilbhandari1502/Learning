import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import random

#Share related information
rfr = 0.1
S_0 = 100
sigma = 0.3

#Option related information

strike = 110
T = 1
current_time = 0


#function for terminal stock value
def terminal_shareprice(S_0,rfr,sigma,Z,T):
    return S_0*np.exp((rfr-sigma**2/2)*T+sigma*np.sqrt(T)*Z)

#function to compute discounted call value
def discouted_call_payoff(S_T,K,rfr,T):
    return np.exp(-rfr*T)*np.maximum(S_T-K,0)


np.random.seed(0)
monte_put_estimate = [None]*50
monte_put_std = [None]*50

for i in range(1,51):
    norm_array = norm.rvs(size = i*1000)
    monte_term_val = terminal_shareprice(S_0,rfr,sigma,norm_array,T-current_time)
    monte_put_val = discouted_call_payoff(monte_term_val,strike,rfr,T-current_time)
    monte_put_estimate[i-1] = np.mean(monte_put_val)
    monte_put_std[i-1] = np.std(monte_put_val)/np.sqrt(i*1000)

#Black scholes value of put option

d_1 = (math.log(S_0/strike)+(rfr+sigma**2/2)*(T-current_time))/(sigma*math.sqrt(T-current_time))
d_2 = d_1 - sigma*math.sqrt(T-current_time)
analytic_put_price = S_0*norm.cdf(d_1)-strike*math.exp(-rfr*(T-current_time))*norm.cdf(d_2)

plt.plot([analytic_put_price]*50)
plt.plot(monte_put_estimate ,'.')
plt.plot(analytic_put_price+np.array(monte_put_std)*3, 'r')
plt.plot(analytic_put_price-np.array(monte_put_std)*3,'r')
plt.xlabel("Sample Size")
plt.ylabel("Value")
plt.show()
