# Importing the relevant libraries
import numpy as np
from scipy.stats import norm
import scipy.optimize

# Setting variables
r = 0.06
S0 = 90
K = 120
T = 3
KC = 100
TC = 2
price = 5

# Functions for call
def d_1c(x):
    return 1/(x*np.sqrt(TC))*(np.log(S0/KC)+(r+x**2/2)*TC)
def d_2c(x):
    return d_1c(x)-x*np.sqrt(TC)
def C(x):
    return norm.cdf(d_1c(x))*S0 - norm.cdf(d_2c(x))*KC*np.exp(-r*TC)
def F(x):
    return C(x)-price

# Solving for sigma
sigma = scipy.optimize.broyden1(F, 0.2)

print(sigma)
# Finding the put price
def d_1(x):
    return 1/(x*np.sqrt(T))*(np.log(S0/K)+(r+x**2/2)*T)
def d_2(x):
    return d_1(x)-x*np.sqrt(T)
def P(x):
    return -norm.cdf(-d_1(x))*S0 + norm.cdf(-d_2(x))*K*np.exp(-r*T)
put_price = P(sigma)

a = 1