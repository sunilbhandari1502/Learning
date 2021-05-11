import numpy as np
from scipy.stats import uniform
import scipy.optimize as opt
import matplotlib.pyplot as plt

price = 15

# def logchar_func(u,sigma):
#     return np.exp(1j*u*sigma)
#
# def option_price(sigma):
#     return (logchar_func(sigma)*sigma).real
#
# def F(sigma):
#     return np.absolute(option_price(sigma) - price)
#
# call_sigma = opt.fmin_slsqp(F,0.3)


def logchar_func(u,sigma):
    return np.exp(1j*u*sigma)

def option_price(sigma):
    return (logchar_func(2,sigma)*sigma).real

def F(sigma):
    return np.absolute(option_price(sigma) - price)

cali_sigma = opt.fmin_slsqp(F,0.3)


print(call_sigma)