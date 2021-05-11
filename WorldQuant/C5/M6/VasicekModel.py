#vasicek Model

#Importing relevant libraries
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import random

#Parameters
r0 = 0.05
alpha = 0.2
b = 0.08
sigma = 0.025

# Useful functions
def vasi_mean(r,t1,t2):
    """Gives the mean under the Vasicek model. Note that t2 > t1. r is the
    interest rate from the beginning of the period"""
    return np.exp(-alpha*(t2-t1))*r+b*(1-np.exp(-alpha*(t2-t1)))

def vasi_var(t1,t2):
    """Gives the variance under the Vasicek model. Note that t2 > t1"""
    return (sigma**2)*(1-np.exp(-2*alpha*(t2-t1)))/(2*alpha)

# Simulating interest rate paths
# NB short rates are simulated on an annual basis
np.random.seed(0)

n_years = 15
n_simulations = 10

t = np.array(range(0,n_years+1))


Z = norm.rvs(size = [n_simulations,n_years])
r_sim = np.zeros([n_simulations,n_years+1])
r_sim[:,0] = r0 #Sets the first column (the initial value of each simulation) to r(0)


for i in range(n_years):
    r_sim[:,i+1] = vasi_mean(r_sim[:,i],t[i],t[i+1]) + np.sqrt(vasi_var(t[i],t[i+1]))*Z[:,i]

s_mean = np.exp(-alpha*t)*r0 + b*(1-np.exp(-alpha*t))
# Plotting the results
t_graph = np.ones(r_sim.shape)*t
plt.plot(np.transpose(t_graph),np.transpose(r_sim*100),'r')
plt.plot(t,s_mean*100)
plt.show()