# Importing libraries
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.optimize

# Creating a yield curve for integer maturities
years = np.linspace(1,10,10)
yield_curve = (years)**(1/5)/75 + 0.04
bond_prices = np.exp(-yield_curve*years)

# Plotting the yield curve
plt.plot(years,yield_curve*100)

# Analytical bond price
def A(t1,t2,alpha):
    return (1-np.exp(-alpha*(t2-t1)))/alpha

def D(t1,t2,alpha,b,sigma):
    val1 = (t2-t1-A(t1,t2,alpha))*(sigma**2/(2*alpha**2)-b)
    val2 = sigma**2*A(t1,t2,alpha)**2/(4*alpha)
    return val1-val2

def bond_price_fun(r,t,T,alpha,b,sigma):
    return np.exp(-A(t,T,alpha)*r+D(t,T,alpha,b,sigma))

r0 = 0.05

# Difference between model bond prices and yield curve bond prices
def F(x):
    alpha = x[0]
    b = x[1]
    sigma = x[2]
    return sum(np.abs(bond_price_fun(r0,0,years,alpha,b,sigma)-bond_prices))

# Minimizing F
bnds = ((0,1),(0,0.2),(0,0.2))
opt_val = scipy.optimize.fmin_slsqp(F, (0.3,0.05,0.03), bounds=bnds)
opt_alpha = opt_val[0]
opt_b = opt_val[1]
opt_sig = opt_val[2]
print(opt_val)

# Calculating model prices and yield
model_prices = bond_price_fun(r0,0,years,opt_alpha,opt_b,opt_sig)
model_yield = -np.log(model_prices)/years

# Plotting the market vs model prices
plt.plot(years,bond_prices)
plt.plot(years,model_prices,'.')

# Plotting the market vs model yields
plt.plot(years,yield_curve*100)
plt.plot(years,model_yield*100,'x')