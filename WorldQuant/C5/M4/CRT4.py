import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
#share specific information
r = 0.10
S0 = 120
sigma = 0.25

# call option specific information
K = 100
T = 2
k_log = np.log(K)


#BSM
d_1_stock = (np.log(S0/K) + (r+sigma**2/2)*(T))/(sigma*np.sqrt(T))
d_2_stock = d_1_stock - sigma*np.sqrt(T)
analytical_putprice = K*np.exp(-r*(T))*norm.cdf(-d_2_stock) - S0 * norm.cdf(-d_1_stock)

print("BSM Put Price: ", analytical_putprice)

# Pricing using Characteristic funtions
def c_M1(t):
    return np.exp(-1j*t*(np.log(S0)+(r-sigma**2/2)*T)-(sigma**2)*T*(t**2)/2)
def c_M2(t):
    return np.exp(-1j*t*sigma**2*T)*c_M1(t)

# Choosing t_max and N
t_max = 40
N = 200

# Calculating delta and constructing t_n
delta_t = t_max / N
from_1_to_N = np.linspace(1, N, N)
t_n = (from_1_to_N - 1 / 2) * delta_t

# Approximate integral estimates
first_integral = sum((((np.exp(1j * t_n * k_log) * c_M2(t_n)).imag) / t_n) * delta_t)
second_integral = sum((((np.exp(1j * t_n * k_log) * c_M1(t_n)).imag) / t_n) * delta_t)

#Put value
fourier_put_value = np.exp(-r*T)*K*(1/2+second_integral/np.pi)-S0*(1/2+first_integral/np.pi)
print("Characteristic function Put price: ",fourier_put_value)
#Plotting the .plot([fourier_put_value]*50)
plt.plot([analytical_putprice]*50,'.')
plt.xlabel("BSM vs Charateristic fufnction Put price")
plt.ylabel("Put price")
plt.show()

'''Pricing using Fourier Cos expansion'''
def upsilon_n(b2,b1,d,c,n):
    npi_d = np.pi*n*(d-b1)/(b2-b1)
    npi_c = np.pi*n*(c-b1)/(b2-b1)
    val_one = (np.cos(npi_d)*np.exp(d)-np.cos(npi_c)*np.exp(c))
    val_two = (n*np.pi*(np.sin(npi_d)*np.exp(d)-np.sin(npi_c)*np.exp(c))/(b2-b1))
    return (val_one+val_two)/(1+(n*np.pi/(b2-b1))**2)
def psi_n(b2,b1,d,c,n):
    if n==0:
        return d-c
    else:
        return(b2-b1)*(np.sin(n*np.pi*(d-b1)/(b2-b1)) - np.sin(n*np.pi*(c-b1)/(b2-b1)))/(n*np.pi)
# functions for call valuation
def v_n(K, b2, b1,n):
    return 2*K*(psi_n(b2,b1,0,b1,n)-upsilon_n(b2,b1,0,b1,n) )/(b2-b1)
def logchar_func(u,S0,r,sigma,K,T):
    return np.exp(1j*u*(np.log(S0/K) + (r-sigma**2/2)*T) - (sigma**2)*T*(u**2)/2)
def put_price(N,S0,sigma,r,K,T,b2,b1):
    price = v_n(K,b2,b1,0)*logchar_func(0,S0,r,sigma,K,T)/2
    for n in range(1,N):
        price = price + logchar_func(n*np.pi/(b2-b1),S0,r,sigma,K,T)*np.exp(-1j*n*np.pi*b1/(b2-b1))*v_n(K,b2,b1,n)
    return price.real*np.exp(-r*T)

#b1 , b2 for call
c1 = r
c2 = T*sigma**2
c4 = 0
L = 10
b1 = c1 - L*np.sqrt(c2-np.sqrt(c4))
b2 = c1 + L*np.sqrt(c2-np.sqrt(c4))
# Calculating COS for various N
COS_putprice = [None] * 50
for i in range(1, 51):
    COS_putprice[i-1] = put_price(i, S0, sigma, r, K, T, b2, b1)
#Plotting the results
plt.plot(COS_putprice)
plt.plot([analytical_putprice]*50)
plt.xlabel("N")
plt.ylabel("Put price")
plt.show()
#Plotting the log absolute error
plt.plot(np.log(np.absolute(COS_putprice - analytical_putprice)))
plt.xlabel("N")
plt.ylabel("Log absolute error")
plt.show()


'''Pricing using fast fourier transformation'''
def fft(x):
    N = len(x)
    if N ==1:
        return x
    else:
        ek = fft(x[:-1:2])
        ok = fft(x[1::2])
        m = np.array(range(int(N/2)))
        okm = ok*np.exp(-1j*2*np.pi*m/N)
        return np.concatenate((ek+okm,ek-okm))

#algo info
N = 2**10
delta = 0.25
alpha = -1.5

def log_char(u):
    return np.exp(1j*u*(np.log(S0)+(r-sigma**2/2)*T)-sigma**2*T*u**2/2)
def c_func(v):
    val1 = np.exp(-r*T)*log_char(v-(alpha+1)*1j)
    val2 = alpha**2+alpha-v**2+1j*(2*alpha+1)*v
    return val1/val2
n = np.array(range(N))
delta_k = 2*np.pi/(N*delta)
b = delta_k*(N-1)/2
log_strike = np.linspace(-b,b,N)
actual_Strikes = np.exp(log_strike)
print(np.exp(actual_Strikes))

x = np.exp(1j*b*n*delta)*c_func(n*delta)*delta
x[0] = x[0]*0.5
x[-1] = x[-1] * 0.5
xhat = fft(x).real
fft_put = np.exp(-alpha*log_strike)*xhat/np.pi
#Black Scholes put option value for given strikes
d_1 = (np.log(S0/np.exp(log_strike)) + (r+sigma**2/2)*T)/(sigma*np.sqrt(T))
d_2 = d_1 -sigma*np.sqrt(T)
analytic_putprice = np.exp(log_strike)*np.exp(-r*(T))*norm.cdf(-d_2)-S0*norm.cdf(-d_1)

print("FFT BSM Put price: ",analytic_putprice)
print("FFT Put price: ",fft_put)
plt.plot(np.exp(log_strike[:700]),fft_put[:700],'.',label = "FFT Call Estimate")
plt.plot(np.exp(log_strike[:700]),analytic_putprice[:700],label = "Analytical Call Estimate")
plt.xlabel("Strike")
plt.ylabel("Call price")
plt.legend()
plt.show()