import numpy as np
import pandas as pd
import datetime
import yfinance as yf
# Slecting SPY and other two ETFs
# VOO Vanguard S&P 500 ETF,
# IVV ishare core S&P500
#GSPC S&P 500
tickers = ["SPY","VOO","IVV","^GSPC"]
df= yf.download(tickers, start= "2019-01-01", end= "2020-12-31")
df = pd.DataFrame(df.Close)
df.head()

#Taking the log return of the series
df_log_return = np.log(df[df.columns[:]]).diff()
df_log_return = pd.DataFrame(df_log_return[:][1:])
df_log_return.head()

# active return
ivv_sp500 = df_log_return['IVV']-df_log_return['^GSPC']
voo_sp500 = df_log_return['VOO']-df_log_return['^GSPC']
spy_sp500 = df_log_return['SPY']-df_log_return['^GSPC']

active_retun = pd.DataFrame()
active_retun['ivv_sp500'] = ivv_sp500
active_retun['voo_sp500'] = voo_sp500
active_retun['spy_sp500'] = spy_sp500
active_retun.head()

# Average Active return
active_retun.mean()

#Compute the tracking error
tracking_error_ivv = ivv_sp500.std()
tracking_error_voo = voo_sp500.std()
tracking_error_spy = spy_sp500.std()

print ('tracking error of IVV is:', tracking_error_ivv*100,"%" )
print ('tracking error of VOO is:', tracking_error_voo*100,"%" )
print ('tracking error of SPY is:', tracking_error_spy*100,"%" )

#mean-adjusted tracking error.
MATE_ivv = np.sqrt((np.square(ivv_sp500)).mean())
MATE_voo = np.sqrt((np.square(voo_sp500)).mean())
MATE_spy = np.sqrt((np.square(spy_sp500)).mean())
print ('MATE of IVV is:', MATE_ivv*100,"%" )
print ('MATE of VOO is:', MATE_voo*100,"%" )
print ('MATE of SPY is:', MATE_spy*100,"%" )

# selecting spdrs
spdrs = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
df_spdrs= yf.download(spdrs, start= "2019-01-01", end= "2020-12-31")
df_spdrs = pd.DataFrame(df_spdrs.Close)
df_spdrs.head()

#Taking the log return of the series
df_spdrs_log_return = np.log(df_spdrs[df_spdrs.columns[:]]).diff()
df_spdrs_log_return = pd.DataFrame(df_spdrs_log_return[:][1:])
df_spdrs_log_return.head()

# active returns
active_retun_XLB = df_spdrs_log_return["XLB"] - df_log_return["^GSPC"]
active_retun_XLE = df_spdrs_log_return["XLE"] - df_log_return["^GSPC"]
active_retun_XLF = df_spdrs_log_return["XLF"] - df_log_return["^GSPC"]
active_retun_XLI = df_spdrs_log_return["XLI"] - df_log_return["^GSPC"]
active_retun_XLK = df_spdrs_log_return["XLK"] - df_log_return["^GSPC"]
active_retun_XLP = df_spdrs_log_return["XLP"] - df_log_return["^GSPC"]
active_retun_XLRE = df_spdrs_log_return["XLRE"] - df_log_return["^GSPC"]
active_retun_XLU = df_spdrs_log_return["XLU"] - df_log_return["^GSPC"]
active_retun_XLV = df_spdrs_log_return["XLV"] - df_log_return["^GSPC"]
active_retun_XLY = df_spdrs_log_return["XLY"] - df_log_return["^GSPC"]
#MATE
print ( "MATE_XLB is=", np.sqrt(active_retun_XLB).mean()*100,"%")
print ("MATE_XLE is=", np.sqrt(active_retun_XLE).mean()*100,"%")
print ("MATE_XLF =", np.sqrt(active_retun_XLF).mean()*100,"%")
print("MATE_XLI =", np.sqrt(active_retun_XLI).mean()*100,"%")
print ("MATE_XLK =", np.sqrt(active_retun_XLK).mean()*100,"%")
print ("MATE_XLP =", np.sqrt(active_retun_XLP).mean()*100,"%")
print ("MATE_XLRE =", np.sqrt(active_retun_XLRE).mean()*100,"%")
print ("MATE_XLU =", np.sqrt(active_retun_XLU).mean()*100,"%")
print ("MATE_XLV =", np.sqrt(active_retun_XLV).mean()*100,"%")
print ("MATE_XLY =", np.sqrt(active_retun_XLY).mean()*100,"%")