import numpy as np
import datetime
import pandas as pd
import time
import Code.sendEmail as SE

def Exec(kite):
    currentTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    MarketClose = datetime.datetime.now().replace(hour=15, minute=30, second=0, microsecond=0).strftime('%Y-%m-%d %H:%M:%S')
    MarketOpen = datetime.datetime.now().replace(hour=9, minute=15, second=0, microsecond=0).strftime('%Y-%m-%d %H:%M:%S')

    # Strategy parameters
    strategySpread = 0
    bottomLevel = 1.11875
    topLevel = 1.13125
    month = 'O'
    nextweekExpiry = '24'
    exchangeOption = 'NFO'
    year = 19
    optionTypeCall = 'CE'
    optionTypePut = 'PE'

    # Fetching Nifty current price
    exchange_NIFTY = 'NSE'
    instrument_token_Nifty50 = '256265'
    tradingsymbol_Nifty = 'NIFTY 50'
    lastPriceNifty = kite.ltp('%s:%s' % (exchange_NIFTY, tradingsymbol_Nifty))['%s:%s' % (exchange_NIFTY, tradingsymbol_Nifty)]['last_price']
    # lastPriceNifty = 10700
    # getting the nearest 100 multiple of Nifty based on previous Nifty close
    nearest100Multiple = (np.round(lastPriceNifty / 100) * 100).astype(int)
    callStrike = nearest100Multiple - strategySpread
    putStrike = nearest100Multiple + strategySpread

     # Fetching VIX data
    exchange_VIX = 'NSE'
    tradingsymbol_VIX = 'INDIA VIX'

    # check if position is open
    #Positions = kite.positions()

    niftyActualOpenFlag = False
    preLunch = True


    return