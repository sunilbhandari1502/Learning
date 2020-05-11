import numpy as np


exchange_NIFTY='NSE'
instrument_token_Nifty50 = '256265'
tradingsymbol_Nifty = 'NIFTY 50'
lastPriceNifty = kite.ltp('%s:%s'%(exchange_NIFTY, tradingsymbol_Nifty))['%s:%s'%(exchange_NIFTY, tradingsymbol_Nifty)]['last_price']
nearest100Multiple = (np.round(lastPriceNifty/100)*100).astype(int)

strategySpread = 400
callStrike = nearest100Multiple - strategySpread
putStrike = nearest100Multiple + strategySpread

month = 'DEC'
exchangeOption = 'NFO'
Year = 18
optionTypeCall = 'CE'
optionTypePut = 'PE'


instrumentNameCall = 'NIFTY'+str(Year)+month+str(callStrike)+optionTypeCall
instrumentNamePut = 'NIFTY'+str(Year)+month+str(putStrike)+optionTypePut


# buying when straddle cost goes below intrinsic value
positionTakenBuy = False
while positionTakenBuy == False:
    lastPriceCall = kite.ltp('%s:%s' % (exchangeOption, instrumentNameCall))['%s:%s' % (exchangeOption, instrumentNameCall)]['last_price']
    lastPricePut = kite.ltp('%s:%s' % (exchangeOption, instrumentNamePut))['%s:%s' % (exchangeOption, instrumentNamePut)]['last_price']
    StraddleCost = lastPriceCall + lastPricePut
    if StraddleCost <= strategySpread*2*0.995:
        #Buy call
        callBuy = False
        putBuy = False
        while callBuy == False:
            order_id_CallBuy = kite.place_order(tradingsymbol=instrumentNameCall, quantity= 1, exchange=exchangeOption, order_type="MARKET",transaction_type="BUY", product="NRML",variety = "regular")
            if len(kite.trades(order_id_CallBuy)) == 0:
                print("Cannot buy Call Option")
            else:
                print("Call Option bought") #Fetch info from dict output using kite.orders()
                callBuy = True
        while putBuy == False:
            order_id_PutBuy = kite.place_order(tradingsymbol=instrumentNamePut, quantity= 1, exchange=exchangeOption, order_type="MARKET",transaction_type="BUY", product="NRML")
            if len(kite.trades(order_id_PutBuy)) == 0:
               print("Cannot buy Put Option")
            else:
                print("Put Option bought")
                putBuy = True
        positionTakenBuy = True

# selling when straddle cost goes below intrinsic value
positionTakenSell = False
while positionTakenSell == False:
    lastPriceCall = kite.ltp('%s:%s' % (exchangeOption, instrumentNameCall))['%s:%s' % (exchangeOption, instrumentNameCall)]['last_price']
    lastPricePut = kite.ltp('%s:%s' % (exchangeOption, instrumentNamePut))['%s:%s' % (exchangeOption, instrumentNamePut)]['last_price']
    StraddleCost = lastPriceCall + lastPricePut
    if StraddleCost >= strategySpread*2*1.025:
        #Sell call
        callSell = False
        putSell = False
        while callSell == False:
            order_id_CallSell = kite.place_order(tradingsymbol=instrumentNameCall, quantity= 1, exchange=exchangeOption, order_type="MARKET",transaction_type="SELL", product="NRML",variety = "regular")
            if len(kite.trades(order_id_CallSell)) == 0:
                print("Cannot Sell Call Option")
            else:
                print("Call Option Sold") #Fetch info from dict output using kite.orders()
                callSell = True
        while putSell == False:
            order_id_PutSell = kite.place_order(tradingsymbol=instrumentNamePut, quantity= 1, exchange=exchangeOption, order_type="MARKET",transaction_type="SELL", product="NRML")
            if len(kite.trades(order_id_PutSell)) == 0:
               print("Cannot buy Put Option")
            else:
                print("Put Option bought")
                putSell = True
        positionTakenSell = True



