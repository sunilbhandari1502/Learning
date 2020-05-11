import numpy as np
import datetime
import pandas as pd
import time


def Exec(kite):
    currentTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    MarketClose = datetime.datetime.now().replace(hour=15, minute=30, second=0, microsecond=0).strftime(
        '%Y-%m-%d %H:%M:%S')
    MarketOpen = datetime.datetime.now().replace(hour=9, minute=15, second=0, microsecond=0).strftime(
        '%Y-%m-%d %H:%M:%S')

    # Strategy parameters
    strategySpread = 400
    bottomLevel = 1.11875
    topLevel = 1.13125
    month = 'OCT'
    nextweekExpiry = '03'
    exchangeOption = 'NFO'
    Year = 19
    optionTypeCall = 'CE'
    optionTypePut = 'PE'

    # Fetching Nifty current price
    exchange_NIFTY = 'NSE'
    instrument_token_Nifty50 = '256265'
    tradingsymbol_Nifty = 'NIFTY 50'
    lastPriceNifty = \
    kite.ltp('%s:%s' % (exchange_NIFTY, tradingsymbol_Nifty))['%s:%s' % (exchange_NIFTY, tradingsymbol_Nifty)][
        'last_price']
    # lastPriceNifty = 10700
    # getting the nearest 100 multiple of Nifty based on previous Nifty close
    nearest100Multiple = (np.round(lastPriceNifty / 100) * 100).astype(int)
    callStrike = nearest100Multiple - strategySpread
    putStrike = nearest100Multiple + strategySpread

    # Creating instrument name based on nifty level
    instrumentNameCall = 'NIFTY ' + nextweekExpiry + ' ' + month + nextweekExpiry + str(
        callStrike) + ' ' + optionTypeCall
    instrumentNamePut = 'NIFTY ' + nextweekExpiry + ' ' + month + ' ' + str(putStrike) + ' ' + optionTypePut
    tradeDetails = pd.DataFrame(
        columns=['TradeNo', 'Buy Time', 'Call Buy', 'Put Buy', 'Strangle Cost', 'Sell Time', 'Call Sell', 'Put Sell',
                 'Strangle Sell', 'P&L'])
    tradeNumber = 0

    # Fetching VIX data
    exchange_VIX = 'NSE'
    tradingsymbol_VIX = 'INDIA VIX'

    # check if position is open
    Positions = kite.positions()
    try:
        callOpenQuantity = Positions["net"][0]["overnight_quantity"] - Positions["net"][0]["overnight_quantity"]
    except:
        callOpenQuantity = 0
    if callOpenQuantity > 0:
        positionTakenBuy = True
        niftyActualOpenFlag = True
        instrumentNameCall = Positions["net"][0][
            "tradingsymbol"]  # not correct in case stocks are also traded and carried check
        instrumentNamePut = Positions["net"][1]["tradingsymbol"]
        callBuyPrice = Positions["net"][0]["buy_price"]
        putBuyPrice = Positions["net"][1]["buy_price"]
    else:
        positionTakenBuy = False
        niftyActualOpenFlag = False

    while datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') < MarketClose:
        if datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') < MarketOpen:
            continue
        # creating instrument ticker based on nifty opening level
        if niftyActualOpenFlag == False:
            lastPriceNifty = \
            kite.ltp('%s:%s' % (exchange_NIFTY, tradingsymbol_Nifty))['%s:%s' % (exchange_NIFTY, tradingsymbol_Nifty)][
                'last_price']
            fileName = datetime.datetime.now().strftime('%Y-%m-%d') + "_Pre1230_Output.txt"
            print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Nifty Open Price:", lastPriceNifty)
            print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Nifty Open Price:", lastPriceNifty,
                  file=open(fileName, "a"))
            # getting the nearest 100 multiple of Nifty based on previous Nifty close
            nearest100Multiple = (np.round(lastPriceNifty / 100) * 100).astype(int)
            callStrike = nearest100Multiple - strategySpread
            putStrike = nearest100Multiple + strategySpread
            # Creating instrument name based on nifty level
            instrumentNameCall = 'NIFTY' + str(year) + month + nextweekExpiry + str(callStrike) + optionTypeCall
            instrumentNameCall_1 = 'NIFTY' + str(year) + month + nextweekExpiry + str(callStrike + 100) + optionTypeCall
            instrumentNameCall_2 = 'NIFTY' + str(year) + month + nextweekExpiry + str(callStrike + 200) + optionTypeCall
            instrumentNameCall_3 = 'NIFTY' + str(year) + month + nextweekExpiry + str(callStrike + 300) + optionTypeCall
            instrumentNameCall_01 = 'NIFTY' + str(year) + month + nextweekExpiry + str(
                callStrike - 100) + optionTypeCall
            instrumentNameCall_02 = 'NIFTY' + str(year) + month + nextweekExpiry + str(
                callStrike - 200) + optionTypeCall
            instrumentNameCall_03 = 'NIFTY' + str(year) + month + nextweekExpiry + str(
                callStrike - 300) + optionTypeCall

            instrumentNamePut = 'NIFTY' + str(year) + month + nextweekExpiry + str(putStrike) + optionTypePut
            instrumentNamePut_1 = 'NIFTY' + str(year) + month + nextweekExpiry + str(putStrike + 100) + optionTypePut
            instrumentNamePut_2 = 'NIFTY' + str(year) + month + nextweekExpiry + str(callStrike + 200) + optionTypePut
            instrumentNamePut_3 = 'NIFTY' + str(year) + month + nextweekExpiry + str(callStrike + 300) + optionTypePut
            instrumentNamePut_01 = 'NIFTY' + str(year) + month + nextweekExpiry + str(callStrike - 100) + optionTypePut
            instrumentNamePut_02 = 'NIFTY' + str(year) + month + nextweekExpiry + str(callStrike - 200) + optionTypePut
            instrumentNamePut_03 = 'NIFTY' + str(year) + month + nextweekExpiry + str(callStrike - 300) + optionTypePut
            niftyActualOpenFlag = True

        # buying when straddle cost goes below intrinsic value
        if callOpenQuantity == 0:
            positionTakenBuy = False  # commented only for 31st Jan
        while positionTakenBuy == False and datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') < MarketClose:
            # time.sleep(1)
            try:
                # fetching OHLC for 4 tickers Nifty, VIX, Call and Put
                tempdata = kite.quote(
                    [exchangeOption + ":" + instrumentNameCall, exchangeOption + ":" + instrumentNamePut,
                     exchange_NIFTY + ":" + tradingsymbol_Nifty, exchange_VIX + ":" + tradingsymbol_VIX])
            except:
                pass
            try:
                bestOfferCall = tempdata[exchangeOption + ":" + instrumentNameCall]["depth"]["sell"][0]["price"]
                bestOfferPut = tempdata[exchangeOption + ":" + instrumentNamePut]["depth"]["sell"][0]["price"]
                bestBidCall = tempdata[exchangeOption + ":" + instrumentNameCall]["depth"]["buy"][0]["price"]
                bestBidPut = tempdata[exchangeOption + ":" + instrumentNamePut]["depth"]["buy"][0]["price"]
                lastTradeCall = tempdata[exchangeOption + ":" + instrumentNameCall]["last_price"]
                lastTradePut = tempdata[exchangeOption + ":" + instrumentNamePut]["last_price"]
            except:
                print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "No quotes for options")
                print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "No quotes for options",
                      file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))
                continue
            lastPriceVIX = tempdata[exchange_VIX + ":" + tradingsymbol_VIX]["last_price"]
            lastPriceNifty = tempdata[exchange_NIFTY + ":" + tradingsymbol_Nifty]["last_price"]
            StraddleCost = bestOfferCall + bestOfferPut
            if bestOfferCall == 0 or bestOfferPut == 0:
                print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Call or Put with zero value")
                print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Call or Put with zero value",
                      file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))
                continue
            print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "NIFTY:", lastPriceNifty, "VIX:", lastPriceVIX,
                  instrumentNameCall, bestBidCall, bestOfferCall, lastTradeCall, instrumentNamePut, bestBidPut,
                  bestOfferPut, lastTradePut, "StraddleBuy:", round(StraddleCost, 2), "StraddleSell:",
                  round((bestBidPut + bestBidCall), 2), "No Open Position")

            print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "NIFTY:", lastPriceNifty, "VIX:", lastPriceVIX,
                  instrumentNameCall, bestBidCall, bestOfferCall, lastTradeCall, instrumentNamePut, bestBidPut,
                  bestOfferPut, lastTradePut, "StraddleBuy:", round(StraddleCost, 2), "StraddleSell:",
                  round((bestBidPut + bestBidCall), 2), "No Open Position",
                  file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))

            if StraddleCost <= strategySpread * 2 * bottomLevel:
                callBuy = False
                putBuy = False
                while callBuy == False and datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') < MarketClose:
                    # Trying to buy a call
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Sending Call Option buy request")
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Sending Call Option buy request",
                          file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))
                    order_id_CallBuy = kite.place_order(tradingsymbol=instrumentNameCall, quantity=75,
                                                        exchange=exchangeOption, order_type="MARKET",
                                                        transaction_type="BUY", product="NRML", variety="regular")
                    orderbook = kite.orders()  # Fetching all orders details
                    # Finding the status of latest order placed which comes at the last dictionary of list
                    latestOrderStatus = [d['status'] for d in orderbook][-1]
                    tradeMessage = [d['status_message'] for d in orderbook][-1]
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Last trade status:",
                          latestOrderStatus,
                          "Trade Message:", tradeMessage)
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Last trade status", latestOrderStatus,
                          "Trade Message:", tradeMessage,
                          file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))
                    if latestOrderStatus != 'COMPLETE':
                        print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                              "Cannot buy Call Option trying again",
                              "Trade Message:", tradeMessage)
                        print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                              "Cannot buy Call Option trying again",
                              "Trade Message:", tradeMessage,
                              file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))
                        continue
                    # Fetch info from dict output using kite.orders()
                    callBuyPrice = [d['average_price'] for d in orderbook][-1]
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Call Option bought: ", callBuyPrice)
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Call Option bought: ", callBuyPrice,
                          file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))
                    callBuy = True
                while putBuy == False and datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') < MarketClose:
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Sending Put Option buy request")
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Sending Put Option buy request",
                          file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))
                    order_id_PutBuy = kite.place_order(tradingsymbol=instrumentNamePut, quantity=75,
                                                       exchange=exchangeOption, order_type="MARKET",
                                                       transaction_type="BUY",
                                                       product="NRML", variety="regular")
                    orderbook = kite.orders()
                    latestOrderStatus = [d['status'] for d in orderbook][-1]
                    tradeMessage = [d['status_message'] for d in orderbook][-1]
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Last trade status:",
                          latestOrderStatus,
                          "Trade Message:", tradeMessage)
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Last trade status", latestOrderStatus,
                          "Trade Message:", tradeMessage,
                          file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))
                    if latestOrderStatus != 'COMPLETE':
                        print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                              "Cannot buy Put Option trying again",
                              "Trade Message:", tradeMessage)
                        print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                              "Cannot buy Put Option trying again",
                              "Trade Message:", tradeMessage,
                              file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))
                        continue
                    putBuyPrice = [d['average_price'] for d in orderbook][-1]
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Put Option bought: ", putBuyPrice)
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Put Option bought: ", putBuyPrice,
                          file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))
                    putBuy = True
                positionTakenBuy = True
                BuyTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # selling when straddle cost goes below intrinsic value
        positionTakenSell = False
        while positionTakenSell == False and datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') < MarketClose:
            # time.sleep(1)
            try:
                tempdata = kite.quote(
                    [exchangeOption + ":" + instrumentNameCall, exchangeOption + ":" + instrumentNamePut,
                     exchange_NIFTY + ":" + tradingsymbol_Nifty, exchange_VIX + ":" + tradingsymbol_VIX])
            except:
                pass
            try:
                bestOfferCall = tempdata[exchangeOption + ":" + instrumentNameCall]["depth"]["sell"][0]["price"]
                bestOfferPut = tempdata[exchangeOption + ":" + instrumentNamePut]["depth"]["sell"][0]["price"]
                bestBidCall = tempdata[exchangeOption + ":" + instrumentNameCall]["depth"]["buy"][0]["price"]
                bestBidPut = tempdata[exchangeOption + ":" + instrumentNamePut]["depth"]["buy"][0]["price"]
                lastTradeCall = tempdata[exchangeOption + ":" + instrumentNameCall]["last_price"]
                lastTradePut = tempdata[exchangeOption + ":" + instrumentNamePut]["last_price"]
            except:
                print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "No quotes for options")
                print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "No quotes for options",
                      file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))
                continue
            lastPriceVIX = tempdata[exchange_VIX + ":" + tradingsymbol_VIX]["last_price"]
            lastPriceNifty = tempdata[exchange_NIFTY + ":" + tradingsymbol_Nifty]["last_price"]
            StraddleSell = bestBidCall + bestBidPut
            if bestBidCall == 0 or bestBidPut == 0:
                print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Call or Put with zero value")
                print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Call or Put with zero value",
                      file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))
                continue
            print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "NIFTY:", lastPriceNifty, "VIX:", lastPriceVIX,
                  instrumentNameCall, bestBidCall, bestOfferCall, lastTradeCall, instrumentNamePut, bestBidPut,
                  bestOfferPut, lastTradePut, "StraddleBuy:", round((bestOfferPut + bestOfferCall), 2), "StraddleSell:",
                  round(StraddleSell, 2), "Open position at", callBuyPrice + putBuyPrice, "P&L:",
                  round(((StraddleSell - (callBuyPrice + putBuyPrice)) * 75), 2))

            print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "NIFTY:", lastPriceNifty, "VIX:", lastPriceVIX,
                  instrumentNameCall, bestBidCall, bestOfferCall, lastTradeCall, instrumentNamePut, bestBidPut,
                  bestOfferPut, lastTradePut, "StraddleBuy:", round((bestOfferPut + bestOfferCall), 2), "StraddleSell:",
                  round(StraddleSell, 2), "Open position at", callBuyPrice + putBuyPrice, "P&L:",
                  round(((StraddleSell - (callBuyPrice + putBuyPrice)) * 75), 2),
                  file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))

            if StraddleCost >= strategySpread * 2 * topLevel:
                # Sell call
                callSell = False
                putSell = False
                while callSell == False and datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') < MarketClose:
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Sending Call Option sell request")
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Sending Call Option sell request",
                          file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))
                    order_id_CallSell = kite.place_order(tradingsymbol=instrumentNameCall, quantity=75,
                                                         exchange=exchangeOption, order_type="MARKET",
                                                         transaction_type="SELL", product="NRML",
                                                         variety="regular")
                    orderbook = kite.orders()
                    latestOrderStatus = [d['status'] for d in orderbook][-1]
                    tradeMessage = [d['status_message'] for d in orderbook][-1]
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Last trade status:",
                          latestOrderStatus,
                          "Trade Message:", tradeMessage)
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Last trade status", latestOrderStatus,
                          "Trade Message:", tradeMessage,
                          file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))
                    if latestOrderStatus != 'COMPLETE':
                        print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                              "Cannot sell Call Option trying again",
                              "Trade Message:", tradeMessage)
                        print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                              "Cannot sell Call Option trying again",
                              "Trade Message:", tradeMessage,
                              file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))
                        continue
                    callSellPrice = [d['average_price'] for d in orderbook][-1]
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Call Option Sold: ", callSellPrice)
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Call Option Sold: ", callSellPrice,
                          file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt",
                                    "a"))  # Fetch info from dict output using kite.orders()
                    callSell = True
                while putSell == False and datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') < MarketClose:
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Sending Put Option sell request")
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Sending Put Option sell request",
                          file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))
                    order_id_PutSell = kite.place_order(tradingsymbol=instrumentNamePut, quantity=75,
                                                        exchange=exchangeOption,
                                                        order_type="MARKET", transaction_type="SELL", product="NRML",
                                                        variety="regular")
                    orderbook = kite.orders()
                    latestOrderStatus = [d['status'] for d in orderbook][-1]
                    tradeMessage = [d['status_message'] for d in orderbook][-1]
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Last trade status:",
                          latestOrderStatus,
                          "Trade Message:", tradeMessage)
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Last trade status", latestOrderStatus,
                          "Trade Message:", tradeMessage,
                          file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))
                    if latestOrderStatus != 'COMPLETE':
                        print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                              "Cannot sell Call Option trying again",
                              "Trade Message:", tradeMessage)
                        print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                              "Cannot sell Call Option trying again",
                              "Trade Message:", tradeMessage,
                              file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))
                        continue
                    putSellPrice = [d['average_price'] for d in orderbook][-1]
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Put Option Sold: ", putSellPrice)
                    print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "Put Option Sold: ", putSellPrice,
                          file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))
                    putSell = True
                positionTakenSell = True
                callOpenQuantity = 0
                SellTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # time.sleep(1)
        tradeNumber = tradeNumber + 1
        try:
            tradeDetails.loc[len(tradeDetails)] = [tradeNumber, BuyTime, callBuyPrice, putBuyPrice,
                                                   callBuyPrice + putBuyPrice,
                                                   SellTime, callSellPrice, putSellPrice, callSellPrice + putSellPrice,
                                                   (callBuyPrice + putBuyPrice - callSellPrice - putSellPrice) * 75]

            print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "TradeComplete:", tradeNumber, callBuyPrice,
                  putBuyPrice, callBuyPrice + putBuyPrice,
                  callSellPrice, putSellPrice, callSellPrice + putSellPrice,
                  (callBuyPrice + putBuyPrice - callSellPrice - putSellPrice) * 75)
            print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "TradeComplete:", tradeNumber, callBuyPrice,
                  putBuyPrice, callBuyPrice + putBuyPrice,
                  callSellPrice, putSellPrice, callSellPrice + putSellPrice,
                  (callBuyPrice + putBuyPrice - callSellPrice - putSellPrice) * 75,
                  file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))
        except:
            print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                  "Day Closed with either no Trade or Open position")
            print(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                  "Day Closed with either no Trade or Open position",
                  file=open(datetime.datetime.now().strftime('%Y-%m-%d') + "output.txt", "a"))
            pass
        tradeDetails.to_csv('tradeDetails.csv')
    return
