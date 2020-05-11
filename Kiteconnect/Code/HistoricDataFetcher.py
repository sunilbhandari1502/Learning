import pandas as pd
import datetime


#Read instrument code file
# input_file_path = ("C:\\Users\\sunilbhandari1\\Desktop\\Kiteconnect\\Code\\instrumentsId.xlsx")
# input_file = pd.ExcelFile(input_file_path)
# df_instruCode = input_file.parse("InstrumentsId")

#fetching all instruments using insturment call and converting that list of dictionary to DataFrame
allInstrument_list = kite.instruments()    # can pass echange if need list of particular exchange, removing will give for all exchange
allInstrument_dataFrame = pd.DataFrame(allInstrument_list)


startingStrike = 9000
endingStrike = 12000
underlying = 'NIFTY'
month = 'JAN'
year = 19
from_date = '2018-01-08'
to_date = '2019-01-15'
interval = 'minute'  #3minute
continuous = False

tempStrike = startingStrike
DataAllStrike = pd.DataFrame()

while tempStrike <= endingStrike:
    optionType = 'CE'
    callInstrument = underlying+str(year)+month+str(tempStrike)+optionType
    instrument_Code = allInstrument_dataFrame[allInstrument_dataFrame['tradingsymbol'] == callInstrument]
    instrument_Code = instrument_Code['instrument_token'].values[0]
    tempDataCall_list = kite.historical_data(instrument_token=instrument_Code, from_date=from_date, to_date=to_date, interval=interval,continuous=continuous)
    tempDataCall = pd.DataFrame(tempDataCall_list)
    tempDataCall['Instrument'] = callInstrument
    if pd.DataFrame(tempDataCall).empty:
        tempStrike = tempStrike + 50
        continue
    else:
        tempDataCall = tempDataCall.set_index('date')

    optionType = 'PE'
    putInstrument = underlying + str(year) + month + str(tempStrike) + optionType
    instrument_Code = allInstrument_dataFrame[allInstrument_dataFrame['tradingsymbol'] == putInstrument]
    instrument_Code = instrument_Code['instrument_token'].values[0]
    tempDataPut_list = kite.historical_data(instrument_token=instrument_Code, from_date=from_date, to_date=to_date,interval=interval, continuous=continuous)
    tempDataPut = pd.DataFrame(tempDataPut_list)
    tempDataPut['Instrument'] = putInstrument
    if pd.DataFrame(tempDataPut).empty:
        tempStrike = tempStrike + 50
        continue
    else:
        tempDataPut = tempDataPut.set_index('date')

    # tempDataStrike = tempDataCall.append(tempDataPut)
    tempDataStrike = pd.merge(tempDataCall[['close','high','low','open','volume','Instrument']], tempDataPut[['close','high','low','open','volume','Instrument']], how='outer', left_index=True, right_index=True)

    # tempDataStrike = pd.concat([tempDataCall, tempDataPut], axis=0)
    DataAllStrike = pd.merge(DataAllStrike, tempDataStrike, how='outer', left_index=True, right_index=True)
    tempStrike = tempStrike + 50



tempDataNifty_list = kite.historical_data(instrument_token='256265', from_date=from_date, to_date=to_date,interval=interval, continuous=continuous)
tempDataNifty = pd.DataFrame(tempDataNifty_list)
tempDataNifty['Instrument'] = 'NIFTY 50'
tempDataNifty = tempDataNifty.set_index('date')

tempDataVIX_list = kite.historical_data(instrument_token='264969', from_date=from_date, to_date=to_date, interval=interval, continuous=continuous)
tempDataVIX = pd.DataFrame(tempDataVIX_list)
tempDataVIX['Instrument'] = 'India VIX'
tempDataVIX = tempDataVIX.set_index('date')


NiftyFutureInstrument = underlying+str(year)+month+'FUT'
instrument_Code = allInstrument_dataFrame[allInstrument_dataFrame['tradingsymbol'] == NiftyFutureInstrument]
instrument_Code = instrument_Code['instrument_token'].values[0]
tempDataNiftyFuture_list = kite.historical_data(instrument_token=instrument_Code, from_date=from_date, to_date=to_date,interval=interval, continuous=continuous)
tempDataNiftyFuture = pd.DataFrame(tempDataNiftyFuture_list)
tempDataNiftyFuture['Instrument'] = NiftyFutureInstrument
tempDataNiftyFuture = tempDataNiftyFuture.set_index('date')

tempDataStrikeNiftyNFutNVIX = pd.merge(tempDataNifty[['close', 'high', 'low', 'open', 'volume', 'Instrument']],
                                       tempDataNiftyFuture[['close', 'high', 'low', 'open', 'volume', 'Instrument']],
                                       how='outer', left_index=True, right_index=True)
tempDataStrikeNiftyNFutNVIX = pd.merge(tempDataStrikeNiftyNFutNVIX, tempDataVIX,
                                       how='outer',left_index=True, right_index=True)

currentTime = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

DataAllStrike.to_csv('AllStrikeDataOpt_'+currentTime+'.csv')
tempDataStrikeNiftyNFutNVIX.to_csv('AllStrikeDataNiftyVixOption_'+currentTime+'.csv')