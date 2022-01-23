import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
import numpy as np

# Fetching price for IBM and Apple from yahoo Finance
ibm = pdr.get_data_yahoo(symbols='IBM', start=datetime(2019, 5, 1), end=datetime(2021, 5, 1))
apple = pdr.get_data_yahoo(symbols='AAPL', start=datetime(2019, 5, 1), end=datetime(2021, 5, 1))

# adding log return to the dataframes
ibm['ibm_log_ret'] = np.log(ibm['Adj Close']) - np.log(ibm['Adj Close'].shift(1))
apple['apple_log_ret'] = np.log(apple['Adj Close']) - np.log(apple['Adj Close'].shift(1))

# Mean, Standard deviation Skewness and excess Kurtosis
ibm_mean = ibm['ibm_log_ret'].mean()
ibm_std = ibm['ibm_log_ret'].std()
ibm_skewness = ibm['ibm_log_ret'].skew()
ibm_kurtosis = ibm['ibm_log_ret'].kurtosis()
print('IBM \nMean:', ibm_mean, '\nStandard Deviation :', ibm_std, '\nSkewness :', ibm_skewness, '\nKurtosis :',
      ibm_kurtosis)

apple_mean = apple['apple_log_ret'].mean()
apple_std = apple['apple_log_ret'].std()
apple_skewness = apple['apple_log_ret'].skew()
apple_kurtosis = apple['apple_log_ret'].kurtosis()
print('Apple \nMean:', apple_mean, '\nStandard Deviation :', apple_std, '\nSkewness :', apple_skewness, '\nKurtosis :',
      apple_kurtosis)

# covariance and correlation
covariance = np.cov(ibm['ibm_log_ret'][1:], apple['apple_log_ret'][1:])
correlation = np.corrcoef(ibm['ibm_log_ret'][1:], apple['apple_log_ret'][1:])
print('Covariance:', covariance[0][1], '\nCorrelation :', correlation[1][0], '\nibm_Vol :', np.sqrt(covariance[0][0]),
      '\napple_vol :', np.sqrt(covariance[1][1]))
print('Correlation using Covariance and vols : ',
      covariance[0][1] / (np.sqrt(covariance[0][0]) * np.sqrt(covariance[1][1])))

trainIBM = ibm['Adj Close'][0:402]
testIBM = ibm['Adj Close'][403:]
trainApple = apple['Adj Close'][0:402]
testApple = apple['Adj Close'][403:]
trainMovement = pd.DataFrame(index=range(401), columns=['Movement'])
testMovement = pd.DataFrame(index=range(102), columns=['Movement'])
for i in range(1, len(trainIBM), 1):
    if trainIBM[i] > trainIBM[i - 1]:
        if trainApple[i] > trainApple[i - 1]:
            trainMovement['Movement'][i - 1] = 'UU'
        else:
            trainMovement['Movement'][i - 1] = 'UD'
    else:
        if trainApple[i] > trainApple[i - 1]:
            trainMovement['Movement'][i - 1] = 'DU'
        else:
            trainMovement['Movement'][i - 1] = 'DD'

for i in range(1, len(testIBM), 1):
    if testIBM[i] > testIBM[i - 1]:
        if testApple[i] > testApple[i - 1]:
            testMovement['Movement'][i - 1] = 'UU'
        else:
            testMovement['Movement'][i - 1] = 'UD'
    else:
        if testApple[i] > testApple[i - 1]:
            testMovement['Movement'][i - 1] = 'DU'
        else:
            testMovement['Movement'][i - 1] = 'DD'

trainTransitionMatrix = pd.DataFrame(index=['UU', 'DD', 'UD', 'DU'], columns=['UU', 'DD', 'UD', 'DU'])
trainTransitionMatrix.fillna(0, inplace=True)
for i in range(1, len(trainMovement), 1):
    if trainMovement['Movement'][i - 1] == 'UU':
        if trainMovement['Movement'][i] == 'UU':
            trainTransitionMatrix.iloc[0,0] = trainTransitionMatrix.iloc[0,0] + 1
        elif trainMovement['Movement'][i] =='DD':
            trainTransitionMatrix.iloc[0,1] = trainTransitionMatrix.iloc[0,1] + 1
        elif trainMovement['Movement'][i] == 'UD':
            trainTransitionMatrix.iloc[0,2] = trainTransitionMatrix.iloc[0,2] + 1
        elif trainMovement['Movement'][i] == 'DU':
            trainTransitionMatrix.iloc[0,3] = trainTransitionMatrix.iloc[0,3] + 1
    elif trainMovement['Movement'][i - 1] == 'DD':
        if trainMovement['Movement'][i] == 'UU':
            trainTransitionMatrix.iloc[1,0] = trainTransitionMatrix.iloc[1,0] + 1
        elif trainMovement['Movement'][i] == 'DD':
            trainTransitionMatrix.iloc[1,1] = trainTransitionMatrix.iloc[1,1] + 1
        elif trainMovement['Movement'][i] == 'UD':
            trainTransitionMatrix.iloc[1,2] = trainTransitionMatrix.iloc[1,2] + 1
        elif trainMovement['Movement'][i] == 'DU':
            trainTransitionMatrix.iloc[1,3] = trainTransitionMatrix.iloc[1,3] + 1
    elif trainMovement['Movement'][i - 1] == 'UD':
        if trainMovement['Movement'][i] == 'UU':
            trainTransitionMatrix.iloc[2,0] = trainTransitionMatrix.iloc[2,0] + 1
        elif trainMovement['Movement'][i] == 'DD':
            trainTransitionMatrix.iloc[2,1] = trainTransitionMatrix.iloc[2,1] + 1
        elif trainMovement['Movement'][i] == 'UD':
            trainTransitionMatrix.iloc[2,2] = trainTransitionMatrix.iloc[2,2] + 1
        elif trainMovement['Movement'][i] == 'DU':
            trainTransitionMatrix.iloc[2,3] = trainTransitionMatrix.iloc[2,3] + 1
    elif trainMovement['Movement'][i - 1] == 'DU':
        if trainMovement['Movement'][i] == 'UU':
            trainTransitionMatrix.iloc[3,0] = trainTransitionMatrix.iloc[3,0] + 1
        elif trainMovement['Movement'][i] == 'DD':
            trainTransitionMatrix.iloc[3,1] = trainTransitionMatrix.iloc[3,1] + 1
        elif trainMovement['Movement'][i] == 'UD':
            trainTransitionMatrix.iloc[3,] = trainTransitionMatrix.iloc[3,2] + 1
        elif trainMovement['Movement'][i] == 'DU':
            trainTransitionMatrix.iloc[3,3] = trainTransitionMatrix.iloc[3,3] + 1

trainTransitionMatrix = trainTransitionMatrix/trainTransitionMatrix.values.sum()
print ('80% trainTranstion Matrix :\n', trainTransitionMatrix.round(2))

testTransitionMatrix = pd.DataFrame(index=['UU', 'DD', 'UD', 'DU'], columns=['UU', 'DD', 'UD', 'DU'])
testTransitionMatrix.fillna(0, inplace=True)
for i in range(1, len(testMovement), 1):
    if testMovement['Movement'][i - 1] == 'UU':
        if testMovement['Movement'][i] == 'UU':
            testTransitionMatrix.iloc[0,0] = testTransitionMatrix.iloc[0,0] + 1
        elif testMovement['Movement'][i] =='DD':
            testTransitionMatrix.iloc[0,1] = testTransitionMatrix.iloc[0,1] + 1
        elif testMovement['Movement'][i] == 'UD':
            testTransitionMatrix.iloc[0,2] = testTransitionMatrix.iloc[0,2] + 1
        elif testMovement['Movement'][i] == 'DU':
            testTransitionMatrix.iloc[0,3] = testTransitionMatrix.iloc[0,3] + 1
    elif testMovement['Movement'][i - 1] == 'DD':
        if testMovement['Movement'][i] == 'UU':
            testTransitionMatrix.iloc[1,0] = testTransitionMatrix.iloc[1,0] + 1
        elif testMovement['Movement'][i] == 'DD':
            testTransitionMatrix.iloc[1,1] = testTransitionMatrix.iloc[1,1] + 1
        elif testMovement['Movement'][i] == 'UD':
            testTransitionMatrix.iloc[1,2] = testTransitionMatrix.iloc[1,2] + 1
        elif testMovement['Movement'][i] == 'DU':
            testTransitionMatrix.iloc[1,3] = testTransitionMatrix.iloc[1,3] + 1
    elif testMovement['Movement'][i - 1] == 'UD':
        if testMovement['Movement'][i] == 'UU':
            testTransitionMatrix.iloc[2,0] = testTransitionMatrix.iloc[2,0] + 1
        elif testMovement['Movement'][i] == 'DD':
            testTransitionMatrix.iloc[2,1] = testTransitionMatrix.iloc[2,1] + 1
        elif testMovement['Movement'][i] == 'UD':
            testTransitionMatrix.iloc[2,2] = testTransitionMatrix.iloc[2,2] + 1
        elif testMovement['Movement'][i] == 'DU':
            testTransitionMatrix.iloc[2,3] = testTransitionMatrix.iloc[2,3] + 1
    elif testMovement['Movement'][i - 1] == 'DU':
        if testMovement['Movement'][i] == 'UU':
            testTransitionMatrix.iloc[3,0] = testTransitionMatrix.iloc[3,0] + 1
        elif testMovement['Movement'][i] == 'DD':
            testTransitionMatrix.iloc[3,1] = testTransitionMatrix.iloc[3,1] + 1
        elif testMovement['Movement'][i] == 'UD':
            testTransitionMatrix.iloc[3,] = testTransitionMatrix.iloc[3,2] + 1
        elif testMovement['Movement'][i] == 'DU':
            testTransitionMatrix.iloc[3,3] = testTransitionMatrix.iloc[3,3] + 1

testTransitionMatrix = testTransitionMatrix/testTransitionMatrix.values.sum()
print ('20% testTransition Matrix :\n', testTransitionMatrix.round(2))

FinaltrainTransitionMatrix = pd.DataFrame(index=['UU or DD', 'UD or DU'], columns=['UU or DD', 'UD or DU'])
FinaltrainTransitionMatrix.fillna(0, inplace=True)
FinaltrainTransitionMatrix.iloc[0,0] = trainTransitionMatrix.iloc[0:2,0:2].values.sum()
FinaltrainTransitionMatrix.iloc[0,1] = trainTransitionMatrix.iloc[0:2,2:4].sum()
FinaltrainTransitionMatrix.iloc[1,0] = trainTransitionMatrix.iloc[2:4,0:2]
FinaltrainTransitionMatrix.iloc[1,1] = trainTransitionMatrix.iloc[2:4,2:4]
print ('20% FinaltrainTransition Matrix :\n', FinaltrainTransitionMatrix.round(2))