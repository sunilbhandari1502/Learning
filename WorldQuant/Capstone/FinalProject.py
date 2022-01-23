import pandas as pd
from yahoofinancials import YahooFinancials
import matplotlib.pyplot as plt

# Getting S&P 500 data
yahoo_financials = YahooFinancials('^GSPC')
data = yahoo_financials.get_historical_price_data(start_date='1990-01-01',
                                                  end_date='2021-12-31',
                                                  time_interval='Daily')
sp500_df = pd.DataFrame(data['^GSPC']['prices'])
sp500_df = sp500_df.drop('date', axis=1).set_index('formatted_date')
sp500_df.head()

# Getting VXX data
yahoo_financials = YahooFinancials('VXX')
data = yahoo_financials.get_historical_price_data(start_date='1990-01-01',
                                                  end_date='2021-12-31',
                                                  time_interval='Daily')
VXX_df = pd.DataFrame(data['VXX']['prices'])
VXX_df = VXX_df.drop('date', axis=1).set_index('formatted_date')
VXX_df.head()

start_date = '2018-01-25'
end_date = '2021-12-31'

# Strategy 1 S&P 500
PortfolioStartValue = 100
VxxAllocation = 0
strategyLevel1 = pd.DataFrame(columns=['Date', 'Level'])

# Strategy 1 Level calculation
for date in VXX_df.index:
    if date == start_date:
        SpUnit = PortfolioStartValue * (1 - VxxAllocation) / (sp500_df["close"][sp500_df.index == date])[0]
        VxxUnit = PortfolioStartValue * VxxAllocation / (VXX_df["close"][VXX_df.index == date])[0]
        strategyLevel1 = strategyLevel1.append({'Date': date, 'Level': PortfolioStartValue}, ignore_index=True)
    else:
        PortfolioValue = SpUnit * (sp500_df["close"][sp500_df.index == date])[0] + VxxUnit * \
                         (VXX_df["close"][VXX_df.index == date])[0]
        strategyLevel1 = strategyLevel1.append({'Date': date, 'Level': PortfolioValue}, ignore_index=True)

strategyLevel1

# Strategy 2 S&P 500 90% VXX 10%
PortfolioStartValue = 100
VxxAllocation = 0.1
strategyLevel2 = pd.DataFrame(columns=['Date', 'Level'])

# Strategy 2 Level calculation
for date in VXX_df.index:
    if date == start_date:
        SpUnit = PortfolioStartValue * (1 - VxxAllocation) / (sp500_df["close"][sp500_df.index == date])[0]
        VxxUnit = PortfolioStartValue * VxxAllocation / (VXX_df["close"][VXX_df.index == date])[0]
        strategyLevel2 = strategyLevel2.append({'Date': date, 'Level': PortfolioStartValue}, ignore_index=True)
    else:
        PortfolioValue = SpUnit * (sp500_df["close"][sp500_df.index == date])[0] + VxxUnit * \
                         (VXX_df["close"][VXX_df.index == date])[0]
        strategyLevel2 = strategyLevel2.append({'Date': date, 'Level': PortfolioValue}, ignore_index=True)

strategyLevel2

# Strategy 3 S&P 500 90% VXX 25%
PortfolioStartValue = 100
VxxAllocation = 0.25
strategyLevel3 = pd.DataFrame(columns=['Date', 'Level'])

# Strategy 3 Level calculation
for date in VXX_df.index:
    if date == start_date:
        SpUnit = PortfolioStartValue * (1 - VxxAllocation) / (sp500_df["close"][sp500_df.index == date])[0]
        VxxUnit = PortfolioStartValue * VxxAllocation / (VXX_df["close"][VXX_df.index == date])[0]
        strategyLevel3 = strategyLevel3.append({'Date': date, 'Level': PortfolioStartValue}, ignore_index=True)
    else:
        PortfolioValue = SpUnit * (sp500_df["close"][sp500_df.index == date])[0] + VxxUnit * \
                         (VXX_df["close"][VXX_df.index == date])[0]
        strategyLevel3 = strategyLevel3.append({'Date': date, 'Level': PortfolioValue}, ignore_index=True)

strategyLevel3

# Plotting performance of different strategy
ax = plt.gca()
# line plot for Strategy 1
strategyLevel1.plot(kind='line', x='Date', y='Level', color='green', ax=ax)
# line plot for Strategy 2
strategyLevel2.plot(kind='line', x='Date', y='Level', color='blue', ax=ax)
# line plot for Strategy 3
strategyLevel3.plot(kind='line', x='Date', y='Level', color='red', ax=ax)
# set the title
plt.title('Performance comparison')
# show the plot
plt.show()
