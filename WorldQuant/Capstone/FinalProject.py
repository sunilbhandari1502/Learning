import pandas as pd
from yahoofinancials import YahooFinancials
import matplotlib.pyplot as plt
import quantstats as qs

# Getting S&P 500 data
yahoo_financials = YahooFinancials('^GSPC')
data = yahoo_financials.get_historical_price_data(start_date='1990-01-01',
                                                  end_date='2022-02-04',
                                                  time_interval='Daily')
sp500_df = pd.DataFrame(data['^GSPC']['prices'])
sp500_df = sp500_df.drop('date', axis=1).set_index('formatted_date')
sp500_df.head()

yahoo_financials = YahooFinancials('^VIX')
data = yahoo_financials.get_historical_price_data(start_date='1990-01-01',
                                                  end_date='2022-02-04',
                                                  time_interval='Daily')
VIX_df = pd.DataFrame(data['^VIX']['prices'])
VIX_df = VIX_df.drop('date', axis=1).set_index('formatted_date')
VIX_df.head()

# Getting VXX data
yahoo_financials = YahooFinancials('VXX')
data = yahoo_financials.get_historical_price_data(start_date='1990-01-01',
                                                  end_date='2022-02-04',
                                                  time_interval='Daily')
VXX_df = pd.DataFrame(data['VXX']['prices'])
VXX_df = VXX_df.drop('date', axis=1).set_index('formatted_date')
VXX_df.head()

# Plotting Vix Historical level
ax = plt.gca()
# line plot for Strategy 1
# VIX_df.reset_index().plot(kind='scatter', x='formatted_date', y='close', color='blue', ax=ax, s = 10)
# # set the title
# plt.title('Vix Scatter')
# # show the plot
# plt.show()

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

# Startegy 1 Analystics
qs.extend_pandas()
# fetch the daily returns for a stock
returns = strategyLevel1.set_index('Date').pct_change()
stock = pd.Series( returns.Level , index = pd.to_datetime(strategyLevel1['Date'], infer_datetime_format=True))
# show snapshot
# qs.plots.snapshot(stock, title='100% Long S&P 500')
qs.reports.full(stock, "^GSPC")



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

# Startegy 2 Analystics
qs.extend_pandas()
# fetch the daily returns for a stock
returns = strategyLevel2.set_index('Date').pct_change()
stock = pd.Series( returns.Level , index = pd.to_datetime(strategyLevel2['Date'], infer_datetime_format=True))
# show snapshot
# qs.plots.snapshot(stock, title='100% Long S&P 500')
qs.reports.full(stock, "^GSPC")

# Strategy 3 S&P 500 75% VXX 25%
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

# Startegy 3 Analystics
qs.extend_pandas()
# fetch the daily returns for a stock
returns = strategyLevel3.set_index('Date').pct_change()
stock = pd.Series( returns.Level , index = pd.to_datetime(strategyLevel3['Date'], infer_datetime_format=True))
# show snapshot
# qs.plots.snapshot(stock, title='100% Long S&P 500')
qs.reports.full(stock, "^GSPC")

# Strategy 4 S&P 500 80% VXX 20% Short VIX
PortfolioStartValue = 100
VxxAllocation = 0.10
strategyLevel4 = pd.DataFrame(columns=['Date', 'Level'])

# Strategy 4 Level calculation
for date in VXX_df.index:
    if date == start_date:
        VxxAllocation = VxxAllocation * -1;
        SpUnit = PortfolioStartValue * (1 - VxxAllocation) / (sp500_df["close"][sp500_df.index == date])[0]
        VxxUnit = PortfolioStartValue * VxxAllocation / (VXX_df["close"][VXX_df.index == date])[0]
        strategyLevel4 = strategyLevel4.append({'Date': date, 'Level': PortfolioStartValue}, ignore_index=True)
    else:
        PortfolioValue = SpUnit * (sp500_df["close"][sp500_df.index == date])[0] + VxxUnit * \
                         (VXX_df["close"][VXX_df.index == date])[0]
        strategyLevel4 = strategyLevel4.append({'Date': date, 'Level': PortfolioValue}, ignore_index=True)

strategyLevel4

# Startegy 4 Analystics
qs.extend_pandas()
# fetch the daily returns for a stock
returns = strategyLevel4.set_index('Date').pct_change()
stock = pd.Series( returns.Level , index = pd.to_datetime(strategyLevel4['Date'], infer_datetime_format=True))
# show snapshot
# qs.plots.snapshot(stock, title='100% Long S&P 500')
qs.reports.full(stock, "^GSPC")

# Strategy 5 S&P 500 80% VXX 20% Short VIX
PortfolioStartValue = 100
VxxAllocation = 0.25
strategyLevel5 = pd.DataFrame(columns=['Date', 'Level'])

# Strategy 5 Level calculation
for date in VXX_df.index:
    if date == start_date:
        VxxAllocation = VxxAllocation * -1;
        SpUnit = PortfolioStartValue * (1 - VxxAllocation) / (sp500_df["close"][sp500_df.index == date])[0]
        VxxUnit = PortfolioStartValue * VxxAllocation / (VXX_df["close"][VXX_df.index == date])[0]
        strategyLevel5 = strategyLevel5.append({'Date': date, 'Level': PortfolioStartValue}, ignore_index=True)
    else:
        PortfolioValue = SpUnit * (sp500_df["close"][sp500_df.index == date])[0] + VxxUnit * \
                         (VXX_df["close"][VXX_df.index == date])[0]
        strategyLevel5 = strategyLevel5.append({'Date': date, 'Level': PortfolioValue}, ignore_index=True)

strategyLevel5

# Startegy 5 Analystics
qs.extend_pandas()
# fetch the daily returns for a stock
returns = strategyLevel5.set_index('Date').pct_change()
stock = pd.Series( returns.Level , index = pd.to_datetime(strategyLevel5['Date'], infer_datetime_format=True))
# show snapshot
# qs.plots.snapshot(stock, title='100% Long S&P 500')
qs.reports.full(stock, "^GSPC")

# Strategy 6 S&P 500 80% VXX 20% Long VIX if VIX > VixLong Go Long and after tht if it comes below VixShort go short again
PortfolioStartValue = 100
VxxAllocation = 0.15
VixLong = 35
VixShort = 25
strategyLevel6 = pd.DataFrame(columns=['Date', 'Level'])

# Strategy 6 Level calculation
for date in VXX_df.index:
    if date == start_date:
        if (VIX_df["close"][VIX_df.index == date])[0] < VixLong:
            VxxAllocation = VxxAllocation * -1;
        SpUnit = PortfolioStartValue * (1 - VxxAllocation) / (sp500_df["close"][sp500_df.index == date])[0]
        VxxUnit = PortfolioStartValue * VxxAllocation / (VXX_df["close"][VXX_df.index == date])[0]
        strategyLevel6 = strategyLevel6.append({'Date': date, 'Level': PortfolioStartValue}, ignore_index=True)
    #         print('Date', date, 'Level', PortfolioStartValue,strategyLevel5)
    else:
        if VxxAllocation > 0 and (VIX_df["close"][VIX_df.index == date])[0] < VixShort:
            VxxAllocation = VxxAllocation * -1
            SpUnit = PortfolioValue * (1 - VxxAllocation) / (sp500_df["close"][sp500_df.index == date])[0]
            VxxUnit = PortfolioValue * VxxAllocation / (VXX_df["close"][VXX_df.index == date])[0]
        elif VxxAllocation < 0 and (VIX_df["close"][VIX_df.index == date])[0] > VixLong:
            VxxAllocation = VxxAllocation * -1
            SpUnit = PortfolioValue * (1 - VxxAllocation) / (sp500_df["close"][sp500_df.index == date])[0]
            VxxUnit = PortfolioValue * VxxAllocation / (VXX_df["close"][VXX_df.index == date])[0]

        PortfolioValue = SpUnit * (sp500_df["close"][sp500_df.index == date])[0] + VxxUnit * \
                         (VXX_df["close"][VXX_df.index == date])[0]
        strategyLevel6 = strategyLevel6.append({'Date': date, 'Level': PortfolioValue}, ignore_index=True)
#         print('Date', date, 'Level', PortfolioStartValue,strategyLevel5)

strategyLevel6

# Startegy 6 Analystics
qs.extend_pandas()
# fetch the daily returns for a stock
returns = strategyLevel6.set_index('Date').pct_change()
stock = pd.Series( returns.Level , index = pd.to_datetime(strategyLevel6['Date'], infer_datetime_format=True))
# show snapshot
# qs.plots.snapshot(stock, title='100% Long S&P 500')
qs.reports.full(stock, "^GSPC")

# Plotting performance of different strategy
ax = plt.gca()
# line plot for Strategy 1
strategyLevel1.plot(kind='line', x='Date', y='Level', label = "100% Equity", color='green', ax=ax)
# line plot for Strategy 2
strategyLevel2.plot(kind='line', x='Date', y='Level',label = "10% VVX long Only", color='blue', ax=ax)
# line plot for Strategy 3
strategyLevel3.plot(kind='line', x='Date', y='Level', label = "25% VVX long Only",color='red', ax=ax)
# line plot for Strategy 4
strategyLevel4.plot(kind='line', x='Date', y='Level', label = "10% VVX Short Only",color='brown', ax=ax)
# line plot for Strategy 5
strategyLevel5.plot(kind='line', x='Date', y='Level', label = "25% VXX Short Only", color='violet', ax=ax)
# line plot for Strategy 6
strategyLevel6.plot(kind='line', x='Date', y='Level', label = "15% VXX Long/Short VIX based", color='grey', ax=ax)
# set the title
plt.title('Performance comparison')
# show the plot
plt.show()



#reading VXX data
VXX_Historical = pd.read_excel('E:\\WQMsc\\Course10-CapstoneProject\\Vol&Risk\\Week7\\vix-funds-models-no-formulas.xls')
#Changing datetime column to date
VXX_Historical['formatted_date'] = pd.to_datetime(VXX_Historical['formatted_date']).dt.date
#changing date colum to str
VXX_Historical['formatted_date'] = VXX_Historical['formatted_date'].astype(str)
VXX_Historical.set_index("formatted_date", inplace = True)
VXX_Historical.head()
# Strategy 1 S&P 500
VXX_df = VXX_Historical
start_date = '2004-03-26'
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