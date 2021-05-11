# Let's start by putting the following (completely fictional) share price lists into NumPy arrays

microsoft = [120.3, 121.7, 119.8, 122.5, 123.3]
apple = [124.1, 124.3, 124.2, 124.8, 124.3]
facebook = [110.2, 112.4, 116.8, 115.3, 119.6]

# TASK 1: Import NumPy and create arrays
import numpy as np
microsoft_np = np.array(microsoft)
apple_np = np.array(apple)
facebook_np = np.array(facebook)
print(microsoft, type(microsoft) , microsoft_np, type(microsoft_np))

# Now let's use some NumPy functions to learn some interesting statistical information


# TASK 2A: Calculate the mean share price for each company, and store in suitably named variables.
#          Then print them out (in order - Microsoft, then Apple, then Facebook).
microsoft_mean_price = np.mean(microsoft_np)
apple_mean_price = np.mean(apple_np)
facebook_mean_price = np.mean(facebook_np)
print("Mean values:", microsoft_mean_price , apple_mean_price,facebook_mean_price)

# TASK 2B: Calculate the standard deviation of share prices for each company using a NumPy function,
#          and store in suitably named variables. Then print them out (in order - Microsoft, then Apple, then Facebook).
microsoft_std = np.std(microsoft_np)
apple_std = np.std(apple_np)
facebook_std = np.std(facebook_np)
print("Std. deviation values:", microsoft_std , apple_std,facebook_std)

# TASK 2C: Calculate the percentage change in share price from the first price to the last price, and store
#          in suitably named variables. Then print them out (in order - Microsoft, then Apple, then Facebook).
microsoft_percent_change = ((microsoft_np[np.size(microsoft_np)-1]/microsoft_np[0])-1)*100
apple_percent_change = ((apple_np[np.size(apple_np)-1]/apple_np[0])-1)*100
facebook_percent_change = ((facebook_np[np.size(facebook_np)-1]/facebook_np[0])-1)*100
print("% change values:", microsoft_percent_change , apple_percent_change,facebook_percent_change)

# Now we're going to load our NumPy arrays into a Pandas Dataframe
# The 5 values in each company's array is the share price for the first 5 months of 2018

# TASK 3: Import Pandas and load the arrays into a dataframe with appropriate row and column labels, and then
#         output the dataframe by simply typing its variable name on a new line
import pandas as pd
prices = np.array([[120.3, 121.7, 119.8, 122.5, 123.3],[124.1, 124.3, 124.2, 124.8, 124.3],[110.2, 112.4, 116.8, 115.3, 119.6]])
names = ["Microsoft","Apple","Facebook"]
dates  = ["Jan-2018","Feb-2018","Mar-2018","Apr-2018","May-2018"]
share_price = pd.DataFrame(prices,index=names,columns=dates)
share_price

# Finally, let's draw some interesting graphs!
# You will need to use some of the statistical values you calculated earlier in order to do this

# TO DO: Import Matplotlib's Pyplot submodule

# TASK 4A: Draw a line graph for one of the companies, showing their share prices over time, using the dataframe.
#          Provide suitable labels. Hint: using your_dataframe_variable.loc["your_row_label", :] will extract a single row.
import matplotlib.pyplot as plt
share_Price = share_price.loc["Microsoft",:]
plt.xlabel("Time")
plt.ylabel("Microsoft Share Price")
plt.plot(share_Price)
plt.show()
# TASK 4B: Draw a bar chart, using any data structure, comparing each company's value for any statistic of your choice
#          from Task 2 (A, B or C)
import matplotlib.pyplot as plt
mean_Price = [microsoft_mean_price,apple_mean_price,facebook_mean_price]
names = ["Microsoft","Apple","Facebook"]
plt.xlabel("Company")
plt.ylabel("Mean Price")
plt.bar(names,mean_Price,alpha = 0.5)
plt.show()