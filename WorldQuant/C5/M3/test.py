prices = [2,2,3,]
cva_estimates = [4,2,4]

temp = [None]*3
for i in range (1,51) :
    temp[i-1] = prices[i-1] - cva_estimates[i-1]