import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt

np.random.seed(0)

mcos_estimates = [None]*50
mcos_std = [None]*50

for i in range(1,51):
    unif_array = uniform.rvs(size = i*1000)*2
    cos_val = np.cos(unif_array)*2
    mcos_estimates[i-1] = np.mean(cos_val)
    mcos_std[i-1] = np.std(cos_val)/np.sqrt(i*1000)


plt.plot([np.sin(2)]*50)
plt.plot(mcos_estimates ,'.')
plt.plot(np.sin(2)+np.array(mcos_std)*3)
plt.plot(np.sin(2)-np.array(mcos_std)*3)
plt.xlabel("Sample Size")
plt.ylabel("Value")
plt.show()