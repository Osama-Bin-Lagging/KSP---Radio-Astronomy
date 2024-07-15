import numpy as np
import matplotlib.pyplot as plt
import scipy

mean = []
var = []
skew = []
k = []
for x in range(10):
	plot = np.random.rand(pow(4,x))	
	mean_x = np.mean(plot)
	var_x = np.var(plot)
	skew_x = scipy.stats.skew(plot) #, bias=True)
	k_x = scipy.stats.kurtosis(plot)
	mean.append(mean_x)
	var.append(var_x)
	skew.append(skew_x)
	k.append(k_x)

plt.scatter(np.arange(1, 11), mean, label = 'mean')
plt.scatter(np.arange(1, 11), var, label = 'var')
plt.scatter(np.arange(1, 11), skew, label = 'skew')
plt.scatter(np.arange(1, 11), k, label = 'k')
plt.legend()
plt.show()



