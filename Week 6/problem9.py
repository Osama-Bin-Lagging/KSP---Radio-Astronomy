import matplotlib.pyplot as plt
import numpy as np

def p(x):
	mean = 2
	var = 2
	return (1/np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean)**2 / (2 * var))

def q(x, x_prime):
		if x or x_prime == -np.inf:
			return 1
		mean = x
		var = 1
		return (1/np.sqrt(2 * np.pi * var)) * np.exp(-(x_prime - mean)**2 / (2 * var))

x = 0.1
samples = []
steps = 10000

for _ in range(steps):
	log_x_prime = np.random.normal(loc = np.log(x), scale = 1)
	x_prime = np.exp(log_x_prime)
	acceptance_ratio = (p(x_prime) * q(log_x_prime, np.log(x))) / (p(x) * q(np.log(x), log_x_prime))
	if np.random.rand() < acceptance_ratio:
		x = x_prime

	samples.append(x)

samples = np.exp(np.array(samples))
plt.hist(samples, bins=200, density=True, alpha=0.6, color='g', label='Sampled Density')

# Plot the true density function
x_values = np.log(np.linspace(0.01, 8, 1000))
plt.plot(x_values, p(x_values), 'r-', lw=2, label='True Density')

plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.title('Metropolis-Hastings MCMC Sampling in Log Space')
plt.show()
