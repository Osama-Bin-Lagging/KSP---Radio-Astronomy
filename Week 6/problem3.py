#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

def p(x):
	return np.where((3 < x) & (x < 7), 1 / (7 - 3), 0)

def q(x, x_prime):
        mean = x
        var = 1
        return (1/np.sqrt(2 * np.pi * var)) * np.exp(-(x_prime - mean)**2 / (2 * var))

x = 5
samples = []
steps = 10000

for _ in range(steps):
        x_prime = np.random.normal(loc = x, scale = 1)
        acceptance_ratio = (p(x_prime) * q(x_prime, x)) / (p(x) * q(x, x_prime))
        if np.random.rand() < acceptance_ratio:
                x = x_prime

        samples.append(x)

samples = np.array(samples)
plt.hist(samples, bins = 100, label = 'Sampled Density', density=True)

x_values = np.linspace(1, 9, 1000)
plt.plot(x_values, p(x_values), 'r-', lw=2, label='True Density')

plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.title('Metropolis-Hastings MCMC Sampling')
plt.show()
