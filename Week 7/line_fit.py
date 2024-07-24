import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
true_m = 3.5
true_b = 1.7
x_data = np.linspace(0, 10, 100)
y_data = true_m * x_data + true_b + np.random.normal(0, 1, size=x_data.shape)

# Define the model
def model(x, m, b):
    return m * x + b

# Define the log-likelihood function
def log_likelihood(theta, x, y, sigma):
    m, b = theta
    model_y = model(x, m, b)
    return -0.5 * np.sum(((y - model_y) / sigma)**2 + np.log(2 * np.pi * sigma**2))

# Define the log-prior function
def log_prior(theta):
    m, b = theta
    if 0 < m < 5 and 0 < b < 5:  # Uniform priors
        return 0
    else:
        return -np.inf

# Define the log-posterior function
def log_posterior(theta, x, y, sigma):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, sigma)

# Metropolis-Hastings MCMC sampler
def metropolis_hastings(log_posterior, initial_theta, iterations, step_size, x, y, sigma):
    theta = initial_theta
    samples = []
    log_posterior_current = log_posterior(theta, x, y, sigma)
    
    for i in range(iterations):
        theta_proposal = theta + np.random.normal(0, step_size, size=theta.shape)
        log_posterior_proposal = log_posterior(theta_proposal, x, y, sigma)
        
        if np.random.rand() < np.exp(log_posterior_proposal - log_posterior_current):
            theta = theta_proposal
            log_posterior_current = log_posterior_proposal
        
        samples.append(theta)
    
    return np.array(samples)

# Run the MCMC sampler
initial_theta = np.array([0.5, 0.5])
iterations = 10000
step_size = 0.1
sigma = 1.0
samples = metropolis_hastings(log_posterior, initial_theta, iterations, step_size, x_data, y_data, sigma)

# Extract the samples for each parameter
m_samples = samples[:, 0]
b_samples = samples[:, 1]

# Plot the posterior distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(m_samples, bins=30, density=True, color='b', alpha=0.7)
axes[0].set_xlabel('Slope (m)')
axes[0].set_ylabel('Density')
axes[0].set_title('Posterior of Slope (m)')

axes[1].hist(b_samples, bins=30, density=True, color='r', alpha=0.7)
axes[1].set_xlabel('Intercept (b)')
axes[1].set_ylabel('Density')
axes[1].set_title('Posterior of Intercept (b)')

plt.show()

# Plot the data and the fitted model
plt.scatter(x_data, y_data, label='Data')
x_fit = np.linspace(0, 10, 100)
y_fit = model(x_fit, np.mean(m_samples), np.mean(b_samples))
plt.plot(x_fit, y_fit, color='k', label='Fitted model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Data and Fitted Model')
plt.show()

