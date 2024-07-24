import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define the target bivariate Gaussian density function p(x, y)
mean = [0, 0]
cov = [[2.0, 1.2], [1.2, 2.0]]

def p(x, y):
    return multivariate_normal.pdf([x, y], mean=mean, cov=cov)

# Define the proposal density function q(x', y' | x, y), which is a Gaussian with mean (x, y) and covariance I
def q(current_state, proposed_state):
    proposal_mean = current_state
    proposal_cov = [[1, 0], [0, 1]]  # Identity matrix for independent proposals
    return multivariate_normal.pdf(proposed_state, mean=proposal_mean, cov=proposal_cov)

# Initialize the sampler
current_state = np.array([0.0, 0.0])
samples = []
n_steps = 10000

# Run the Metropolis-Hastings algorithm
for _ in range(n_steps):
    # Propose a new sample from a bivariate Gaussian centered at the current state
    proposed_state = np.random.multivariate_normal(current_state, [[1, 0], [0, 1]])
    
    # Calculate the acceptance ratio
    p_current = p(current_state[0], current_state[1])
    p_proposed = p(proposed_state[0], proposed_state[1])
    q_current_to_proposed = q(current_state, proposed_state)
    q_proposed_to_current = q(proposed_state, current_state)
    
    # Avoid division by zero by checking if p_current is zero
    if p_current == 0:
        acceptance_ratio = 1
    else:
        acceptance_ratio = (p_proposed * q_proposed_to_current) / (p_current * q_current_to_proposed)
    
    # Accept or reject the new sample
    if np.random.rand() < acceptance_ratio:
        current_state = proposed_state
    
    samples.append(current_state)

# Convert samples to numpy array for easier manipulation
samples = np.array(samples)

fig, axes = plt.subplots(2, 2, figsize=(7, 7))

# 2D Histogram with true density contours
x, y = np.mgrid[-5:5:.01, -5:5:.01]
pos = np.dstack((x, y))
rv = multivariate_normal(mean, cov)
axes[1, 0].hist2d(samples[:, 0], samples[:, 1], bins=50, density=True, cmap='viridis')
axes[1, 0].contour(x, y, rv.pdf(pos), colors='black')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')

# Histogram for x
x_vals = np.linspace(-5, 5, 1000)
x_density = multivariate_normal(mean=mean[0], cov=cov[0][0]).pdf(x_vals)
axes[0, 0].hist(samples[:, 0], bins=50, density=True, label='Sampled')
axes[0, 0].plot(x_vals, x_density, lw=2, label='True')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('Density')

# Histogram for y
y_vals = np.linspace(-5, 5, 1000)
y_density = multivariate_normal(mean=mean[1], cov=cov[1][1]).pdf(y_vals)
axes[1, 1].hist(samples[:, 1], bins=50, density=True, orientation='horizontal', label='Sampled')
axes[1, 1].plot(y_density, y_vals, lw=2, label='True')
axes[1, 1].set_xlabel('Density')
axes[1, 1].set_ylabel('y')

# Remove the empty subplot
axes[0, 1].axis('off')

plt.tight_layout()
plt.show()
