import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define the target bivariate Gaussian density function p(x, y)

mean = [0, 0]
cov = [[2.0, 1.2], [1.2, 2.0]]

def p(x, y):
    return np.where((3 < x) & (x < 7) & (1 < y) & (y < 9), 1 / (7 - 3) * 1 / (9 - 1), 0)

# Define the proposal density function q(x', y' | x, y), which is a Gaussian with mean (x, y) and covariance I
def q(current_state, proposed_state):
    proposal_mean = current_state
    proposal_cov = [[1, 0], [0, 1]]  # Identity matrix for independent proposals
    return multivariate_normal.pdf(proposed_state, mean=proposal_mean, cov=proposal_cov)

# Initialize the sampler
current_state = np.array([5.0, 5.0])
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

# 2D Histogram with uniform density region
axes[1, 0].hist2d(samples[:, 0], samples[:, 1], bins=50, density=True, cmap='viridis')
axes[1, 0].set_xlim(2, 8)
axes[1, 0].set_ylim(0, 10)
axes[1, 0].axvline(3, color='black', linestyle='--')
axes[1, 0].axvline(7, color='black', linestyle='--')
axes[1, 0].axhline(1, color='black', linestyle='--')
axes[1, 0].axhline(9, color='black', linestyle='--')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')

# Histogram for x with true uniform density
x_vals = np.linspace(2, 8, 1000)
x_density = np.where((x_vals > 3) & (x_vals < 7), 1.0 / (7 - 3), 0)
axes[0, 0].hist(samples[:, 0], bins=50, density=True, color='blue', alpha=0.7, label='Sampled')
axes[0, 0].plot(x_vals, x_density, color='red', lw=2, label='True')
axes[0, 0].set_xlim(2, 8)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('Density')
axes[0, 0].legend()

# Histogram for y with true uniform density
y_vals = np.linspace(0, 10, 1000)
y_density = np.where((y_vals > 1) & (y_vals < 9), 1.0 / (9 - 1), 0)
axes[1, 1].hist(samples[:, 1], bins=50, density=True, color='green', alpha=0.7, orientation='horizontal', label='Sampled')
axes[1, 1].plot(y_density, y_vals, color='red', lw=2, label='True')
axes[1, 1].set_ylim(0, 10)
axes[1, 1].set_xlabel('Density')
axes[1, 1].set_ylabel('y')
axes[1, 1].legend()

# Remove the empty subplot
axes[0, 1].axis('off')

plt.tight_layout()
plt.show()
