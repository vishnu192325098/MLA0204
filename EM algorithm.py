import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
X = np.array([5.9, 4.6, 6.2, 4.7, 5.5, 5.0, 5.2, 5.9, 6.0, 6.1])  # example data
np.random.seed(0)
k = 2  
mu = np.random.choice(X, k)
sigma = np.random.random(k) + 0.5
pi = np.ones(k) / k
def e_step(X, mu, sigma, pi):
    gamma = np.zeros((len(X), k))
    for j in range(k):
        gamma[:, j] = pi[j] * norm.pdf(X, mu[j], sigma[j])
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma
def m_step(X, gamma):
    N_k = gamma.sum(axis=0)
    mu = np.sum(gamma * X[:, np.newaxis], axis=0) / N_k
    sigma = np.sqrt(np.sum(gamma * (X[:, np.newaxis] - mu)**2, axis=0) / N_k)
    pi = N_k / len(X)
    return mu, sigma, pi
n_iter = 100
for i in range(n_iter):
    gamma = e_step(X, mu, sigma, pi)
    mu, sigma, pi = m_step(X, gamma)
print("Final Means:", mu)
print("Final Std Deviations:", sigma)
print("Final Mixing Coefficients:", pi)
x_vals = np.linspace(min(X)-1, max(X)+1, 1000)
for j in range(k):
    plt.plot(x_vals, pi[j] * norm.pdf(x_vals, mu[j], sigma[j]), label=f'Component {j+1}')
plt.hist(X, bins=10, density=True, alpha=0.5, label='Data')
plt.legend()
plt.title('EM Algorithm - Gaussian Mixture')
plt.show()
