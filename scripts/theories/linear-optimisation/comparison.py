from   ml_explained.jt.optimiser.linear import gradient_descent, conjugate_gradient

import matplotlib.pyplot as plt
import seaborn           as sns
import numpy             as np

import os

sns.set_theme(style = 'whitegrid')

# number of samples and feature dimension size
N           = 100
M           = 10

# random number generator
rng         = np.random.default_rng(0)

# generate X from a standard Gaussian
X           = rng.normal(size = (N, M))

# generate target weights and response vector
w           = rng.normal(size = M)
y           = X @ w

# define A and b for our optimisers
A           = X.T @ X
b           = X.T @ y
x0          = np.zeros(M)

# compute estimates of the target weight all starting from the 0-vector
hat_gd      = gradient_descent(  A, b, x0, track = True, max_iter = 100, alpha = 1e-2)
hat_sd      = gradient_descent(  A, b, x0, track = True, max_iter = 100, alpha = 'optimal')
hat_cg      = conjugate_gradient(A, b, x0, track = True, max_iter = 100)

# plot the cost function that is being minimised
for hat, label in zip([hat_gd, hat_sd, hat_cg], ['gradient descent ($10^{-2}$)', 'gradient descent (optimal)', 'conjugate gradient']):
    plt.semilogy(np.linalg.norm(hat @ A  - b, ord = 2, axis = 1) ** 2, label = label)

# plot convergence threshold
plt.hlines([1e-16] * 2, 0, 100, 'k', '--', label = 'threshold ($10^{-16}$)', zorder = -1)

# aesthetics
plt.grid(ls = (0, (5, 5)))
plt.legend()
plt.ylabel('$||\mathbf{b} - \mathbf{Ax}||_2^2$')
plt.xlabel('No. Iterations')
plt.tight_layout()

dirname = os.path.dirname(__file__)
split   = dirname.index('scripts')
sub     = dirname[split + 8:]
name    = 'comparison.png'

plt.savefig(f'docs/_static/{sub}/{name}', dpi = 300)
