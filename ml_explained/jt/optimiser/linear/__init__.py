import numpy as np

def gradient_descent(A, b, x0, alpha = 'optimal', tau = 1e-8, max_iter = 100, track = False, **kwargs):
    """
    Gradient Descent 
    ================

    Gradient Descent Optimiser for a System of Linear Equations.

    See https://ml-explained.readthedocs.io/en/latest/theories/linear-optimisation/gradient-descent.html for more details.

    Parameters
    ----------
        A        : array[M,M]
                   Design matrix expected to be SPD.

        b        : array[M]
                   Response vector.

        x0       : array[M]
                   Initial guess of the solution.

        alpha    : str, float
                   If set to 'optimal', compute the optimal alpha at every iteration, otherwise take the provided alpha constant.

        tau      : float
                   Convergence threshold.

        max_iter : int
                   Maximum number of iterations. If convergence threshold is met, an early break is executed.

        track    : bool
                   Returns the history of solutions instead if set to True.

    Returns
    -------
        x        : array
                   Solution to the System of Linear Equations. If track = True, returns the history of solutions.
    """
    X    = np.zeros((max_iter + 1, len(x0)))
    X[0] = x0
    r    = b - A @ X[0]
    rr   = r @ r
    for i in range(1, max_iter + 1):
        if np.sqrt(rr) < tau:
            break
        a     = rr / (r @ A @ r) if alpha == 'optimal' else alpha
        X[i]  = X[i - 1] + a * r
        r     = b - A @ X[i]
        rr    = r @ r
    return X[:i + 1] if track else X[i]

def conjugate_gradient(A, b, x0, tau = 1e-8, max_iter = 100, restart = 'auto', track = False, **kwargs):
    """
    Conjugate Gradient Descent 
    ==========================

    Conjugate Gradient Descent Optimiser for a System of Linear Equations.

    See https://ml-explained.readthedocs.io/en/latest/theories/linear-optimisation/conjugate-gradient.html for more details.

    Parameters
    ----------
        A        : array[M,M]
                   Design matrix expected to be SPD.

        b        : array[M]
                   Response vector.

        x0       : array[M]
                   Initial guess of the solution.
    
        tau      : float
                   Convergence threshold.

        max_iter : int
                   Maximum number of iterations. If convergence threshold is met, an early break is executed.

        restart  : int, str = "auto"
                   Number of iterations before setting beta = 0. If restart = "auto", it will be replaced with M. Restarts could 
                   slow down convergence but may improve stability if the conjugate gradient method misbehaves.

        track    : bool
                   Returns the history of solutions instead if set to True.

    Returns
    -------
        x        : array
                   Solution to the System of Linear Equations. If track = True, returns the history of solutions.
    """
    if restart == 'auto':
        restart = len(x0)
    X    = np.zeros((max_iter + 1, len(x0)))
    X[0] = x0
    r    = b - A @ X[0]
    p    = r
    rr   = r @ r
    for i in range(1, max_iter + 1):
        if np.sqrt(rr) < tau:
            break
        Ap     = A @ p
        alpha  = rr / (p @ Ap)
        X[i]   = X[i - 1] + alpha * p
        r     -= alpha * Ap
        rr_new = r @ r
        beta   = (rr_new / rr) if i % restart else 0
        p      = r + beta * p
        rr     = rr_new
    return X[:i + 1] if track else X[i]
