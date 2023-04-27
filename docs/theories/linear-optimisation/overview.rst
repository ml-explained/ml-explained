###################
Linear Optimisation
###################

*Lead Author: Jordan*

Linear optimisation is the fundamental crux serving as a basis for most modern optimisation techniques, even for non-linear problems. In this section, I focus on 
the well known problem

.. math::
    \begin{equation}
        \mathcal{L}(\mathbf{w}) := ||\mathbf{y} - \mathbf{Xw}||_2^2,
    \end{equation}

which most commonly presents itself in Linear Regression where :math:`\mathbf{y}\in\mathbb{R}^N` is the target response vector, :math:`\mathbf{X}\in\mathbb{R}^{N,M}` 
is our observation or design matrix, and :math:`\mathbf{w}\in\mathbb{R}^{M}` are our linear coefficients of interest. By differentiating and setting to 0, the solution 
can be analytically computed by solving 

.. math::
    \begin{equation}
        \mathbf{X}^\text{T}\mathbf{Xw} = \mathbf{X}^\text{T}\mathbf{y},
    \end{equation}

for :math:`\mathbf{w}`. This works well when our linear system of equations are *over-determined* i.e. :math:`N \gg M` but for the case when we have a linear system of 
equations that is severely *under-determined*, it may be more efficient to solve 

.. math::
    \begin{equation}
        \mathbf{XX}^\text{T}\mathbf{v} = \mathbf{y},
    \end{equation}

for :math:`\mathbf{v}` and then compute :math:`\mathbf{w} = \mathbf{X}^\text{T}\mathbf{v}`. Either way, we are looking to solve a system of linear equations of the 
form

.. math::
    \begin{equation}
        \mathbf{Ax} = \mathbf{b},
    \end{equation}

for :math:`\mathbf{x}` where :math:`\mathbf{A}` is symmetric.

.. note::

    Change of variables to match common notation in Mathematical literature.

When we have a large system of linear equations, we prefer to not directly invert :math:`\mathbf{A}` and instead, resort to an iterative method and wish to minimise 

.. math::
    :label: eq-quadratic

    \phi(\mathbf{x}) := \frac{1}{2}\mathbf{x}^\text{T}\mathbf{Ax} - \mathbf{x}^\text{T}\mathbf{b}.

By differentiating :eq:`eq-quadratic` and setting to 0, we can show that the minimimum is achieved when we have found the solution to our original problem.
We, again, constrain ourselves against computing this quantity using the analytic solution (as it may be computationally expensive), and instead seek an iterative method that 
updates our :math:`k`-th guess of the solution. These class of methods are commonly known as *linear gradient descent* methods. For generality, they all have 
the following update rule:

.. math::
    \mathbf{x}_k = \mathbf{x}_{k-1} + \alpha_k\mathbf{p}_k

where :math:`\alpha_k` is known as the *learning rate* and :math:`\mathbf{p}_k` is the search direction. Below we state desirable properties we would like 
:math:`\mathbf{x}_k` to have:

+ **Existence.** 
    There exists some quantity :math:`\mathbf{x}_k` such that :math:`||\mathbf{x}_k||_2^2 < \infty` i.e. we do not want a
    solution that explodes. Why this is a good property to have becomes clearer when we consider equations in a physical context where :math:`\mathbf{x}_k` may 
    be bounded between reasonably sized numbers.

+ **Uniqueness.** 
    Given :math:`\mathbf{x}_0` is an arbitrary initialisation, for any :math:`\mathbf{x}_0`, :math:`\mathbf{x}_k \rightarrow \mathbf{x}_*` as 
    :math:`k \rightarrow \infty` for some :math:`\mathbf{x}_*`.

+ **Stability.** 
    Given an observation matrix :math:`\mathbf{A}` and some response vector :math:`\mathbf{b}`, if we perturb :math:`\mathbf{A}` and / or 
    :math:`\mathbf{b}` by some small noise i.e. use the data :math:`\mathbf{\tilde{A}}` and :math:`\mathbf{\tilde{b}}`, where the tilde (~) notation represents 
    a noisy representation of their respective non-noisy counter-parts, the solution :math:`\mathbf{\tilde{x}}_*` should satisfy 
    :math:`||\mathbf{\tilde{x}}_* - \mathbf{x}_*||_2^2 \le \epsilon` for some reasonably small :math:`\epsilon`.

+ **Monotonicity.** 
    We desire that with every iteration, we are closer to the true underlying solution i.e. 
    :math:`||\mathbf{x}_k - \mathbf{x}_*||_2^2 < ||\mathbf{x}_{k-1} - \mathbf{x}_*||_2^2\ \forall\ k\in\{1, 2, 3,...\}`.

Here we show two different methods starting with the more naive approach.

Gradient Descent (Steepest Descent)
===================================

The naive approach. By setting the search direction as the gradient of :eq:`eq-quadratic`, we can set a sufficiently small learning rate :math:`\alpha` to update 
our :math:`k`-th guess of :math:`\mathbf{x}_k`. By doing this there is no safe guarantee that we have monotonicity throughout the entire optimisation and if we set 
the learning rate too small, it may take far more iterations than required to converge. By using some simple Maths, we can compute the best learning rate at each 
iteration at every step of the minimisation process. See |gradient-descent| for more details.

Conjugate Gradient
==================

This is the most common linear optimiser in the literature with a much faster convergence than the previous methods. See |conjugate-gradient-descent| for more details.


Comparison
==========

For illustration purposes, I have made a simple toy problem where the data and true solution are drawn from a Gaussian to show convergence rates for each of the described 
linear optimisers.

.. figure:: /_static/theories/linear-optimisation/comparison.png
    :scale: 40%
    :align: center


.. literalinclude:: /../scripts/theories/linear-optimisation/comparison.py
    :linenos:
    :lines: 1-6, 9-46

.. toctree::
    :hidden:
    :caption: Linear Optimisation

    gradient-descent.rst
    conjugate-gradient.rst


.. |gradient-descent| raw:: html

    <a href="gradient-descent.html" target="_blank">Gradient Descent</a>

.. |conjugate-gradient-descent| raw:: html

    <a href="conjugate-gradient.html" target="_blank">Conjugate Gradient</a>