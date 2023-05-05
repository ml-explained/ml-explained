##################
Conjugate Gradient
##################

*Lead Author: Jordan*

The method of conjugate gradient utilises past search directions when selecting the next search direction. At the :math:`k`-th iteration not only do we know the current 
gradient  :math:`\mathbf{r}_{k-1}`, we know all the previous gradients  :math:`\{\mathbf{r}_0,...,\mathbf{r}_{k-2}\}`. By utilising this information we can search in 
some orthogonal space and converge much quicker than the method of steepest descent.

The general idea behind this algorithm is: since  :math:`\alpha_k := \mathbf{p}_k^\text{T}\mathbf{r}_{k-1}\ /\ ||\mathbf{p}_k||_\mathbf{A}^2`, we are looking for a 
search direction  :math:`\mathbf{p}_k` where :math:`\mathbf{p}_k \neq \mathbf{r}_{k-1}`, as in the case of steepest descent, and 
:math:`\mathbf{p}_k^\text{T}\mathbf{r}_{k-1} \neq 0`. We want  :math:`\mathbf{p}_k` and  :math:`\mathbf{x}_k` to satisfy the following conditions:


+  :math:`\mathbf{p}_1,...,\mathbf{p}_k` should be linearly independent.

+  :math:`\phi(\mathbf{x}_k) = \min_{\mathbf{x}\in\mathbf{x}_0 + \text{span}\{\mathbf{p}_1,...,\mathbf{p}_k\}}\phi(\mathbf{x})`.

+  :math:`\mathbf{x}_k` can be calculated easily from  :math:`\mathbf{x}_{k-1}`.


Consider the iterative update equation for  :math:`\mathbf{x}_k`:

.. math::

    \begin{align}
    \mathbf{x}_1 &= \mathbf{x}_0 + \alpha_1\mathbf{p}_1\\
    \mathbf{x}_2 &= \mathbf{x}_1 + \alpha_2\mathbf{p}_2 = \mathbf{x}_0 + \alpha_1\mathbf{p}_1 + \alpha_2\mathbf{p}_2\\
    \vdots\ \  &\\
    \mathbf{x}_k &= \mathbf{x}_0 + \mathbf{P}_{k-1}\mathbf{y}_k + \alpha_k\mathbf{p}_k
    \end{align}


where  :math:`\mathbf{P}_{k-1} = [\mathbf{p}_1,...,\mathbf{p}_{k-1}]` with parameters  :math:`\mathbf{y}_k` and  :math:`\alpha_k`. The objective is to determine the 
parameters  :math:`\mathbf{y}_k` and  :math:`\alpha_k`:

.. math::

    \begin{align}
    \phi(\mathbf{x}_k) &= \frac{1}{2}(\mathbf{x}_0 + \mathbf{P}_{k-1}\mathbf{y}_k + \alpha_k\mathbf{p}_k)^\text{T}\mathbf{A}(\mathbf{x}_0 + \mathbf{P}_{k-1}\mathbf{y}_k + \alpha_k\mathbf{p}_k) - (\mathbf{x}_0 + \mathbf{P}_{k-1}\mathbf{y}_k + \alpha_k\mathbf{p}_k)^\text{T}\mathbf{b}\\
    &= \phi(\mathbf{x}_0 + \mathbf{P}_{k-1}\mathbf{y}_k) + \alpha_k\mathbf{p}_k^\text{T}\mathbf{A}(\mathbf{x}_0 + \mathbf{P}_{k-1}\mathbf{y}_k) - \alpha_k\mathbf{p}_k^\text{T}\mathbf{b} + \frac{\alpha_k^2}{2}||\mathbf{p}_k||_\mathbf{A}^2\\
    &= \textcolor{blue}{\phi(\mathbf{x}_0 + \mathbf{P}_{k-1}\mathbf{y}_k)} + \alpha_k\mathbf{p}_k^\text{T}\mathbf{A}\mathbf{P}_{k-1}\mathbf{y}_k + \textcolor{red}{\frac{\alpha_k^2}{2}||\mathbf{p}_k||_\mathbf{A}^2 - \alpha_k\mathbf{p}_k^\text{T}\mathbf{r}_0},\quad \text{as } \mathbf{b} - \mathbf{A}\mathbf{x}_0 = \mathbf{r}_0
    \end{align}


We tried to separate the  :math:`\textcolor{blue}{\mathbf{y}_k}` and  :math:`\textcolor{red}{\alpha}` terms in our calculations but have a mixed middle term. If we 
did not have this mixed term in the middle, we could just minimise over the two variables separately. Hence, we choose  :math:`\mathbf{p}_k` such that:

.. math::

    \mathbf{p}_k^\text{T}\mathbf{A}\mathbf{P}_{k-1} = \mathbf{0},

by employing the Gram-Schmidt orthonormalisation process

.. math::

	\mathbf{p}_k = \mathbf{r}_{k - 1} - \sum_{j = 1}^{k - 1} \frac{\langle\mathbf{r}_{k - 1},\ \mathbf{p}_j\rangle_\mathbf{A}}{||\mathbf{p}_j||_\mathbf{A}^2}\mathbf{p}_j,

and we are left with the following minimisation task:

.. math::

    \min_{\mathbf{x}_k\in\mathbf{x}_0+\text{span}\{\mathbf{p}_1,...,\mathbf{p}_k\}}\phi(\mathbf{x}_k) = \min_{\mathbf{y}}\big(\phi\left(\mathbf{x}_0 + \mathbf{P}_{k-1}\mathbf{y}\right)\big) + \min_{\alpha_k}\left(\frac{\alpha_k^2}{2}||\mathbf{p}_k||_\mathbf{A}^2 - \alpha_k\mathbf{p}_k^\text{T}\mathbf{r}_0\right)


The first minimisation problem is solved by :math:`\mathbf{y} = \mathbf{y}_{k-1}` (this has already been calculated in the previous step) then 
:math:`\mathbf{x}_k=\mathbf{x}_0+\mathbf{P}_{k}\mathbf{y}_k` satisfies

.. math::

    \phi(\mathbf{x}_k) = \min_{\mathbf{x}_0 + \text{span}\{\mathbf{p}_1,...,\mathbf{p}_k\}} \phi(\mathbf{x})


By completing the square, we can optimally compute  :math:`\alpha_k` for any search direction  :math:`\mathbf{p}_k` giving the result 
:math:`\alpha_k=\mathbf{p}_k^\text{T}\mathbf{r}_0/||\mathbf{p}_k||_\mathbf{A}^2=||\mathbf{r}_{k-1}||_2^2/||\mathbf{p}_k||_\mathbf{A}^2`. We can show this by considering

.. math::

    \begin{align}
    \mathbf{p}_k^\text{T}\mathbf{r}_{k-1} &= \mathbf{p}_k^\text{T}(\mathbf{b} - \mathbf{Ax}_{k-1})\\
    &= \mathbf{p}_k^\text{T}(\mathbf{b} - \mathbf{A}(\mathbf{x}_0 + \mathbf{P}_{k-1}\mathbf{y}_{k-1}))\\
    &= \mathbf{p}_k^\text{T}\mathbf{r}_0 - \mathbf{p}_k^\text{T}\mathbf{AP}_{k-1}\mathbf{y}_{k-1}\\
    &= \mathbf{p}_k^\text{T}\mathbf{r}_0
    \end{align}

Further, it can be shown that :math:`\mathbf{r}_i^\text{T}\mathbf{r}_j = 0,\ \forall i\neq j` as :math:`\mathbf{p}_i` and :math:`\mathbf{r}_i` span the same Krylov subspace i.e.
:math:`\mathbf{r}_i` form the orthogonal basis with respect to the standard inner product whilst :math:`\mathbf{p}_i` form the orthogonal basis with respect to the inner product 
induced by :math:`\mathbf{A}`.

.. pcode::
   :linenos:

    \begin{algorithm}
    \caption{Conjugate Gradient}
    \begin{algorithmic}
    \PROCEDURE{ConjugateGradient}{$\mathbf{A}, \mathbf{b}, \mathbf{x}_0, \tau, K$}
    \STATE $\mathbf{r}_0 = \mathbf{b} - \mathbf{Ax}_0$
    \STATE $\mathbf{p}_1 = \mathbf{r}_0$
    \FOR{$k = 1$ \TO $K$}
        \IF{$||\mathbf{r}_{k-1}||_2 \le \tau$}
            \BREAK
        \ENDIF
        \STATE $\alpha_k = ||\mathbf{r}_{k-1}||_2^2 / ||\mathbf{p}_{k}||_\mathbf{A}^2$
        \STATE $\mathbf{x}_k = \mathbf{x}_{k-1} + \alpha_k\mathbf{p}_k$
        \STATE $\mathbf{r}_k = \mathbf{r}_{k-1} - \alpha_k\mathbf{Ap}_{k}$
        \STATE $\beta_k = ||\mathbf{r}_k||_2^2 / ||\mathbf{r}_{k-1}||_2^2$
        \STATE $\mathbf{p}_{k+1} = \mathbf{r}_k + \beta_k\mathbf{p}_{k}$
    \ENDFOR
    \RETURN $\mathbf{x}_k$
    \ENDPROCEDURE
    \end{algorithmic}
    \end{algorithm}

Convergence
===========

The conjugate gradient method has the following convergence rate

.. math::

    ||\mathbf{e}||_\mathbf{A}^2 \leq 2\left(\frac{\sqrt{K_2(\mathbf{A})} - 1}{\sqrt{K_2(\mathbf{A})} + 1}\right)^k||\mathbf{e}_0||_\mathbf{A}^2 = 2\left(1 - \frac{2}{\sqrt{K_2(\mathbf{A})} + 1}\right)^k||\mathbf{e}_0||_\mathbf{A}^2.

.. note::

    The complete derivation is a little complex and lengthy so see the following for more details:

    + Polyak, Boris (1987). *Introduction to Optimization*, p68-74
    + Hackbusch, W. (2016). *Iterative Solution of Large Sparse Systems of Equations* (2nd ed.), p238-243




