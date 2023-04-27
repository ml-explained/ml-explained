################
Gradient Descent
################

*Lead Author: Jordan*

The method of Gradient Descent (also known as Steepest Descent) considers only the current gradient when choosing the search direction :math:`\mathbf{p}_k`. 
This method can be thought of as being *memoryless* as it does not utilise the history of search directions taken i.e. :math:`\mathbf{p}_1,...,\mathbf{p}_{k-1}`. 
By differentiating the quadratic loss function, :math:`\mathcal{L}(\mathbf{x}_{k}):=\mathbf{x}_k^\text{T}\mathbf{Ax}_k - \mathbf{x}_k^\text{T}\mathbf{b}`, we 
know the minimum is achieved when :math:`\mathbf{A}\mathbf{x}_{k} = \mathbf{b}`. By using the negative gradient, :math:`\mathbf{r}_k:=\mathbf{b} - \mathbf{Ax}_k`, 
we can choose a sufficiently small learning rate, :math:`\alpha` for the problem to converge. By additionally defining some convergence threshold, 
:math:`\tau>0` and the maximum number of iterations, :math:`K`, the gradient descent algorithm for a linear system is defined below.

.. pcode::
   :linenos:

    \begin{algorithm}
    \caption{Gradient Descent}
    \begin{algorithmic}
    \PROCEDURE{GradientDescent}{$\mathbf{A}, \mathbf{b}, \mathbf{x}_0, \alpha, \tau, K$}
    \STATE $\mathbf{r}_0 = \mathbf{b} - \mathbf{Ax}_0$
    \FOR{$k = 1$ \TO $K$}
        \IF{$||\mathbf{r}_{k-1}||_2 \le \tau$}
            \BREAK
        \ENDIF
        \STATE $\mathbf{p}_k = \mathbf{r}_{k-1}$
        \STATE $\mathbf{x}_k = \mathbf{x}_{k-1} + \alpha\mathbf{p}_k$
        \STATE $\mathbf{r}_k = \mathbf{b} - \mathbf{Ax}_{k}$
    \ENDFOR
    \RETURN $\mathbf{x}_k$
    \ENDPROCEDURE
    \end{algorithmic}
    \end{algorithm}

An issue with the current method is that we have to choose a *sufficiently* small learning rate ourselves which is data dependent. Further, setting it 
too small results in more iterations needed before convergence and setting it too large results in our solution exploding. By the use of some Mathematics, 
we can determine the optimal learning rate at every iteration step.

Determining the Optimal Learning Rate
=====================================

Here we determine the optimal learning rate :math:`\alpha_k` by considering the following functional:

.. math::

	\phi(\mathbf{x}) := \frac{1}{2}\mathbf{x}^\text{T}\mathbf{A}\mathbf{x} - \mathbf{x}^\text{T}\mathbf{b}.

We see that by considering the derivative, the minimisation of :math:`\phi` occurs when :math:`\mathbf{A}\mathbf{x} = \mathbf{b}`. By considering some arbitrary 
:math:`\mathbf{y}_k` we can write :math:`\phi(\mathbf{x})` in terms of :math:`\mathbf{y}_k` and a couple of other terms:

.. math::

    \begin{align}
    \phi(\mathbf{x}_k) &= \phi(\mathbf{y}_k + \mathbf{x}_k - \mathbf{y}_k)\nonumber\\
    &= \frac{1}{2}(\mathbf{y}_k + \mathbf{x}_k - \mathbf{y}_k)^\text{T}\mathbf{A}(\mathbf{y}_k + \mathbf{x}_k - \mathbf{y}_k) - (\mathbf{y}_k + \mathbf{x}_k - \mathbf{y}_k)^\text{T}\mathbf{b}\\
    &= \phi(\mathbf{y}_k) + \frac{1}{2}||\mathbf{x}_k - \mathbf{y}_k||_\mathbf{A}^2 - (\mathbf{x}_k - \mathbf{y}_k)^\text{T}\mathbf{r}(\mathbf{y}_k).\quad(\text{let } \mathbf{y}_k = \mathbf{x}_{k - 1} \text{ and recall that }\mathbf{x}_k = \mathbf{x}_{k - 1} + \alpha_k\mathbf{p}_k)\\\\
    \phi(\mathbf{x}_k) &= \phi(\mathbf{x}_{k-1}) + \frac{\alpha_k^2}{2}||\mathbf{p}_k||_\mathbf{A}^2 - \alpha_k\mathbf{p}_k^\text{T}\mathbf{r}_{k-1}\\
    &= \phi(\mathbf{x}_{k-1}) + \frac{||\mathbf{p}_k||_\mathbf{A}^2}{2}\left(\alpha_k^2 - 2\alpha_k\frac{\mathbf{p}_k^\text{T}\mathbf{r}_{k-1}}{||\mathbf{p}_k||_\mathbf{A}^2}\right)\\
    &= \phi(\mathbf{x}_{k-1}) + \frac{||\mathbf{p}_k||_\mathbf{A}^2}{2}\left(\left(\alpha_k - \frac{\mathbf{p}_k^\text{T}\mathbf{r}_{k-1}}{||\mathbf{p}_k||_\mathbf{A}^2}\right)^2 - \frac{(\mathbf{p}_k^\text{T}\mathbf{r}_{k-1})^2}{||\mathbf{p}_k||_\mathbf{A}^4}\right)\\
    &= \phi(\mathbf{x}_{k-1}) + \frac{||\mathbf{p}_k||_\mathbf{A}^2}{2}\left(\alpha_k - \frac{\mathbf{p}_k^\text{T}\mathbf{r}_{k-1}}{||\mathbf{p}_k||_\mathbf{A}^2}\right)^2 - \frac{(\mathbf{p}_k^\text{T}\mathbf{r}_{k-1})^2}{2||\mathbf{p}_k||_\mathbf{A}^2}
    \end{align}


What this result shows is that regardless of the search direction :math:`\mathbf{p}_k`, we can choose :math:`\alpha_k` to ensure the cost function :math:`\phi` 
is being minimised by setting :math:`\alpha_k = \mathbf{p}_k^\text{T}\mathbf{r}_{k-1}\ /\ ||\mathbf{p}_k||_\mathbf{A}^2`. In the case of gradient descent, we set 
:math:`\mathbf{p}_k = \mathbf{r}_{k-1}`.

.. pcode::
   :linenos:

    \begin{algorithm}
    \caption{Gradient Descent with Optimal $\alpha$}
    \begin{algorithmic}
    \PROCEDURE{GradientDescent}{$\mathbf{A}, \mathbf{b}, \mathbf{x}_0, \tau, K$}
    \STATE $\mathbf{r}_0 = \mathbf{b} - \mathbf{Ax}_0$
    \FOR{$k = 1$ \TO $K$}
        \IF{$||\mathbf{r}_{k-1}||_2 \le \tau$}
            \BREAK
        \ENDIF
        \STATE $\mathbf{p}_k = \mathbf{r}_{k-1}$
        \STATE $\alpha = ||\mathbf{p}_k||_2^2 / ||\mathbf{p}_k||_\mathbf{A}^2$
        \STATE $\mathbf{x}_k = \mathbf{x}_{k-1} + \alpha\mathbf{p}_k$
        \STATE $\mathbf{r}_k = \mathbf{b} - \mathbf{Ax}_{k}$
    \ENDFOR
    \RETURN $\mathbf{x}_k$
    \ENDPROCEDURE
    \end{algorithmic}
    \end{algorithm}

Convergence
===========

Defining the error vector as :math:`\mathbf{e} := \mathbf{x}^* - \mathbf{x}`, the method of gradient descent has the following convergence rate:

.. math::

    ||\mathbf{e}_k(\alpha)||_\mathbf{A} = ||\mathbf{x}^*-\mathbf{x}_k||_\mathbf{A} \leq \left(\frac{K_2(\mathbf{A}) - 1}{K_2(\mathbf{A}) + 1}\right)^k||\mathbf{e}_0||_\mathbf{A} = \left(1 - \frac{2}{K_2(\mathbf{A}) + 1}\right)^k||\mathbf{e}_0||_\mathbf{A},

where :math:`K_2` is the condition number in the 2-norm. This can be shown by considering :math:`\mathbf{x}_k(\alpha) = \mathbf{x}_{k-1} + \alpha\mathbf{r}_{k-1}` as a function of :math:`\alpha\in\mathbb{R}^+`.

.. math::

    \begin{align}
    \mathbf{x}_{k}(\alpha) &= \mathbf{x}_{k-1} + \alpha \mathbf{A}(\mathbf{x}^*-\mathbf{x}),\quad \text{as } \mathbf{r}_{k-1} = \mathbf{b} - \mathbf{Ax}_{k-1} = \mathbf{Ax}^* - \mathbf{Ax}_{k-1},\\
    \Rightarrow \mathbf{x}^* - \mathbf{x}_{k}(\alpha) &= (1 - \alpha\mathbf{A})(\mathbf{x}^*-\mathbf{x}_{k-1}),\\
    \Rightarrow \mathbf{e}_{k} &= (1 - \alpha\mathbf{A})\mathbf{e}_{k-1}\\
    \Rightarrow ||\mathbf{e}_{k}||_\mathbf{A}^2 &= \mathbf{e}_{k-1}^\text{T}(1 - \alpha\mathbf{A})^\text{T}\mathbf{A}(1 - \alpha\mathbf{A}).
    \end{align}

Expanding :math:`\mathbf{e}_k = \sum_{j}^M a_j\mathbf{z}_j` w.r.t. the orthogonal basis of eigenvectors of :math:`\mathbf{A}`, then for some coefficients 
:math:`\{a_j\}_{j=1}^M\subset\mathbb{R}`, we obtain

.. math::

    \begin{align}
    ||\mathbf{e}_{k}(\alpha)||_\mathbf{A}^2 &= \sum_{j=1}^M \lambda_j a_j^2(1 - \alpha\lambda_j)^2\\
    \Rightarrow ||\mathbf{e}_{k}(\hat{\alpha})||_\mathbf{A}^2 &= \sum_{j}^M \lambda_j a_j^2\left(\frac{\lambda_1 + \lambda_M - 2\lambda_j}{\lambda_1 + \lambda_M}\right)^2, \qquad \text{for } \hat{\alpha} = \frac{2}{\lambda_1 + \lambda_M},\\
    &= \sum_{j}^M\lambda_j a_j^2\frac{(\lambda_1 - \lambda_M)^2 - 4(\lambda_1 - \lambda_j)(\lambda_j - \lambda_M)}{(\lambda_1 + \lambda_M)^2}\\
    &\leq \left(\frac{\lambda_1 - \lambda_M}{\lambda_1 + \lambda_M}\right)^2\sum_j^M \lambda_j\alpha_j^2,\\
    &= \left(\frac{\lambda_1 - \lambda_M}{\lambda_1 + \lambda_M}\right)^2||\mathbf{e}_{k-1}||_\mathbf{A}^2,
    \end{align}

where :math:`\lambda_j` is the :math:`j`-th eigenvalue of :math:`\mathbf{A}` with :math:`\lambda_1 \ge \lambda_j \ge \lambda_M`.