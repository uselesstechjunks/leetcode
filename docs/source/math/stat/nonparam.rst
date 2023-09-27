########################################################################################
Non-Parametric Methods for Estimating Statistical Functionals
########################################################################################

Empirical distribution function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The estimator for :math:`F` is :math:`\hat{F_n}` which assigns a mass :math:`1/n` to every point in sample :math:`\{X_i\}_{i=1}^n`.

.. note::		
	For a given :math:`x`,

		* :math:`\mathbb{E}[\hat{F_n}(x)]=F(x)`
		* :math:`\text{Var}(\hat{F_n})=\frac{F(x)(1-F(x))}{n}`

Plug-in Estimator
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
.. note::
	The plug-in estimator for any statistical functional :math:`T(F)` can be obtained by replacing it with :math:`\hat{F_n}` as :math:`T(\hat{F_n})`.

For the empirical distribution model, following are some interesting results.

.. note::
        * Empirical distribution function is a consistent estimator for any distribution.
    * **Glivenko-Cantelli Theorem**: :math:`||\hat{F_n}(x)-F(x)||_\infty=\sup_{x}|\hat{F_n}(x)-F(x)|\xrightarrow[]{as} 0`.
    * **Dvoretzsky-Kiefer-Wolfowitz (DKW) Inequality**: For any :math:`\epsilon>0`,
    
        .. math::
            \mathbb{P}(\sup_x|\hat{F_n}(x)-F(x)|>\epsilon) \le 2\exp(-2n\epsilon^2)

    * It can be derived from DKW that we can form a :math:`1-\alpha` CI of width :math:`2\epsilon_n` around :math:`\hat{F_n}` where :math:`\epsilon_n=\sqrt{\frac{1}{2n}\ln(\frac{2}{\alpha})}`.
