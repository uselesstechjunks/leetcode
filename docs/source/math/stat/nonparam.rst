########################################################################################
Non-Parametric Methods for Estimating Statistical Functionals
########################################################################################

****************************************************************************************
Empirical distribution function
****************************************************************************************
The estimator for any CDF :math:`F` is the discrete estimator :math:`\hat{F}_n` which assigns a mass :math:`1/n` to every point in sample :math:`\{X_i\}_{i=1}^n`.

.. note::
	Let :math:`I(X_i\leq x)=\begin{cases}1 & \text{if $X_i\leq x$}\\ 0 & \text{otherwise}\end{cases}`. Then

		.. math:: \hat{F}_n(x)=\frac{\sum_{i=1}^n I(X_i\leq x_i)}{n}

.. attention::
	* Unbiased: :math:`\mathbb{E}[\hat{F}_n(x)]=F(x)`

		* :math:`\text{b}_F(\hat{F}_n)=0`.
	* :math:`\mathbb{V}(\hat{F}_n)=\frac{F(x)(1-F(x))}{n}`

		* Therefore, :math:`\lim\limits_{n\to\infty}\text{mse}(\hat{F}_n)=0`.
	* Empirical distribution function is a consistent estimator for any distribution.

		.. math:: \hat{F}_n(x)\xrightarrow[]{P}F(x)

****************************************************************************************
Plug-in Estimator
****************************************************************************************
.. note::
	The plug-in estimator for any statistical functional :math:`T(F)` can be obtained by replacing it with :math:`\hat{F}_n` as :math:`T(\hat{F}_n`)`.

Confidence interval for plug-in estimator
========================================================================================
.. note::        
	* **Glivenko-Cantelli Theorem**: :math:`||\hat{F_n}(x)-F(x)||_\infty=\sup_{x}|\hat{F_n}(x)-F(x)|\xrightarrow[]{as} 0`.
	* **Dvoretzsky-Kiefer-Wolfowitz (DKW) Inequality**: For any :math:`\epsilon>0`,
    
		.. math:: \mathbb{P}(\sup_x|\hat{F_n}(x)-F(x)|>\epsilon) \le 2\exp(-2n\epsilon^2)

	* It can be derived from DKW that we can form a :math:`1-\alpha` CI of width :math:`2\epsilon_n` around :math:`\hat{F_n}` where :math:`\epsilon_n=\sqrt{\frac{1}{2n}\ln(\frac{2}{\alpha})}`.
