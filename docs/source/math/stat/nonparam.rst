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
	* :math:`\mathbb{V}(\hat{F}_n)=\frac{F(x)(1-F(x))}{n}`, and :math:`\lim\limits_{n\to\infty}\text{mse}(\hat{F}_n)=0`.
	* Empirical distribution function is a consistent estimator for any distribution.

		.. math:: \hat{F}_n(x)\xrightarrow[]{P}F(x)

Confidence interval for :math:`\hat{F}_n`
========================================================================================
.. note::        
	* **Glivenko-Cantelli Theorem**: :math:`||\hat{F_n}(x)-F(x)||_\infty=\sup_{x}|\hat{F_n}(x)-F(x)|\xrightarrow[]{as} 0`.
	* **Dvoretzsky-Kiefer-Wolfowitz (DKW) Inequality**: For any :math:`\epsilon>0`,
    
		.. math:: \mathbb{P}(\sup_x|\hat{F_n}(x)-F(x)|>\epsilon) \le 2\exp(-2n\epsilon^2)

.. tip::
	* It can be derived from DKW that we can form a :math:`1-\alpha` CI of width :math:`2\epsilon_n` around :math:`\hat{F_n}` where :math:`\epsilon_n=\sqrt{\frac{1}{2n}\ln(\frac{2}{\alpha})}`.

		* TODO: derive.

****************************************************************************************
Plug-in Estimator
****************************************************************************************
The plug-in estimator for any :math:`T(F)` can be obtained by replacing :math:`F` with :math:`\hat{F}_n`.

Estimator for mean
========================================================================================
.. note::
	Here :math:`T(F)=\mathbb{E}[X]=\int x\mathop{dF}`. Since :math:`\hat{F}_n` is discrete

		.. math:: \hat{T}(F)=T(\hat{F}_n)=\frac{1}{n}\sum_{i=1}^nX_i=\bar{X}

	* :math:`\text{se}_F^2(\hat{T}(F))=\mathbb{V}(\widehat{T(F)})=\frac{\sigma^2}{n}`.
	* :math:`\text{b}_F(\hat{T}(F))=0`.

.. tip::
	If the true variance :math:`\sigma^2` is not known, it can be estimated as the next step.

Estimtor for variance
========================================================================================
.. note::
	Here :math:`T(F)=\mathbb{E}[(X-\mathbb{E}[X]^2)]=\int (x-\mathbb{E}[X]^2)\mathop{dF}`. Since :math:`\hat{F}_n` is discrete

		.. math::  \hat{T}(F)=T(\hat{F}_n)=\frac{1}{n}\sum_{i=1}^n(X_i-\bar{X})^2

	* TODO: bias of sample variance
	* For sample mean estimator, :math:`\hat{\text{se}}_F^2(\hat{T}(F))=\frac{1}{n^2}\sum_{i=1}^n(X_i-\bar{X})^2`
