########################################################################################
Non-Parametric Methods
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
Plug-in Estimator for Estimating Statistical Functionals
****************************************************************************************
The plug-in estimator :math:`\hat{T}_n(F)` for any :math:`T(F)` can be obtained by replacing :math:`F` with :math:`\hat{F}_n`.

Estimator for mean
========================================================================================
.. note::
	Here :math:`T(F)=\int x\mathop{dF}`. Since :math:`\hat{F}_n` is discrete

		.. math:: \hat{T}_n(F)=T(\hat{F}_n)=\frac{1}{n}\sum_{i=1}^nX_i=\bar{X}

	* :math:`\text{se}_F^2(\hat{T}_n)=\mathbb{V}_F(\hat{T}_n)=\frac{\sigma^2}{n}`.
	* CLT says that this estimator is asymptotically normal.

.. tip::
	* For :math:`\text{se}_F`, it depends on the true distribution :math:`F`.
	* If the true variance :math:`\sigma^2` is not known, it can be estimated as the next step.
	* Let the estimate for :math:`\text{se}_F` be :math:`\hat{\text{se}}_n(\hat{T}_n)`. Assuming asymptotic normality, we can compute confidence interval as

		.. math:: T(\hat{F}_n)\pm z_{\alpha/2}(\hat{\text{se}}_n(\hat{T}_n))

Estimtor for variance
========================================================================================
.. note::
	Here :math:`T(F)=\int (x-\mathbb{E}[X]^2)\mathop{dF}`. Since :math:`\hat{F}_n` is discrete

		.. math::  \hat{T}(F)=T(\hat{F}_n)=\frac{1}{n}\sum_{i=1}^n(X_i-\bar{X})^2=S^2_n

	* TODO: bias of sample variance
	* For sample mean estimator, :math:`\hat{\text{se}}^2_n(\hat{T}_n)=\frac{1}{n^2}\sum_{i=1}^n(X_i-\bar{X})^2`

.. tip::
	We can use similar techniques for estimating any moments of :math:`F`.

Estimator for other functionals
=========================================================================================
The estimator can be obtained similarly.

.. tip::
	* :math:`\text{se}_F` often has to be estimated in order to obtain a confidence interval.
	* As the estimator is also a statistic, the variance can be obtained using the following methodology.

****************************************************************************************
Variance of a Statistic
****************************************************************************************
We're interested in estimating the variance of a statistic :math:`g(X_1,\cdots,X_n)`.

Bootstrap
========================================================================================
.. note::
	* The key idea:

		* Let :math:`Y=g(X_1,\cdots,X_n)`
		* WLLN: :math:`\frac{1}{n}\sum_{i=1}^nY_i\xrightarrow[]{P}\mathbb{E}[Y]`
		* :math:`\frac{1}{n}\sum_{i=1}^nh(Y_i)\xrightarrow[]{P}\mathbb{E}[h(Y)]`
		* :math:`\frac{1}{n}\sum_{i=1}^n(Y_i-\bar{Y})^2=\frac{1}{n}\sum_{i=1}^n Y_i^2-\left(\frac{1}{n}\sum_{i=1}^n Y_i\right)^2\xrightarrow[]{P}\mathbb{E}[Y^2]-(\mathbb{E}[Y])^2=\mathbb{V}(Y)`

.. tip::
	* We can therefore estimate the variance of a statistic by sample variance.

.. note::
	* For :math:`i=1` to :math:`n`:

		* Obtain a sample :math:`X^*=(X^*_1,\cdots,X^*_2)` by drawing **with replacement** from a given sample :math:`X=(X_1,\cdots,X_n)`.
	* Compute sample variance as the estimator.

.. tip::
	* :math:`\mathbb{V}_F(\hat{T}_n)\approx\mathbb{V}_{\hat{F}_n}(\hat{T}_n)\approx v_{\text{boot}}`
	* We can use :math:`v_{\text{boot}}` to obtain :math:`\text{se}` and compute CI.

Jack knife
========================================================================================
