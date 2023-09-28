################################################################################
Parametric Point Estimation
################################################################################

********************************************************************************
Method of Moments
********************************************************************************
.. note::
	* Let the parameter vector be :math:`\boldsymbol{\theta}=(\theta_1,\cdots,\theta_k)`.
	* Let the estimator be :math:`\widehat{\Theta}_n=(\widehat{\theta_1},\cdots,\widehat{\theta_k})`.
	* Let :math:`\alpha_j=\alpha_j({\boldsymbol{\theta}})=\mathbb{E}_{\boldsymbol{\theta}}[X^j]=\int x^j\mathop{dF_{\boldsymbol{\theta}}}(x)` for :math:`1\leq j\leq k`.
	* We assume that the moments exist and can be expressed in closed form as equations involving the parameters :math:`\theta_j`.
	* Estimate moments with sample moments as

		.. math:: \widehat{\alpha_j}({\boldsymbol{\theta}})=\alpha(\widehat{\Theta}_n)=\frac{1}{n}\sum_{i=1}^n X_i^j
	* We'd have a system of equations k equations with k unknowns involving :math:`(\widehat{\theta}_j)_{j=1}^k`.

Properties
================================================================================
.. seealso::
	* :math:`\widehat{\Theta}_n\xrightarrow[]{P}\boldsymbol{\theta}`
	* The estimate is asymptotically normal.

		* TODO write expression

********************************************************************************
Maximum Likelihood Estimation
********************************************************************************
Likelihood function
================================================================================
Log likelihood
================================================================================

Properties
================================================================================
