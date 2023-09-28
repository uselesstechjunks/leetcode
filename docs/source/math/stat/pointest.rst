################################################################################
Parametric Point Estimation
################################################################################

********************************************************************************
Classical Infernece
********************************************************************************

Method of Moments Estimator (MOM)
================================================================================
.. note::
	* Let the parameter vector be :math:`\boldsymbol{\theta}=(\theta_1,\cdots,\theta_k)`.
	* Let the estimator be :math:`\widehat{\Theta}_n=(\widehat{\theta_1},\cdots,\widehat{\theta_k})`.
	* Let :math:`\alpha_j=\alpha_j({\boldsymbol{\theta}})=\mathbb{E}_{\boldsymbol{\theta}}[X^j]=\int x^j\mathop{dF_{\boldsymbol{\theta}}}(x)` for :math:`1\leq j\leq k`.
	* We assume that the moments exist and can be expressed in closed form as equations involving the parameters :math:`\theta_j`.
	* Estimate moments with sample moments as

		.. math:: \widehat{\alpha_j}({\boldsymbol{\theta}})=\alpha(\widehat{\Theta}_n)=\frac{1}{n}\sum_{i=1}^n X_i^j
	* We'd have a system of equations k equations with k unknowns involving :math:`(\widehat{\theta}_j)_{j=1}^k`.

Properties
--------------------------------------------------------------------------------
.. seealso::
	* **Consistent**: :math:`\widehat{\Theta}_n\xrightarrow[]{P}\boldsymbol{\theta}`
	* **Asymptotically normal**:

		* TODO write expression

Common Estimators
--------------------------------------------------------------------------------

Bernoulli
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
	* We have samples :math:`X=(X_1,\cdots,X_n)` from a Bernoulli with unknown :math:`p`.
	* :math:`\widehat{\alpha_0}=p=\frac{1}{n}\sum_{i=1}^n X_i`.

Normal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
	* We have samples :math:`X=(X_1,\cdots,X_n)` from a Normal with unknown :math:`\mu,\sigma`.
	* :math:`\widehat{\alpha_0}=\mu=\frac{1}{n}\sum_{i=1}^n X_i`.
	* :math:`\widehat{\alpha_1}=\mu^2+\sigma^2=\frac{1}{n}\sum_{i=1}^n X^2_i`.

Maximum Likelihood Estimator (MLE)
================================================================================

Likelihood function
--------------------------------------------------------------------------------
.. note::
	* We assume that we have samples of size :math:`n`, :math:`X=(X_1,\cdots,X_n)` such that :math:`X_i\sim f_{X_i}(x_i; \theta)`.
	* Underlying probability model: :math:`f_X(x; \theta)=f_{X_1,\cdots,X_n}(x_1,\cdots,x_n;\theta)`.
	* Independence assumption:

		.. math:: \mathcal{L}(\theta)=f_{X_1,\cdots,X_n}(x_1,\cdots,x_n;\theta)=\prod_{i=1}^n f_{X_i}(x_i;\theta)	

.. warning::
	* Given a particular observation :math:`X=x=(x_1,\cdots,x_n)`, the function :math:`f_X(x; \theta)` is no longer a density, but just a function of :math:`\theta`.
	* For discrete case, :math:`p_X(x; \theta)=\mathbb{P}(X_1=x_1,\cdots,X_n=x_n;\theta)`.

		* This is the probability that the observation would match current data under a particular :math:`\theta`.
		* If this probability is higher under :math:`\theta_1` compared to :math:\theta_2`, it is more likely that the underlying parameter is :math:`\theta_2`.

.. note::
	We estimate :math:`\widehat{\Theta}_n=\mathop{\underset{\theta}{\mathrm{argmax}}}\mathcal{L}(\theta)`.
	
Log likelihood
--------------------------------------------------------------------------------
	* Identical distribution assumption: 

		.. math:: \mathcal{L}(\theta)=f_{X_1,\cdots,X_n}(x_1,\cdots,x_n;\theta)=\prod_{i=1}^n f_X(x_i;\theta)
	* Log likelihood is defined as

		.. math:: \mathcal{l}(\theta)=\log{\mathcal{L}(\theta)}=\sum_{i=1}^n \log(f_X(x_i;\theta))
	* As log is a monotonic increasing function

		.. math:: \mathop{\underset{\theta}{\mathrm{argmax}}}\mathcal{l}(\theta)=\mathop{\underset{\theta}{\mathrm{argmax}}}\mathcal{L}(\theta)

Properties
--------------------------------------------------------------------------------
.. note::
	* **Consistent**: :math:`\widehat{\Theta}_{\text{ML}}\xrightarrow[]{P}\theta`.
	* **Equivariant**: If :math:`\widehat{\Theta}_{\text{ML}}` is the MLE for :math:`\theta`, then :math:`g(\widehat{\Theta}_{\text{ML}})` is the MLE for :math:`g(\theta)`.
	* **Asymptotically normal**: :math:`\frac{\widehat{\Theta}_{\text{ML}}-\theta}{\widehat{\text{se}}(\widehat{\Theta}_{\text{ML}})}\xrightarrow[]{D}\mathcal{N}(0,1)`
	* **Asymptotically optimal**: Estimator has least variance for large sample size.

Computing CI for MLE
--------------------------------------------------------------------------------

Common Estimators
--------------------------------------------------------------------------------

Bernoulli
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Uniform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Binomial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Geometric
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multinomial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Exponential
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Normal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Iterative Method of Computation
--------------------------------------------------------------------------------

Newton Raphson
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The EM Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

********************************************************************************
Bayesian Inference
********************************************************************************

Maximum A Posterior Estimator (MAP)
================================================================================

Common Estimators
--------------------------------------------------------------------------------

Bernoulli
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Normal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Minimum Mean Squared Error Estimator (MMSE)
================================================================================
