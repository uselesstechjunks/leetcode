################################################################################
Sample Questions
################################################################################

********************************************************************************
Theoretical Background
********************************************************************************

Probability, Statistics and Statistical Learning Principles
================================================================================
.. note::
	* You're given a sample from :math:`F_X` of size :math:`N`, and I give you an estimator :math:`\hat{x}`. Write down the MSE expression and break it down into bias and variance terms.
	* Suppose you know the underlying data distribution, :math:`F_X`. What estimator would you choose so that it minimizes MSE?
	* I explain to you the conditional mean estimator for regression. I give you two ways to approximate this by averaging.

		* For every point :math:`x`, check it's :math:`K` neighbours, check their values and average them. The average becomes your estimate.
		* You define some rectangular regions, with some homogeneaty criteria - points that fall within the same region more or less have very similar values.

	  which one of thse would have higher bias than the other? Can you explain the trends in bias vs variance if I allow you to have tiny rectangular regions, vs larger rectangular regions?
	* I explain to you the MAP estimator for conditional density for classification. Say, you have sample from two joint distributions and you want to build a MAP estimate classifier. I tell you to model the densities as Gaussian. Can you explain how do you come up with the classification rule? If those Gaussians share their covariance, does that simplify things?
	* I give you a system where you can have trees only upto 10 nodes. But you have the option to get multiple of them running in parallel. Can you use this system to do better than individual ones? What type of error would your approach reduce?

Classical ML
================================================================================

Deep Learning
================================================================================

********************************************************************************
Designing ML Systems
********************************************************************************
