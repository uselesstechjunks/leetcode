################################################################################
Sample Questions
################################################################################

********************************************************************************
Theoretical Background
********************************************************************************

Statistical Learning: Probability, Statistics, Learning Theory
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

Classical ML: Optimisation
================================================================================
.. note::
	* Max-Margin classifiers

		* Constrained convex optimisation - KKT conditions
		* Separable non-separable case.
	* Linear regression - ridge, LASSO.
	* How do you move beyond linearity? Basis expansion. Infinite dimensional expansion using kernels.
	* Explain gradient descent, stochastic gradient descent, co-ordinate descent

		* How does that work for non-convex error surfaces?
		* How do you identify that you're in a local minima?

Deep Learning
================================================================================

********************************************************************************
Designing ML Systems
********************************************************************************

********************************************************************************
Related StackExchanges
********************************************************************************
.. note::
	* `stats.stackexchange <https://stats.stackexchange.com/>`_
	* `datascience.stackexchange <https://datascience.stackexchange.com/>`_
	* `ai.stackexchange <https://ai.stackexchange.com/>`_

********************************************************************************
Sample Questions
********************************************************************************
Feature Engineering
================================================================================
.. note::
	* When do we need to scale features?
	* How to handle categorical features for

		* categories with a small number of possible values
		* categories with a very large number of possible values
		* ordinal categories (an order associated with them)

Mathematics
================================================================================
.. note::
	* Different types of matrix factorizations. 
	* How are eigenvalues related to singular values.

Statistics
================================================================================
.. note::
	* You have 3 features, X, Y, Z. X and Y are correlated, Y and Z are correlated. Should X and Z also be correlated always?

Classical ML
================================================================================
.. note::
	* Regression

		* What are the different ways to measure performance of a linear regression model.
	* Naive Bayes

		* Some zero problem on Naive Bayes
	* Trees

		* Difference between gradient boosting and XGBoost.
