###########################################################################
Linear Methods for Regression
###########################################################################
.. note::
	* For a regresion problem, we assume that the true value of the target :math:`Y` has a normal distribution which

		* has a mean that can be modeled by a regression function :math:`f(X)`
		* has a unknown variance :math:`\sigma^2`
	* This formulation can be written as 

		* :math:`Y=f(X)+\epsilon` where :math:`\epsilon\sim\mathcal{N}(0,\sigma^2)` or 
		* :math:`Y\sim\mathcal{N}(f(X),\sigma^2)`.

.. attention::
	* We also assume that the observations are independent.
	* We compute the MLE estimate of :math:`\hat{f}` from the likelihood function.

		.. math:: L(X;f)=-N\log(\sigma)-\frac{N}{2}\log(2\pi)-\frac{1}{2\sigma^2}\sum_{i=1}^N(y_i-f(x_i))^2
	* This gives the objective function

		.. math:: \hat{f}=\underset{f}{\arg\max}\sum_{i=1}^N(y_i-f(x_i))^2

***************************************************************************
Linear Regression
***************************************************************************
Optimisation
===========================================================================
.. note::
	* In linear regression, we assume that true regression function is an affine transform of the data, :math:`f(X)=X^T\beta+\beta_0` where :math:`\beta_0` and :math:`\beta` are unknown constants which need to be estimated.
	* For notational simplification, we introduce a dummy column :math:`\mathbf{x}_0=\mathbf{1}\in\mathbb{R}^N` and express the objective as 

		.. math:: \sum_{i=1}^N(y_i-x_i^T\beta))^2=||\mathbf{y}-\mathbf{X}\boldsymbol(\beta)||=(\mathbf{y}-\mathbf{X}\boldsymbol(\beta))^T(\mathbf{y}-\mathbf{X}\boldsymbol(\beta))

Orthogonalisation for Mutltiple Regression
===========================================================================

***************************************************************************
Subset Selection Methods
***************************************************************************

***************************************************************************
Shrinkage Methods
***************************************************************************

Ridge Regression
===========================================================================

LASSO
===========================================================================
