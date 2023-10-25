###########################################################################
Linear Methods for Regression
###########################################################################
.. note::
  * For a regresion problem, we assume that the true value of the target :math:`y` has a normal distribution which

    * has a mean that can be modeled by a regression function :math:`f(x)`
    * has a unknown variance :math:`\sigma^2`
  * This formulation can be written as 
  
    * :math:`y=f(x)+\epsilon` where :math:`\epsilon\sim\mathcal{N}(0,\sigma^2)` or 
    * :math:`y\sim\mathcal{N}(f(x),\sigma^2)`.

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
