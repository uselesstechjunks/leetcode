##################################################################################
Fundamentals
##################################################################################

**********************************************************************************
Notation
**********************************************************************************
.. note::
	* All vectors are named for their column vector form.
	* Our observed data points are in :math:`\mathbb{R}^d` for some :math:`d\geq 1`.
	* Our target data points are in :math:`\mathbb{R}^K` for some :math:`K\geq 1`.
	* The data is associated with a random variable :math:`X` which might be a random vector for :math:`d> 1` 

		.. math:: X=(X_1,\cdots,X_d)
	* The target is associated with a random variable :math:`Y` which might also be a random vector

		.. math:: Y=(Y_1,\cdots,Y_K)
	* An observation for :math:`X=x_i\in\mathbb{R}^d` might be taken in the row-vector form, :math:`x_i^T\in\mathbb{R}_{1\times d}`.
	* Single dimensional observations for target are usually written as :math:`Y=y_i\in\mathbb{R}`.

		* For :math:`K> 1`, we can also associate it with the row vector form, :math:`y_i^T\in\mathbb{R}_{1\times K}`.
	* We have a total of :math:`N` observations, and all the observations together are taken in the matrix form

		.. math:: \mathbf{X}_{N\times d}=\begin{bmatrix}-& x_1^T & - \\ \vdots & \vdots & \vdots \\ -& x_N^T & -\end{bmatrix}
	* The vector :math:`\mathbf{x}_j\in\mathbb{R}^N` represents the column vector for all the observations for rv :math:`X_j`.

**********************************************************************************
Statistical Decision Theory
**********************************************************************************
Regression
==================================================================================
.. note::
	* We're interested in finding an estimator for :math:`Y`

		.. math:: \hat{Y}=f(X)
	* Estimation error: :math:`\tilde{Y}=\hat{Y}-Y`
	* Bias: :math:`\mathbb{E}_Y[\tilde{Y}]`
	* Standard error (se): :math:`\sqrt{\mathbb{V}_Y(\hat{Y})}`
	* Mean-squared error (mse): :math:`\mathbb{E}_Y[\tilde{Y}^2]`

.. tip::
	* Assuming that :math:`\hat{Y}` and :math:`Y` are independent, we note that

		.. math:: \mathbb{V}_Y(\tilde{Y})=\mathbb{V}_Y(\hat{Y})+\mathbb{V}_Y(Y)

.. attention::
	* We note that

		.. math:: \text{mse}(\hat{Y})=\mathbb{E}_Y[\tilde{Y}^2]=\mathbb{E}_Y[(\hat{Y}-Y)^2]=\mathbb{V}_Y(\hat{Y}-Y)+\left(\mathbb{E}_Y[\hat{Y}-Y]\right)^2=\mathbb{V}_Y(\hat{Y})+\mathbb{V}_Y(Y)+\left(\mathbb{E}_Y[\hat{Y}-y]\right)^2=\text{se}^2(\hat{Y})+\text{bias}^2(\hat{Y})+\mathbb{V}_Y(Y)
	* If the unknown :math:`Y` is some constant :math:`y` instead of a rv, then :math:`\mathbb{V}_Y(Y)=0` and we have 

		.. math:: \mathbb{V}_Y(\tilde{Y})=\mathbb{V}_Y(\hat{Y})=\text{se}^2
	* In that case

		.. math:: \text{mse}(\hat{Y})=\text{se}^2(\hat{Y})+\text{bias}^2(\hat{Y})

Bayes Estimator
----------------------------------------------------------------------------------
.. note::
	* This is the estimator which minimises mse.

		.. math:: f^*=\underset{f}{\arg\min}\left(\mathbb{E}_Y[(f(X)-Y)^2]\right)=\underset{f}{\arg\min}\left(\mathbb{E}_X\left[\mathbb{E}_{Y|X}[(f(X)-Y)^2]|X\right]\right)
	* [WHY??] This minimisation problem is equivalent to finding a pointwise minimum, such that, for each :math:`X=x`, 

		.. math:: f(x)=\underset{\hat{y}}{\arg\min}\left(\mathbb{E}_X\left[\mathbb{E}_{Y|X}[(\hat{y}-Y)^2]|X=x\right]\right)
	* [WHY??] The solution is :math:`f(x)=\mathbb{E}_{Y|X}[Y|X=x]` which is the conditional expectation estimator or Bayes estimator.
	* We note that this estimator is unbiased.

Approximating The Bayes Estimator
----------------------------------------------------------------------------------
Assuming locally constant nature of the fucntion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
	* In kNN regression approach, we approximate Bayes estimator by 

		* replacing expectation with sample average
		* approximating the point :math:`X=x` with a neighbourhood :math:`N(x)` where :math:`|N(x)|=k`
	* In this case :math:`f(x)=\mathbb{E}_{Y|X}[Y|X=x]\approx\frac{1}{k) \sum_{x_i\in N(x)} x_i`
	* The implicit assumption is that the function behaves locally constant around each point :math:`x` so it can be estimated with the average value of :math:`x_i`

Explicit assumption from a model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
	* In linear regression approach, we explicitly assume that the estimator is linear in :math:`X_j`s.
	* In this case, :math:`f(x)=\mathbb{E}_{Y|X}[Y|X=x]\approx\x^T\beta`

Classification
==================================================================================

Bayes Classifier
----------------------------------------------------------------------------------

**********************************************************************************
Curse of Dimensionality
**********************************************************************************

**********************************************************************************
Statistical Models
**********************************************************************************
Linear Regression
kNN Classification
