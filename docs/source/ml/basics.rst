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

		.. math X=(X_1,\cdots,X_d)
	* The target is associated with a random variable :math:`Y` which might also be a random vector

		.. math:`Y=(Y_1,\cdots,Y_K)
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
	* Mean-squared errr (mse): :math:`\mathbb{E}_Y[\tilde{Y}^2]`

.. tip::
	* We note that 

		.. math:: \mathbb{V}_Y(\tilde{Y})=\mathbb{E}_Y[\tilde{Y}^2]-\left(\mathbb{E}_Y[\tilde{Y}]\right)^2=\mathbb{E}_Y[(\hat{Y}-Y)^2]-\left(\mathbb{E}_Y[(\hat{Y}-Y)]\right)^2=\mathbb{E}_Y[\hat{Y}^2]-2\mathbb{E}_Y[\hat{Y}Y]+\mathbb{E}_Y[Y^2]-(\mathbb{E}_Y[\hat{Y}])^2+2\mathbb{E}_Y[\hat{Y}]\mathbb{E}_Y[Y]-(\mathbb{E}_Y[Y])^2
	* Assuming that :math:`\hat{Y}` and :math:`Y` are independent

		.. math:: \mathbb{V}_Y(\tilde{Y})=\mathbb{V}_Y(\hat{Y})+\mathbb{V}_Y(Y)

.. tip::
	* We note that

		.. math:: \text{mse}(\hat{Y})=\mathbb{E}_Y[(\hat{Y}-Y)^2]=\mathbb{E}_Y[\hat{Y}^2]-2\mathbb{E}_Y[\hat{Y}Y]+\mathbb{E}_Y[Y^2]
	* If the unknown :math:`Y` is some constant :math:`y` instead of a rv, then :math:`\mathbb{V}_Y(Y)=0` and we have 

		.. math:: \mathbb{V}_Y(\tilde{Y})=\mathbb{V}_Y(\hat{Y})=\text{se}^2

Bayes Estimator
----------------------------------------------------------------------------------
.. note::
	* 

Approximating The Bayes Estimator
----------------------------------------------------------------------------------

Classification
==================================================================================
.. note::

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
