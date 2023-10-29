###########################################################################
Linear Methods for Regression
###########################################################################
.. note::
	* For a regresion problem, we assume that the target rv :math:`Y` has a normal distribution which

		* has a mean that can be modeled by a regression function :math:`f(X)`
		* has a unknown variance :math:`\sigma^2`
	* This formulation can be written as 

		* :math:`Y=f(X)+\epsilon` where :math:`\epsilon\sim\mathcal{N}(0,\sigma^2)` or 
		* :math:`Y\sim\mathcal{N}(f(X),\sigma^2)`.

.. attention::
	* We also assume that the observations :math:`Y=y_i` are independent.
	* We compute the MLE estimate of :math:`\hat{f}` from the likelihood function.

		.. math:: L(X;f)=-N\log(\sigma)-\frac{N}{2}\log(2\pi)-\frac{1}{2\sigma^2}\sum_{i=1}^N(y_i-f(x_i))^2
	* This gives the objective function

		.. math:: \hat{f}=\underset{f}{\arg\max}\sum_{i=1}^N(y_i-f(x_i))^2

***************************************************************************
Linear Regression
***************************************************************************
Optimisation: Least Squares
===========================================================================
.. note::
	* In linear regression, we assume that true regression function is an affine transform of the data, :math:`f(X)=X\boldsymbol{\beta}+\beta_0` where :math:`\beta_0\in\mathbb{R}` and :math:`\boldsymbol{\beta}\in\mathbb{R}^d` are unknown constants which need to be estimated.
	* For notational simplification, we introduce a dummy column :math:`\mathbf{x}_0=\mathbf{1}\in\mathbb{R}^N` and define data matrix 

		.. math:: \mathbf{X}=\begin{bmatrix}|&|&\cdots&|\\ \mathbf{x}_0 & \mathbf{x}_1 & \cdots & \mathbf{x}_d \\ |&|&\cdots&|\end{bmatrix},
	* Each individual data point is represented by a row vector :math:`x^T\in\mathbb{R}^{d+1}` with 1 at the first dimension.
	* With this, linear regression is expressed as a linear transform instead of affine, :math:`\mathbf{y}=\mathbf{X}\boldsymbol{\beta}`.
	* We can express the objective as 

		.. math:: R^2(\boldsymbol{\beta})=\sum_{i=1}^N(y_i-x_i^T\beta))^2=||\mathbf{y}-\mathbf{X}\boldsymbol{\beta}||^2=(\mathbf{y}-\mathbf{X}\boldsymbol{\beta})^T(\mathbf{y}-\mathbf{X}\boldsymbol{\beta})
	* In this formulation, :math:`\boldsymbol{\beta}\in\mathbb{R}^{d+1}`.

.. tip::
	* First derivative: :math:`\frac{\partial}{\mathop{\partial\boldsymbol{\beta}}}R^2(\boldsymbol{\beta})=-2\mathbf{X}^T(\mathbf{y}-\mathbf{X}\boldsymbol{\beta})`
	* Second derivative: :math:`\frac{\partial^2}{\mathop{\partial\boldsymbol{\beta}}^2}R^2(\boldsymbol{\beta})=2\mathbf{X}^T\mathbf{X}`.
	* If we assume that the data matrix :math:`\mathbf{X}` is full rank, then :math:`\mathbf{X}^T\mathbf{X}` is symmetric positive definite and therefore :math:`\frac{\partial^2}{\mathop{\partial\boldsymbol{\beta}}^2}R^2(\boldsymbol{\beta})> 0`.
	* We can find the minima from setting :math:`\frac{\partial}{\mathop{\partial\boldsymbol{\beta}}}R^2(\boldsymbol{\beta})=\mathbf{0}`.
	* The estimate for the linear regresson coefficient is obtained from :math:`\hat{\boldsymbol{\beta}}_N=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}`.

.. seealso::
	* The linear regression estimate for :math:`\mathbf{y}` is given by

		.. math:: \hat{\mathbf{y}}=\mathbf{X}\hat{\boldsymbol{\beta}}_N=\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}

Geometric Interpretation
---------------------------------------------------------------------------
In terms of covariates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
	* The equation :math:`\mathbf{y}=\beta_0+X\boldsymbol{\beta}` defines the equation of a plane in :math:`\mathbb{R}^{d+1}` (:math:`d` for covariates, 1 for :math:`y`) in terms of the covariates :math:`(X_1,\cdots,X_d)` with :math:`\beta_0` as the intercept along :math:`y`.
	* When evaluated with the data, each estimate :math:`\hat{y}=\beta_0+x^T\boldsymbol{\beta}` defines a point on the plane :math:`(x,\hat{y})\in\mathbb{R}^{d+1}`.
	* True value of :math:`y` also defines a point :math:`(x,y)\in\mathbb{R}^{d+1}` which is not necessarily on the plane.
	* The residual is measured as :math:`y-\hat{y}=y-\beta_0-x^T\boldsymbol{\beta}` and the residual vector for the entirety of the data is given as

		.. math:: \mathbf{y}-\hat{\mathbf{y}}=\mathbf{y}-\mathbf{X}\boldsymbol{\beta}

In terms of data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
	* We observe from the optimality condition of the objective that

		.. math:: \mathbf{X}^T(\mathbf{y}-\mathbf{X}\boldsymbol{\beta})=\mathbf{0}
	* This means the residual :math:`\mathbf{y}-\mathbf{X}\boldsymbol{\beta}` is in the nullspace of the transposed data matrix :math:`\mathbf{X}^T`.
	* The estimate, :math:`\mathbf{X}\boldsymbol{\beta}`, on the other hand, is in the column space of :math:`\mathbf{X}`.
	* Therefore, the estimate and residual are orthogonal and the estimate can be thought of as an orthogonal projection onto the column space spanned by the data matrix.

Inference about Beta
---------------------------------------------------------------------------
.. note::
	TODO: variance of the estimate, confidence intervals

Gauss Markov Theorem
---------------------------------------------------------------------------
.. attention::
	Of all competing methods, OLS method for estimating :math:`\boldsymbol{\beta}` has the least variance.

Orthogonalisation for Mutltiple Regression
===========================================================================
.. tip::
	* For any two vectors, :math:`\mathbf{u}` and :math:`\mathbf{v}`, we can measure the projection of :math:`\mathbf{v}` onto the direction of :math:`\mathbf{u}` as 

		.. math:: ||\mathbf{v}||\cos\theta=\frac{\langle\mathbf{u},\mathbf{v}\rangle}{||\mathbf{u}||^2}=\frac{\langle\mathbf{u},\mathbf{v}\rangle}{\langle\mathbf{u},\mathbf{u}\rangle}

.. note::
	* **Multiple Regression** is the case where :math:`d> 1`. For this, we can think of a formulation in an iterative fashion starting from the single variable case.	
	* For the univariate case, from the optimality condition, we have 

		.. math:: \hat{\beta}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}=\frac{\sum_{i=1}^N x_i y_i}{\sum_{j=1}^N x_j x_j}=\frac{\langle\mathbf{x},\mathbf{y}\rangle}{\langle\mathbf{x},\mathbf{x}\rangle}
	* Therefore, :math:`\hat{\beta}` measures the projection of the target :math:`\mathbf{y}` along the line of the feature vector :math:`\mathbf{x}`.
	* The residual :math:`\mathbf{r}=\mathbf{y}-\hat{\beta}\mathbf{x}` is orthogonal to :math:`\mathbf{x}`.

.. attention::
	* We can start off with the first column vector from the data matrix, which is :math:`\mathbf{x}_0=\mathbf{1}`.
	* We can compute :math:`\beta_0=\frac{\langle\mathbf{x}_0,\mathbf{y}\rangle}{\langle\mathbf{x}_0,\mathbf{x}_0\rangle}=\langle\mathbf{1},\mathbf{y}\rangle`

***************************************************************************
Subset Selection Methods
***************************************************************************
TODO

***************************************************************************
Shrinkage Methods
***************************************************************************

Ridge Regression
===========================================================================

LASSO
===========================================================================
