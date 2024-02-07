##################################################################################
Fundamentals
##################################################################################

**********************************************************************************
Notation
**********************************************************************************
.. note::
	* All vectors are named for their column vector form. For row-representation, we use the transpose notation.
	* The data is associated with a random variable :math:`X`.

		* It might also be a random vector for some :math:`d> 1`, in which case, individual components can referred to as :math:`X_j` and :math:`X=(X_1,\cdots,X_d)`.
	* For observed data points are instances of the rv, :math:`X=x\in\mathbb{R}^d` for some :math:`d\geq 1`.
	* [Regression] The target quantity is associated with a continuous rv :math:`Y\in\mathbb{R}`. 

		* It might also be a random vector, with :math:`Y=(Y_1,\cdots,Y_K)`, for some :math:`K\geq 1`.
		* Single dimensional observations for target are usually written as :math:`Y=y_i\in\mathbb{R}`.		
	* [Classification] The target quantity is associated with a discrete rv :math:`G\in\mathcal{G}` with :math:`|\mathcal{G}|=K`.		
	* We have a total of :math:`N` observations, and all the observations together are taken in the matrix form

		.. math:: \mathbf{X}_{N\times d}=\begin{bmatrix}-& x_1^T & - \\ \vdots & \vdots & \vdots \\ -& x_N^T & -\end{bmatrix}=\begin{bmatrix}|&\cdots&|\\ \mathbf{x}_1 & \cdots & \mathbf{x}_d \\ |&\cdots&|\end{bmatrix}
	* The vector :math:`\mathbf{x}_j\in\mathbb{R}^N` represents the column vector for all the observations for rv :math:`X_j`.
	* A particular observation for :math:`X=x_i\in\mathbb{R}^d` is better represented in the row-vector form, :math:`x_i^T\in\mathbb{R}_{1\times d}`.
	* For :math:`K> 1`, we can also associate the target with the row vector form, :math:`y_i^T\in\mathbb{R}_{1\times K}` [regression] or g_i^T\in\mathcal{G}_{1\times K} [classification].

**********************************************************************************
Statistical Decision Theory
**********************************************************************************
.. tip::
	* This puts the prediction task under a probabilistic paradigm.
	* We assume that the input variables rv :math:`X` and the target are distributed per some **unknown joint distribution**

		* :math:`X,Y\sim F_{X,Y}(x,y)` or :math:`X,G\sim F_{X,G}(x,g)`
	* We wish to find a predictor as function of data, :math:`\hat{Y}(X)` or :math:`\hat{G}(X)`.
	* We associate a **misprediction penalty** for making prediction error.

		* :math:`L(Y,\hat{Y}(X))` or :math:`L(G,\hat{G}(X))`.
	* The learning theory wishes to choose such predictors for which the expected prediction error (EPE) is minimised.

		* :math:`EPE=\mathbb{E}_{X,Y} L(Y,\hat{Y}(X))` or :math:`EPE=\mathbb{E}_{X,G} L(G,\hat{G}(X))`
	* [TODO: check the conditioning variables in the expectation] This quantity can be conditioned on observed input variables

		* [Regression] :math:`EPE=\mathbb{E}_{X,Y} L(Y,\hat{Y}(X))=\mathbb{E}_{Y|X}\left[\mathbb{E}_{Y}\left(L(Y,\hat{Y}(X))\right) |X\right]`
		* [Classification] :math:`EPE=\mathbb{E}_{X,G} L(G,\hat{G}(X))=\mathbb{E}_{G|X}\left[\mathbb{E}_{G}\left(L(G,\hat{G}(X))\right) |X\right]`
	* This quantity is minimised pointwise (i.e. at each point :math:`X=x`).

		* [Regression] :math:`\hat{Y}(x)=f(x)=\underset{f}{\arg\min}\left(\mathbb{E}_{Y|X}\left[\mathbb{E}_{Y}\left(L(Y,f(x))\right) |X=x\right]\right)`

			* If MSE loss is used, then :math:`\hat{Y}(x)=\mathbb{E}_{Y|X}\mathbb{E}_{Y}[Y|X=x]`
		* [Classification] :math:`\hat{G}(x)=g(x)=\underset{g}{\arg\min}\left(\mathbb{E}_{G|X}\left[\mathbb{E}_{G}\left(L(G,g(x))\right) |X=x\right]\right)`

			* If 0-1 loss is used, then :math:`\hat{G}(x)` corresponds to the predicted class with highest probability.

Regression
==================================================================================
.. note::
	* We're interested in finding an estimator for :math:`Y`

		.. math:: \hat{Y}=f(X)
	* Estimation error: :math:`\tilde{Y}=\hat{Y}-Y`
	* Bias: :math:`\mathbb{E}_Y[\tilde{Y}]`
	* Standard error (se): :math:`\sqrt{\mathbb{V}_Y(\hat{Y})}`
	* Mean-squared error (mse): :math:`\mathbb{E}_Y[\tilde{Y}^2]`

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
	* In this case :math:`f(x)=\mathbb{E}_{Y|X}[Y|X=x]\approx\frac{1}{k}\sum_{x_i\in N(x)} y_i`
	* The implicit assumption is that the function behaves locally constant around each point :math:`x`
	* Therefore, it can be estimated with the average value of the target :math:`y_i` for each data point in the neighbourhood :math:`x_i`.

Explicit assumption from a model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
	* In linear regression approach, we explicitly assume that the estimator is affine in :math:`X_j`.
	* In this case, :math:`f(x)=\mathbb{E}_{Y|X}[Y|X=x]\approx x^T\beta + \beta_0`
	* We usually add a dummy variable :math:`X_0=1` in :math:`X` and write this as a linear function instead

		.. math:: f(x)=\mathbb{E}_{Y|X}[Y|X=x]\approx x^T\beta

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
