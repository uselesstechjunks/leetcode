##################################################################################
Fundamentals
##################################################################################

**********************************************************************************
Notation
**********************************************************************************
.. warning::
	* All vectors are named for their column vector form. 
	* For row-representation, we use the transpose notation.

.. note::
	* The data is associated with a random variable :math:`X`.

		* It might also be a random vector for some :math:`d> 1`, in which case, individual components can referred to as :math:`X_j` and :math:`X=(X_1,\cdots,X_d)`.
	* For observed data points are instances of the rv, :math:`X=x\in\mathbb{R}^d` for some :math:`d\geq 1`.
	* [Regression] The target quantity is associated with a continuous rv :math:`Y\in\mathbb{R}`. 

		* It might also be a random vector, with :math:`Y=(Y_1,\cdots,Y_K)`, for some :math:`K\geq 1`.
		* Single dimensional observations for target are usually written as :math:`Y=y\in\mathbb{R}`.		
	* [Classification] The target quantity is associated with a discrete rv :math:`G\in\mathcal{G}` with :math:`|\mathcal{G}|=K`.		
	* We have a total of :math:`N` observations, and all the observations together are taken in the matrix form

		.. math:: \mathbf{X}_{N\times d}=\begin{bmatrix}-& x_1^T & - \\ \vdots & \vdots & \vdots \\ -& x_N^T & -\end{bmatrix}=\begin{bmatrix}|&\cdots&|\\ \mathbf{x}_1 & \cdots & \mathbf{x}_d \\ |&\cdots&|\end{bmatrix}
	* The vector :math:`\mathbf{x}_j\in\mathbb{R}^N` represents the column vector for all the observations for rv :math:`X_j`.
	* A particular observation for :math:`X=x_i\in\mathbb{R}^d` is taken in the row-vector form, :math:`x_i^T\in\mathbb{R}_{1\times d}`.
	* For :math:`K> 1`, we can also associate the target with the row vector form, :math:`y_i^T\in\mathbb{R}_{1\times K}` [regression] or :math:`g_i^T\in\mathcal{G}_{1\times K}` [classification].

**********************************************************************************
Statistical Decision Theory
**********************************************************************************
This puts the prediction task under a statistical inference paradigm.

Analytic Solutions
==================================================================================
Single Random Variable
----------------------------------------------------------------------------------
.. tip::
	* We have a single real-valued rv :math:`X` from an unknown distribution.
	* We consider the estimation problem where we find an estimate :math:`\hat{x}`, a single value, for any future observation of :math:`X`.

		* We define Prediction Error (PE): The rv :math:`\tilde{X}=X-\hat{x}`, which has the same pdf as :math:`X`.
	* The **optimality** of our estimate is defined with the help of a **loss function**.

		* Loss function is usually some function of PE.
		* MSE loss function is defined as

			.. math:: \mathbb{E}_X[\tilde{X}^2]=\mathbb{E}_X[(X-\hat{x})^2]=\mathbb{E}_X[X^2]-2\mathbb{E}_X[X]\hat{x}+\hat{x}^2
	* To find :math:`\hat{x}`, we can differentiate w.r.t. :math:`\hat{x}` to minimize EPE.

		* Note that :math:`\mathbb{E}_X[X^2]` and :math:`\mathbb{E}_X[X]` are unknown constants.
		* Therefore

			.. math:: \frac{\partial}{\mathop{\partial\hat{x}}}\mathbb{E}_X[(X-\hat{x})^2]=-2\mathbb{E}_X[X]+2\hat{x}\implies\hat{x}_{\text{OPT}}=\mathbb{E}_X[X]

Two Random Variables
----------------------------------------------------------------------------------
.. note::
	* We assume that the :math:`X` and the :math:`Y/G` are distributed per some **unknown joint distribution**

		* [Regression] :math:`X,Y\sim F_{X,Y}(x,y)`
		* [Classification] :math:`X,G\sim F_{X,G}(x,g)`
	* The task is to find an estimator as function of data, :math:`\hat{Y}=f(X)` or :math:`\hat{G}=g(X)`.

		* For a given obs :math:`X=x`, this gives predictors :math:`\hat{Y}=\hat{y}=f(x)` and :math:`\hat{G}=\hat{g}=g(x)`.
	* We associate a non-negative **misprediction penalty**, :math:`L`, for making an error in prediction.

		* [Regression] :math:`L(Y,\hat{Y})`
		* [Classification] :math:`L(G,\hat{G})`
	* We wish the predictors to have minimal expected prediction error (EPE) **over the joint**.

		* [Regression] :math:`EPE=\mathbb{E}_{X,Y} L(Y,\hat{Y})`
		* [Classification] :math:`EPE=\mathbb{E}_{X,G} L(G,\hat{G})`
	* EPE can be reformulated as conditional expectation on observed input variables :math:`X`.

		* [Regression] :math:`EPE=\mathbb{E}_X\left[\mathbb{E}_{Y|X}[L(Y,\hat{Y}|X]\right]`
		* [Classification] :math:`EPE=\mathbb{E}_X\left[\mathbb{E}_{G|X}[L(G,\hat{G}|X]\right]`
	* Since :math:`L` is non-negative, this quantity is minimised when it's minimum at each point :math:`X=x`.
		
		* As we're fixing :math:`X` to a constant, the outer expectation :math:`\mathbb{E}_X` goes away.
		* Therefore, the minimization problem becomes:
		
			* [Regression] :math:`\hat{y}_{\text{OPT}}=\underset{\hat{y}}{\arg\min}\left(\mathbb{E}_{Y|X}[L(Y,\hat{y}|X=x]\right)`
			* [Classification] :math:`\hat{g}_{\text{OPT}}=\underset{\hat{g}}{\arg\min}\left(\mathbb{E}_{G|X}[L(G,\hat{g}|X=x]\right)`.
	* For particular choice of loss functions, we arrive as optimal (Bayes) estimator definitions

		* [Regression] If MSE loss is used, then :math:`\hat{Y}=f(x)=\mathbb{E}_{Y|X}[Y|X=x]`, **mean of the conditional pdf**.
		* [Classification] If 0-1 loss is used, then :math:`\hat{G}=g(x)` corresponds to the **mode of the conditional pmf**.

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

		.. math:: f^*=\underset{f}{\arg\min}\left(\mathbb{E}_{X,Y}[(f(X)-Y)^2]\right)=\underset{f}{\arg\min}\left(\mathbb{E}_X\left[\mathbb{E}_{Y|X}[(f(X)-Y)^2]|X\right]\right)
	* This minimisation problem is equivalent to finding a pointwise minimum, such that, for each :math:`X=x`, 

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
.. note::
	* As we move to higher dimensional space, the notion of **distance** doesn't follow our intuition.
	* As this `SO post <https://stats.stackexchange.com/a/99191>`_ puts it (quoting verbatim)

		* Another application, beyond machine learning, is nearest neighbor search: given an observation of interest, find its nearest neighbors (in the sense that these are the points with the smallest distance from the query point). 
		* But in high dimensions, a curious phenomenon arises: the ratio between the nearest and farthest points approaches 1, i.e. the points essentially become uniformly distant from each other. 
		* This phenomenon can be observed for wide variety of distance metrics, but it is more pronounced for the Euclidean metric than, say, Manhattan distance metric. 
		* The premise of nearest neighbor search is that "closer" points are more relevant than "farther" points, but if all points are essentially uniformly distant from each other, the distinction is meaningless.
	* More resource on this:

		* `On the Surprising Behavior of Distance Metrics in High Dimensional Space <https://bib.dbvis.de/uploadedFiles/155.pdf>`_
		* `When Is "Nearest Neighbor" Meaningful? <https://members.loria.fr/MOBerger/Enseignement/Master2/Exposes/beyer.pdf>`_
		* `Fractional Norms and Quasinorms Do Not Help to Overcome the Curse of Dimensionality <https://www.mdpi.com/1099-4300/22/10/1105/pdf?version=1603175755>`_

**********************************************************************************
Statistical Models
**********************************************************************************
Linear Regression
kNN Classification
