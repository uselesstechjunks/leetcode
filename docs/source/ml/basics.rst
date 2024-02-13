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

.. tip::	
	* We assume that the :math:`X` and the :math:`Y/G` are distributed per some **unknown joint distribution**

		* [Regression] :math:`X,Y\sim F_{X,Y}(x,y)`
		* [Classification] :math:`X,G\sim F_{X,G}(x,g)`
	* The task is to find a predictor as function of data, :math:`\hat{Y}(X)` or :math:`\hat{G}(X)`.
	* We associate a **misprediction penalty**, L, for making an error in prediction.

		* [Regression] :math:`L(Y,\hat{Y}(X))`
		* [Classification] :math:`L(G,\hat{G}(X))`
	* We wish the predictors to have minimal expected prediction error (EPE) over the joint.

		* [Regression] :math:`EPE=\mathbb{E}_{X,Y} L(Y,\hat{Y}(X))`
		* [Classification] :math:`EPE=\mathbb{E}_{X,G} L(G,\hat{G}(X))`
	* EPE can be reformulated as conditional expectation on observed input variables :math:`X`.

		* [Regression] :math:`EPE=\mathbb{E}_{X,Y} L(Y,\hat{Y}(X))=\mathbb{E}_X\left[\mathbb{E}_{Y|X}[L(Y,\hat{Y}(X)|X]\right]=\int_x \mathbb{E}_{Y|X}[L(Y,\hat{Y}(X)|X=x]f_{Y|X}(y|x)\mathop{dx}`
		* [Classification] :math:`EPE=\mathbb{E}_{X,G} L(G,\hat{G}(X))=\mathbb{E}_X\left[\mathbb{E}_{G|X}[L(G,\hat{G}(X)|X]\right]=\int_x \mathbb{E}_{G|X}[L(G,\hat{Y}(X)|X=x]f_{G|X}(y|x)\mathop{dx}`
	* This quantity is minimised pointwise (i.e. at each point :math:`X=x`)

		* (Informally, to minimise, we take derivative of EPE which removes the integral).
		* [Regression] :math:`\hat{Y}(x)=\underset{f}{\arg\min}\left(\mathbb{E}_{Y|X}[L(Y,f(X)|X=x]\right)`
		* [Classification] :math:`\hat{G}(x)=\underset{g}{\arg\min}\left(\mathbb{E}_{G|X}[L(G,g(X)|X=x]\right)`.
	* For particular choice of loss functions, we arrive as optimal (Bayes) estimator definitions

		* [Regression] If MSE loss is used, then :math:`\hat{Y}(x)=\mathbb{E}_{Y|X}[Y|X=x]`.
		* [Classification] If 0-1 loss is used, then :math:`\hat{G}(x)` corresponds to the predicted class with highest probability.

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

**********************************************************************************
Statistical Models
**********************************************************************************
Linear Regression
kNN Classification
