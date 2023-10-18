##################################################################################
Fundamentals
##################################################################################

**********************************************************************************
Notation
**********************************************************************************
.. note::
	* All vectors are named for their column vector form.
	* Our data point is in :math:`\mathbb{R}^d` for some :math:`d\geq 1`.
	* The data is a random variable :math:`X` which might be a random vector for :math:`d> 1` :math:`X=(X_1,\cdots,X_d)`.
	* The target is a random variable :math:`Y` which might also be a randomv vector :math:`Y=(Y_1,\cdots,Y_K)`.
	* An observation for :math:`X=x_i\in\mathbb{R}^d` might be taken in the row-vector form, :math:`x_i^T`.
	* We have a total of :math:`N` observations, and all the observations together are taken in the matrix form

		.. math:: \mathbf{X}=\begin{bmatrix}-& x_1^T & - \\ \vdots & \vdots & \vdots \\ -& x_N^T & -\end{bmatrix}
	* The vector :math:`\mathbf{x}_j\mathbb{R}^N` represents the column vector for all the observations for rv :math:`X_j`.

**********************************************************************************
Statistical Decision Theory
**********************************************************************************
Regression
==================================================================================
Bayes Estimator
----------------------------------------------------------------------------------
Approximating The Bayes Estimator
----------------------------------------------------------------------------------

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
