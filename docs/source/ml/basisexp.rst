##################################################################################
Basis Expansion
##################################################################################
.. warning::
	* We can use the linear model framework to move beyond linearity by using non-linear transforms on the features.
	* We define the transforms as functions :math:`h:\mathbb{R}^d\mapsto\mathcal{F}` where :math:`\mathcal{F}` can be finite or infinite dimensional.
	* Since these methods expands the original dimensions from :math:`d` to :math:`\dim\mathcal{F}`, these methods are called **basis expansion** methods.

**********************************************************************************
Finite Dimensional Expansion
**********************************************************************************
.. note::
	* We define a finite dimensional transform :math:`x\mapsto h_m(x)` where :math:`m=1,\cdots,M`.
	* We use the linear regression or logistic regression frameworks using these transforms, such that

		* Regression: :math:`\hat{y}=f(x)=\sum_{i=1}^M \beta_mh_m(x)=\boldsymbol{\beta}^T\mathbf{h}`
		* Classification: :math:`\log\frac{\mathbb{P}(G=k|X=x)}{\mathbb{P}(G=K|X=x)}=\boldsymbol{\beta}^T\mathbf{h}`

Polynomial
==================================================================================

Piece-wise Polynomial
==================================================================================

Polynomial Spline
==================================================================================

Natural Spine
==================================================================================

Smoothing Spline
==================================================================================

**********************************************************************************
Infinite Dimensional Expansion
**********************************************************************************

Kernel Ridge Regression
==================================================================================
