##################################################################################
Basis Expansion
##################################################################################
.. warning::
	* We can use the framework for linear models to move beyond linearity just by using non-linear transforms on the features.
	* We define the transforms as functions :math:`h:\mathbb{R}^d\mapsto\mathcal{F}` where :math:`\mathcal{F}` can be finite or infinite dimensional.
	* Since these methods expand the basis from :math:`d`-dimensions to :math:`\dim\mathcal{F}`, these methods are called **basis expansion** methods.

.. tip::
	* For most of the methods, we consider the case where :math:`d=1`.

**********************************************************************************
Finite Dimensional Expansion
**********************************************************************************
.. note::
	* We define a finite set of transforms :math:`x\mapsto h_m(x)` where :math:`m=1,\cdots,M`.
	* Each observation is mapped to the transformed vector :math:`\mathbf{h}=(h_1(x),\cdots,h_M(x))^T`
	* We use the linear regression or logistic regression frameworks using these transforms

		* Regression: :math:`\hat{y}=f(x)=\sum_{i=1}^M \beta_mh_m(x)=\boldsymbol{\beta}^T\mathbf{h}`
		* Classification: :math:`\log\frac{\mathbb{P}(G=k|X=x)}{\mathbb{P}(G=K|X=x)}=\boldsymbol{\beta}^T\mathbf{h}`

Polynomial
==================================================================================
.. tip::
	* [`Taylor's series <https://en.wikipedia.org/wiki/Taylor_series>`_] An infinite power series can arbitrarily approximate any real-valued infinitely differentiable function around a point :math:`a`.

		.. math:: f(x)=f(a)+\frac{f'(a)}{2!}(x-a)+\frac{f''(a)}{3!}(x-a)^2+\cdots
	* [`Lagrange Polynomial <https://en.wikipedia.org/wiki/Lagrange_polynomial>`_] A polynomial of degree :math`n` can be fit to pass through exactly :math:`n` points (interpolation)

		* Say, we have a dataset :math:`\mathbf{X}=[x_1,\cdots,x_N]^T` and target :math:`\mathbf{y}=[y_1,\cdots,y_N]^T`.
		* For each point :math:`(x_k,y_k)`, we create the Lagrange coefficients

			.. math:: l_k(x)=\frac{(x-x_1)(x-x_2)\cdots(x-x_{k-1})(x-x_{k+1})\cdots(x-x_{N-1})(x-x_N)}{(x_k-x_1)(x_k-x_2)\cdots(x_k-x_{k-1})(x_k-x_{k+1})\cdots(x_k-x_{N-1})(x_k-x_N)}
		* The polynomial passing through all these points are given by

			.. math:: f(x)=\sum_{i=1}^N y_1 l_i(x)

.. warning::
	* A polynomial of degree :math:`n` has :math:`n-1` turns.
	* With real data, often not all the turns are utilised for the points in the interior regions.
	* With no other constraints applied, polynomials of such higher degrees oscilate crazily towards the left/right extremes where the data density is usually low.
	* Therefore, in those regions, it provides very poor generalisation of the model.
	* The situation is often worsen if we move to higher dimensions, as the proportion of points around the exterior increases in higher dimension.

.. note::
	* With these in mind, a polynomial of smaller order often works best, such as cubic polynomials.

		.. math:: x\overset{h}\mapsto[1,x,x^2,x^3]
	* However, lower degree polynomials fit to the entire data shows higher error rate due to increased bias.

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
