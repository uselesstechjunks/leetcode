################################################################################
Functional Analysis
################################################################################

********************************************************************************
Vector Space of Functions
********************************************************************************
Functions are vectors : Intuition
================================================================================
.. tip::
	* We usually think of vectors in finite dimensional case, e.g. :math:`\mathbf{y}\in\mathbb{R}^4`, as a list

		.. math:: \mathbf{y}=\begin{bmatrix}0.3426 \\1.3258 \\6.8943 \\8.3878 \\\end{bmatrix}\\
	* An alternate way to look at this is as a list of tuples, which binds the dimension (integer index in this case) and the value of that dimension

		.. math:: \mathbf{y}=\left((0,0.3426),(1,1.3258),(2,6.8943),(3,8.387)\right)
	* This can be represented by a function

		.. math:: f:[0,1,2,3]\mapsto[0.3426,1.3258,6.8943,8.387]
	* If we extend the index set to have real values, then, theoretically, we end up with infinite lists

		.. math:: \mathbf{y}=\{(x,y)|x,y\in\mathbb{R}\}
	* This is the way functions are defined in set theory.

Addition of two functions
--------------------------------------------------------------------------------
.. tip::
	* When we add two vectors, :math:`\mathbf{u}` and :math:`\mathbf{v}`, we add the corresponding values for each dimension.
	* For functions :math:`f:\mathcal{X}\mapsto\mathcal{Y}` and :math:`g:\mathcal{X}\mapsto\mathcal{Y}`, the dimensions are the values :math:`x\in\mathcal{X}`. 

		* [Vector Addition]: We can define **vector addition** of a function for each "index" :math:`x` as

			.. math:: (f + g)(x) = f(x) + g(x)
		* [Scalar Multiplication]: For any :math:`\alpha\in\mathcal{F}`, where :math:`\mathcal{F}` is a field, we have, for any :math:`x\in\mathcal{X}`,

			.. math:: (\alpha\cdot f)(x) = \alpha\cdot f(x)
	* We see that we don't need to restrict :math:`\mathcal{X}` and :math:`\mathcal{Y}` to reals.

		* As long as :math:`+` is well-defined in :math:`\mathcal{Y}`, we can define vector addition for functions.
		* As long as elements in :math:`\mathcal{Y}` satisfy scalar multiplication for some underlying field, we can also define scalar multiplication for functions.

Basis in functional space
================================================================================

Function Norm
================================================================================
Lp Space
--------------------------------------------------------------------------------
Sobolev Space
--------------------------------------------------------------------------------
Holder Space
--------------------------------------------------------------------------------
Hilbert Space
--------------------------------------------------------------------------------
