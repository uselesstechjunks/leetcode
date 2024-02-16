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
	* If we allow the index set to have real values, then, theoretically, we end up with infinite lists, :math:`\mathbf{y}=\{(x,y)|x,y\in\mathbb{R}\}`

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
