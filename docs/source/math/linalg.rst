################################################################################
Linear Algebra
################################################################################

********************************************************************************
Matrix-vector multiplication
********************************************************************************
Let :math:`\mathbf{A}` be a :math:`m\times n` matrix

	.. math:: \mathbf{A}=\begin{bmatrix} | & \cdots & |\\ \mathbf{a}_1 & \cdots & \mathbf{a}_n\\ | & \cdots & |\\ \end{bmatrix}

where :math:`\mathbf{a}_k\in\mathbb{R}^m` are column vectors. Let :math:`\mathbf{x}\in\mathbb{R}^n` be a column vector which can also be thought of as a :math:`n\times 1` matrix

	.. math:: \mathbf{x}=(x_1,\cdots,x_n)^\top=\begin{bmatrix} x_1\\ \vdots\\ x_n \end{bmatrix}

.. note::
	The multiplication :math:`\mathbf{A}\mathbf{x}` is a combination of the column vectors of :math:`\mathbf{A}`, where each vector :math:`\mathbf{a}_k` is scaled as per :math:`x_k`.

		.. math:: \mathbf{A}\mathbf{x}=\begin{bmatrix} | & \cdots & |\\ \mathbf{a}_1 & \cdots & \mathbf{a}_n\\ | & \cdots & |\\ \end{bmatrix}\begin{bmatrix}x_1\\\vdots\\x_n\end{bmatrix}=x_1\begin{bmatrix}|\\ \mathbf{a}_1\\|\end{bmatrix}+\cdots+x_n\begin{bmatrix}|\\ \mathbf{a}_n\\|\end{bmatrix}

.. tip::
	* The matrix :math:`\mathbf{A}` is a linear operator which maps :math:`\mathbb{R}^n` dimensional vectors to :math:`\mathbb{R}^m` dimensional vectors.

		.. math:: \mathbf{A}:\mathbb{R}^n\mapsto\mathbb{R}^m
	* The range of this operator is the **column space** of this operator

		.. math:: C(\mathbf{A})=\{\mathbf{A}\mathbf{x}\mathop{|}\forall \mathbf{x}\in\mathbb{R}^n\}
	* The transposed matrix :math:`\mathbf{A}^\top` does the mapping the other way around (but it's not necessarily the inverse operator)

		.. math:: \mathbf{A}^\top:\mathbb{R}^m\mapsto\mathbb{R}^n
	* The range of the transpose operator is the **row space** of :math:`\mathbf{A}`

		.. math:: C(\mathbf{A}^\top)=\{\mathbf{A}^\top\mathbf{y}\mathop{|}\forall \mathbf{y}\in\mathbb{R}^m\}

.. attention::
	The equation :math:`\mathbf{A}\mathbf{x}=\mathbf{b}` has a unique solution if :math:`\mathbf{b}\in C(\mathbf{A})`.

********************************************************************************
Matrix-matrix multiplication
********************************************************************************
Let :math:`\mathbf{A}` be the matrix as before and let :math:`\mathbf{B}` be a :math:`n\times p` matrix written as a collection of rows similar to a vector

	.. math:: \mathbf{B}=\begin{bmatrix}-&\mathbf{b}^*_1&-\\&\vdots&\\-&\mathbf{b}^*_n&-\end{bmatrix}

where :math:`\mathbf{b}^*_k\in\mathbb{R}^p` are the row vectors.

.. note::
	The multiplication :math:`\mathbf{A}\mathbf{B}` is the sum of outer products :math:`\mathbf{u}\mathbf{v}^\top=\mathbf{a}_k \mathbf{b}^*_k`

		.. math:: \mathbf{A}\mathbf{B}=\begin{bmatrix} | & \cdots & |\\ \mathbf{a}_1 & \cdots & \mathbf{a}_n\\ | & \cdots & |\\ \end{bmatrix}\begin{bmatrix}-&\mathbf{b}^*_1&-\\&\vdots&\\-&\mathbf{b}^*_n&-\end{bmatrix}=\begin{bmatrix}|\\ \mathbf{a}_1\\|\end{bmatrix}\begin{bmatrix}-&\mathbf{b}^*_1&-\end{bmatrix}+\cdots+\begin{bmatrix}|\\ \mathbf{a}_n\\|\end{bmatrix}\begin{bmatrix}-&\mathbf{b}^*_n&-\end{bmatrix}

********************************************************************************
Independence, Rank, Inverse Mapping, Basis and Subspaces
********************************************************************************
Independence
================================================================================
.. note::
	* Vector :math:`\mathbf{u}` is linearly independent of vector :math:`\mathbf{v}` if they are not in the same direction.

		* There is not scalar :math:`a\in\mathbb{R}` such that :math:`\mathbf{u}=a\mathbf{v}`
	* Vector :math:`\mathbf{w}` is linearly independent of vectors :math:`\mathbf{u}` and :math:`\mathbf{v}` if it is not in the same place spanned by these.

		* There are no scalars :math:`a,b\in\mathbb{R}` such that :math:`\mathbf{w}=a\mathbf{u}+b\mathbf{v}`
	* Extends naturally for more dimensions.

Rank
================================================================================
Rank determines whether the linear operator :math:`\mathbf{A}` defines a mapping which is **onto** or **into**.

.. note::
	* The number of independent column vectors in a matrix :math:`\mathbf{A}` is the **column-rank**.
	* The number of independent row vectors in a matrix :math:`\mathbf{A}` is the **row-rank**.

.. attention::
	* For any matrix :math:`\mathbf{A}`, column-rank and row-rank are the same, and it is called the **rank of a matrix**, :math:`r\leq m` and :math:`r\leq n`.
	* :math:`r` is the dimensionality of the column-space :math:`C(\mathbf{A})` as well as the row-space :math:`C(\mathbf{A}^\top)`.
	* If :math:`m=n=r`, then the matrix is **full-rank**.

Inverse Mapping
================================================================================
.. note::
	* A full rank matrix :math:`\mathbf{A}:\mathbb{R}^n\mapsto\mathbb{R}^n` defines a **onto** mapping, i.e. it spans the entire range.
	* In such cases, the operation is **one-to-one** as well. There are no two vectors in the domain which maps to the same vector in the range space.
	* We can define an inverse operator in this case as :math:`\mathbf{A}^{-1}:\mathbb{R}^n\mapsto\mathbb{R}^n`.

Basis
================================================================================
.. note::
	* For a matrix :math:`\mathbf{A}` of rank :math:`r`, there are :math:`r` independent column vectors which span :math:`\mathbb{R}^r`.
	* These column vectors form **one** basis of the column space.
	* We note that these don't necessarily have to be orthogonal.

.. attention::
	* There can be multiple basis vectors for a matrix which span the same column space.

********************************************************************************
Topics
********************************************************************************
#. Fundamental Subspaces
#. Eigen Decomposition
#. Singular Value Decomposition
#. Moore-Penrose Pseudo-inverse
#. Principle Component Analysis
#. Non-negative Matrix Factorisation
#. Computational Aspects
