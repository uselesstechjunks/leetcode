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
	* The multiplication :math:`\mathbf{A}\mathbf{x}` is a combination of the column vectors of :math:`\mathbf{A}`, where each vector :math:`\mathbf{a}_k` is scaled as per :math:`x_k`.

		.. math:: \mathbf{A}\mathbf{x}=\begin{bmatrix} | & \cdots & |\\ \mathbf{a}_1 & \cdots & \mathbf{a}_n\\ | & \cdots & |\\ \end{bmatrix}\begin{bmatrix}x_1\\\vdots\\x_n\end{bmatrix}=x_1\begin{bmatrix}|\\ \mathbf{a}_1\\|\end{bmatrix}+\cdots+x_n\begin{bmatrix}|\\ \mathbf{a}_n\\|\end{bmatrix}

.. tip::
	* The matrix :math:`\mathbf{A}` is a linear operator which maps :math:`\mathbb{R}^n` dimensional vectors to :math:`\mathbb{R}^m` dimensional vectors.

		.. math:: \mathbf{A}:\mathbb{R}^n\mapsto\mathbb{R}^m
	* The range of this operator is the **column space** of this operator

		.. math:: C(\mathbf{A})=\left{\mathbf{A}\mathbf{x}\right|\forall \mathbf{x}\in\mathbb{R}^n}

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
