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
Topics
********************************************************************************
#. Fundamental Subspaces
#. Eigen Decomposition
#. Singular Value Decomposition
#. Moore-Penrose Pseudo-inverse
#. Principle Component Analysis
#. Non-negative Matrix Factorisation
#. Computational Aspects
