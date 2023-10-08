################################################################################
Matrix Factorisation
################################################################################

********************************************************************************
A=CR
********************************************************************************
This factorisation keeps the columns of the original matrix intact.

.. note::
	* Let the column matrix be :math:`\mathbf{C_0}=[]`
	* For :math:`i=1` to :math:`r`:

		* Select column :math:`\mathbf{a}_i` if :math:`\mathbf{a}_i\notin\text{span}(C_i)`
		* Update :math:`\mathbf{C_i}=\begin{bmatrix}\mathbf{C_{i-1}}\\ \mathbf{a}_i\end{bmatrix}`
	* To find :math:`R`:

		* For the columns of :math:`\mathbf{A}` that are already in :math:`\mathbf{C}`, the row would have a 1 to select that column and 0 everywhere else.
		* For the dependent columns, we put the right coefficients which recreates the column from others above it.

.. attention::
	* The column vectors in :math:`\mathbf{C}` create one of the basis for :math:`C(\mathbf{A})`.

.. tip::
	* If the matrix is made of data, then this is desirable as it preserves the original columns.
	* A similar factorisation can also be achieved using original rows as well, :math:`\mathbf{A}=\mathbf{C}\mathbf{M}\mathbf{R}` where :math:`\mathbf{R}` consists of indepoendent row-vectors and :math:`\mathbf{M}_{r\times r}` is a mixing matrix.

********************************************************************************
Gram-Schmidt Orgthogonalisation
********************************************************************************

********************************************************************************
Eigendecomposition
********************************************************************************
.. note::
	* When we're dealing with square matrices :math:`\mathbf{A}_{n\times n}`, we can think of the matrix transforming the input space by rotating or stretching or flipping directions.
	* Under such a linear transform, there are certain directions which stay fixed (they don't rotate). They just get stretched or flipped.
	* For any vector along these directions, the effect of matrix multiplication is just the same as scalar multiplication which stretches/flips the vectors.

		.. math:: \mathbf{A}\mathbf{x}=\lambda\mathbf{x}

		* :math:`\lambda` is a eigenvalue and :math:`\mathbf{x}` is a eigenvector of :math:`\mathbf{A}`.
		* These are determined entirely by the matrix of the linear transform :math:`\mathbf{A}`.
	* If all such vectors are collected in a matrix, then

		.. math:: \mathbf{A}\mathbf{X}=\mathbf{A}\begin{bmatrix}|&\cdots&|\\\mathbf{x}_1&\cdots&\mathbf{x}_n\\|&\cdots&|\end{bmatrix}=\begin{bmatrix}|&\cdots&|\\\mathbf{A}\mathbf{x}_1&\cdots&\mathbf{A}\mathbf{x}_n\\|&\cdots&|\end{bmatrix}=\begin{bmatrix}|&\cdots&|\\\lambda_1\mathbf{x}_1&\cdots&\lambda_n\mathbf{x}_n\\|&\cdots&|\end{bmatrix}=\begin{bmatrix}|&\cdots&|\\\mathbf{x}_1&\cdots&\mathbf{x}_n\\|&\cdots&|\end{bmatrix}\begin{bmatrix}\lambda_1 & 0 & \dots & 0 \\ 0 & \lambda_2 & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & \lambda_n\end{bmatrix}=\mathbf{X}\boldsymbol{\Lambda}

	* Therefore, the matrix factorises as 

		.. math:: \mathbf{A}=\mathbf{A}(\mathbf{X}\mathbf{X}^{-1})=(\mathbf{A}\mathbf{X})\mathbf{X}^{-1}=(\mathbf{X}\boldsymbol{\Lambda})\mathbf{X}^{-1}=\mathbf{X}\boldsymbol{\Lambda}\mathbf{X}^{-1}

Real and Complex Eigenvalues
================================================================================
.. note::
	* The eigenvalues can be real or complex.
	* **Symmetric matrices have real eigenvalues** while **orthogonal matrices have complex eigenvalues**.

Matrix power
================================================================================
	.. math:: \mathbf{A}^n\mathbf{u}=(\mathbf{X}\boldsymbol{\Lambda}\mathbf{X}^{-1})^n\mathbf{u}=(\mathbf{X}\boldsymbol{\Lambda}\mathbf{X}^{-1})(\mathbf{X}\boldsymbol{\Lambda}\mathbf{X}^{-1})\cdots(\mathbf{X}\boldsymbol{\Lambda}\mathbf{X}^{-1})\mathbf{u}=(\mathbf{X}\boldsymbol{\Lambda}^n\mathbf{X}^{-1})\mathbf{u}

.. attention::
	* For the eigenvalues that are real, the vectors get stretched repeatedly (and flipped alternatively, if eigenvalues are negative) along that direction as the effect is the same as multiplication by a real number.
	* For the eigenvalues that are complex, the vectors oscilate as the effect is the same as multiplication by a complex number.

Trace and Determinant
================================================================================
.. note::
	* **Trace**: :math:`\sum_{i=1}^n\lambda_i`
	* **Determinant**: :math:`\prod_{i=1}^n\lambda_i`

Properties
================================================================================
.. warning::
	* It is not necessary that the eigenvectors are orthogonal.

		* Iff :math:`\mathbf{A}\mathbf{A}^\top=\mathbf{A}^\top\mathbf{A}`, then eigenvectors are orthogonal.
	* It is not necessary that the eigenvalues are all distinct.

		* If all eigenvalues are distinct, then the matrix is full rank.
	* Double eigenvalues :math:`\lambda_i=\lambda_j` might or might not have independent eigenvectors.
	* **IT IS NOT TRUE** that 

		* :math:`\lambda(\mathbf{A}+\mathbf{B})=\lambda(\mathbf{A})+\lambda(\mathbf{B})`
		* :math:`\lambda(\mathbf{A}\mathbf{B})=\lambda(\mathbf{A})\cdot\lambda(\mathbf{B})`

.. tip::
	For :math:`\mathbf{B}=\mathbf{A}-a\cdot\mathbf{I}`, :math:`\lambda(\mathbf{B})=\lambda(\mathbf{A})-a`

Special case: Symmetric Real Matrices
================================================================================
.. note::
	* For real symmetric matrices :math:`\mathbf{S}`

		* The eigenvalues are all real
			
			* Proof Hint
		* The eigenvectors are orthogonal

			* Proof Hint
	* We usually write :math:`\mathbf{S}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top`

Positive Definite Matrices
--------------------------------------------------------------------------------
.. note::
	* All eigenvalues are positive.

********************************************************************************
Singular Value Decomposition
********************************************************************************
