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
	* 

Special case: Symmetric Real Matrices
================================================================================

********************************************************************************
Singular Value Decomposition
********************************************************************************
