################################################################################
Applied Linear Algebra
################################################################################

********************************************************************************
Principle Component Analysis
********************************************************************************
.. note::
	* We have a data matrix :math:`\mathbf{X}\in\mathbb{R}^{n\times m}` which represents a sample of size :math:`n`.
	* Each row represents an observation :math:`(\mathbf{x}^*_k)^\top\in\mathbb{R}^m`.
	* We want to reduce the dimensionality of the data to :math:`k\ll m`, keeping as much information about the data as we could.

.. note::
	* Let :math:`\bar{\mathbf{X}}` be the centred version of :math:`\mathbf{X}` after removing sample mean.
	* Then :math:`\mathbf{C}=\bar{\mathbf{X}}^\top\bar{\mathbf{X}}` is the variance-covariance matrix.
	* Total variance

		.. math:: \mathbb{V}(\bar{\mathbf{X}})=\frac{1}{n}\sum_{i=1}^n\mathbb{V}(\bar{\mathbf{x}}^*_i)=\frac{1}{n}\text{trace}(\bar{\mathbf{X}})=\frac{1}{n}(\sigma_1^2+\cdots+\sigma_n^2)
	* Therefore, :math:`\mathbf{v}_1` provides the direction in which the variance of the data is maximised.
	* We can create best :math:`k` rank approximation by :math:`\mathbf{X}_k=\mathbf{U}_k\boldsymbol{\Sigma}_k\mathbf{V}_k` which achieves the goal.

.. attention::
	* This approach minimises the perpendicular distance of each data point to the singular vector directions.

********************************************************************************
Matrix Identities
********************************************************************************
.. note::
	* Matrix Inversion Lemma (needs 1 inverse instead of 3): 

		.. math:: (\mathbf{A}^{-1}+\mathbf{B}^{-1})^{-1}=\mathbf{A}-\mathbf{A}(\mathbf{A}+\mathbf{B})^{-1}\mathbf{A}

		* 1d verification: :math:`\frac{1}{1/a+1/b}=\frac{ab}{a+b}=\frac{a^2-a^2+ab}{a+b}=\frac{a(a+b)-a^2}{a+b}=a-a(a+b)^{-1}a`
	* Woodburry Identity:
