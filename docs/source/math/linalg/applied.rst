################################################################################
Applied Linear Algebra
################################################################################

********************************************************************************
Principle Component Analysis
********************************************************************************
.. note::
	* We have a data matrix :math:`\mathbf{X}\in\mathbb{R}^{n\times m}` which represents a sample of size :math:`n`.
	* Each row represents an observation :math:`(\mathbf{x}^*_k)^\top\in\mathbb{R}^m`.
	* Let :math:`\bar{\mathbf{X}}` be the centred version of :math:`\mathbf{X}` after removing sample mean.	
	* We want to reduce the dimensionality of the data to :math:`k\ll m`, keeping as much information about the data as we could.
