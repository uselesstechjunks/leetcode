################################################################################
Vector Space and Geometric View
################################################################################

********************************************************************************
Vector Space
********************************************************************************
.. note::
	* Let :math:`\mathcal{F}` be a scalar **field** (such as :math:`\mathbb{R}` or :math:`\mathbb{C}`).

		* Field refers to the algebraic definition with properly defined addition and multiplication operators on them. 
		* Not to be confused with **scalar fields** which represents functionals that maps vectors into scalers.

.. admonition:: Definition: Vector Space

	:math:`V_\mathcal{F}` is a vector space over :math:`\mathcal{F}` **iff** we have **scalar multiplication** and a **commutative addition** defined as follows:

	* Scalar Multiplication:

		* For :math:`\mathbf{u}\in V_\mathcal{F}\implies\forall a\in \mathcal{F}, a\cdot\mathbf{u}\in V_\mathcal{F}`
	* Commutative Vector Addition: 

		* For :math:`\mathbf{u},\mathbf{v}\in V_\mathcal{F}\implies \mathbf{u}+\mathbf{v}=\mathbf{v}+\mathbf{u}\in V_\mathcal{F}`
		* There is a unique :math:`\mathbf{0}\in V_\mathcal{F}` such that 

			* For :math:`0\in \mathcal{F}`, :math:`\forall\mathbf{u}\in V_\mathcal{F}, 0\cdot\mathbf{u}=\mathbf{0}\in V_\mathcal{F}`
			* :math:`\mathbf{u}+\mathbf{0}=\mathbf{0}+\mathbf{u}=\mathbf{u}`
		* For every :math:`\mathbf{u}\in V_\mathcal{F}`, there is a unique :math:`\mathbf{v}\in V_\mathcal{F}` such that

			* :math:`\mathbf{u}+\mathbf{v}=\mathbf{0}`
			* We represent :math:`\mathbf{v}` as :math:`-\mathbf{u}`

.. attention::
	Vector spaces are Abelian groups w.r.t :math:`+` but the addition of scalar multiplication provides an even richer structure.

.. tip::	
	* Elements of vector space are called vectors.
	* We often omit the underlying scalar field :math:`\mathcal{F}` and write the vector space as :math:`V`.
	* Example of finite dimensional vectors: Euclidean vectors :math:`\mathbb{R}^n` where the scalar field is :math:`\mathbb{R}` or complex vectors :math:`\mathbb{C}^n` over the scalar field :math:`\mathbb{C}`.

Euclidean Vector Space
================================================================================

********************************************************************************
Affine Sets in Euclidean Vector Space
********************************************************************************

Line
================================================================================

Plane
================================================================================

********************************************************************************
Orthogonal Projections
********************************************************************************
