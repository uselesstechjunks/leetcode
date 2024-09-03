################################################################################
Vector Space and Geometric View
################################################################################

********************************************************************************
Vector Space
********************************************************************************
Let :math:`\mathcal{F}` be a scalar **field** (such as :math:`\mathbb{R}` or :math:`\mathbb{C}`).

.. attention::
	* Field refers to the algebraic definition with properly defined addition and multiplication operators on them. 
	* Not to be confused with **scalar fields** which represents functionals that maps vectors into scalers.

.. admonition:: [Definition] Vector Space

	:math:`V_\mathcal{F}` is a vector space over :math:`\mathcal{F}` with :math:`0\in \mathcal{F}` **iff**:

		* :math:`\forall a\in \mathcal{F},\mathbf{u}\in V_\mathcal{F}\implies a\cdot\mathbf{u}\in V_\mathcal{F}`
		* :math:`\mathbf{u},\mathbf{v}\in V_\mathcal{F}\implies \mathbf{u}+\mathbf{v}\in V_\mathcal{F}`
	
	with the following properties:

		* [Commutative Addition]: :math:`\mathbf{u}+\mathbf{v}=\mathbf{v}+\mathbf{u}`
		* [Identity Element]: :math:`\exists\mathbf{0}\in V_\mathcal{F}` such that

			* :math:`0\cdot\mathbf{u}=\mathbf{0}`
			* :math:`\mathbf{u}+\mathbf{0}=\mathbf{0}+\mathbf{u}=\mathbf{u}`
		* [Inverse Element]: :math:`\forall\mathbf{u}\in V_\mathcal{F},\exists\mathbf{v}\in V_\mathcal{F}` (represented as :math:`-\mathbf{u}`) such that

			:math:`\mathbf{u}+\mathbf{v}=\mathbf{0}`

.. attention::
	Vector spaces are Abelian groups w.r.t :math:`+` but the addition of scalar multiplication provides an even richer structure.

.. tip::	
	* Elements of vector space are called vectors.
	* We often omit the underlying scalar field :math:`\mathcal{F}` and write the vector space as :math:`V`.
	* Example of finite dimensional vectors: Euclidean vectors :math:`\mathbb{R}^n` where the scalar field is :math:`\mathbb{R}` or complex vectors :math:`\mathbb{C}^n` over the scalar field :math:`\mathbb{C}`.

Euclidean Vector Space
================================================================================
These defintions and theorems are based on Graybill.

.. admonition:: [Definition] n-component Vector
	Let :math:`n` be a positive integer and let :math:`a_1,\cdots,a_n` be elements from :math:`\mathcal{F}`. The ordered :math:`n`-tuple :math:`\mathbf{a}=(a_1,\cdots,a_n)^T` is defined as n-component (or :math:`n\times 1` vector)

.. seealso::
	Vector spaces with :math:`n\times 1` vectors are denoted here by :math:`V_n`.

.. admonition:: Theorem
	Let :math:`R_n` be the set of all :math:`n\times 1` vectors for a fixed positive integer :math:`n`. Then :math:`R_n` is a vector space.

.. admonition:: [Definition] Vector Subspace
	Let :math:`S_n` be the subset of vectors in the vector space :math:`V_n`. If the set :math:`S_n` itself is a vector space, then :math:`S_n` is called a vector subspace of :math:`V_n`.

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
