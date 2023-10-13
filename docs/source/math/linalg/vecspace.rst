################################################################################
Vector Space and Linear Transform
################################################################################
.. note::
	* Let :math:`\mathcal{F}` be a scalar **field** (such as :math:`\mathbb{R}` or :math:`\mathbb{C}`).

		* Field refers to the algebraic definition with properly defined addition and multiplication operators on them. 
		* Not to be confused with **scalar fields** which represents functionals that maps vectors into scalers.
	* Then :math:`V_\mathcal{F}` is a vector space over :math:`\mathcal{F}` if we have scalar multiplication and vector addition defined as follows:

		* **Scalar Multiplication**: 

			* For :math:`\mathbf{u}\in V_\mathcal{F}\implies\forall a\in \mathcal{F}, a\cdot\mathbf{u}\in V_\mathcal{F}`
		* **Vector Addition**: 

			* For :math:`\mathbf{u},\mathbf{v}\in V_\mathcal{F}\implies \mathbf{u}+\mathbf{v}\in V_\mathcal{F}`
			* There is a unique :math:`\mathbf{0}\in V_\mathcal{F}` such that 

				* For :math:`0\in \mathcal{F}`, :math:`\forall\mathbf{u}\in V_\mathcal{F}, 0\cdot\mathbf{u}=\mathbf{0}\in V_\mathcal{F}`
				* :math:`\mathbf{u}+\mathbf{0}=\mathbf{0}+\mathbf{u}=\mathbf{u}`
			* For every :math:`\mathbf{u}\in V_\mathcal{F}`, there is a unique :math:`\mathbf{v}\in V_\mathcal{F}` such that

				* :math:`\mathbf{u}+\mathbf{v}=\mathbf{0}`
				* We represent :math:`\mathbf{v}` as :math:`-\mathbf{u}`

.. tip::
	* Elements of vector space are called vectors.
	* We often omit the underlying scalar field :math:`\mathcal{F}` and write the vector space as :math:`V`.
	* Example of finite dimensional vectors: Euclidean vectors :math:`\mathbb{R}^n` where the scalar field is :math:`\mathbb{R}` or complex vectors :math:`\mathbb{C}^n` over the scalar field :math:`\mathbb{C}`.

********************************************************************************
Linear Transform
********************************************************************************
.. note::
	* Let :math:`U` and :math:`W` be two vector spaces over the same scalar field :math:`\mathcal{F}`.
	* A linear transform is a function :math:`T:U\mapsto W` if 

		* :math:`\forall\mathbf{u},\mathbf{v}\in U, T(\mathbf{u}+\mathbf{v})=T(\mathbf{u})+T(\mathbf{v})`
		* :math:`\forall c\in\mathcal{F},\forall\mathbf{u}\in U, T(c\cdot\mathbf{u})=c\cdot T(\mathbf{u})`
	* This means, if we want to add or scale vecetors, it doesn't matter whether we do it in the domain space before the mapping or in the range space after the mapping.

.. tip::
	A linear transform is one-to-one when it's onto.

Linear Operator
================================================================================
.. tip::
	* Linear transforms from :math:`U` to :math:`U` are called Linear Operators.
	* The set of all linear operators :math:`T:U\mapsto U` is represented as :math:`L(U)`.

Space of Linear Transform
================================================================================
.. tip::
	The set of all linear transforms :math:`T:U\mapsto W` is represented as :math:`L(U,W)`.

As a Vector Space over Addition
--------------------------------------------------------------------------------
.. seealso::
	* Let's consider :math:`A,B\in L(U,W)`.
	* We can define an addition operator in :math:`L(U,V)` with the same scalar multiplication of :math:`W`.

		* Let :math:`C=(a\cdot A+b\cdot B)` where for any :math:`a,b\in\mathcal{F}` we have

			.. math:: \forall\mathbf{u}\in U, C(\mathbf{u})=(a\cdot A+b\cdot B)(\mathbf{u})=a\cdot A(\mathbf{u})+b\cdot B(\mathbf{u})
		* We note that :math:`C\in L(U,W)`.
	* We also define an identity operator :math:`0_L\in L(U,W)` such that :math:`\forall \mathbf{u}, 0_L(\mathbf{u})=\mathbf{0}`.

		* We note that :math:`A+0_L=0_L+A=A`.
	* If the transform is **onto**, we can define a unique additive inverse :math:`-A:U\mapsto W`.

		.. math:: A(\mathbf{u})+-A(\mathbf{u})=-A(\mathbf{u})+A(\mathbf{u})=0_L(\mathbf{u})=\mathbf{0}

Composition of Linear Transforms
================================================================================
.. note::
	* We can define composite linear transforms in the usual way.
	* Let :math:`A:U\mapsto V` and :math:`B:V\mapsto W`.
	* Then :math:`(B\circ A)\in L(U,W)` where :math:`\forall\mathbf{u}\in U, (B\circ A)(\mathbf{u})=B(A(\mathbf{u}))`.

As a Vector Space over Composition
--------------------------------------------------------------------------------
.. seealso::
	* Let's consider :math:`A,B\in L(U)`.
	* We note that :math:`\circ` serves as a different "addition" operator with the same scalar multiplication of :math:`U`.

		* Let :math:`C=((b\cdot B)\circ (a\cdot A))\in L(U)` where for any :math:`a,b\in\mathcal{F}` we have

			.. math:: \forall\mathbf{u}\in U, C(\mathbf{u})=((b\cdot B)\circ (a\cdot A))(\mathbf{u})=ab\cdot B(A(\mathbf{u}))
	* We define the identity operator :math:`I:U\mapsto U` such that :math:`\forall \mathbf{u}, I(\mathbf{u})=\mathbf{u}`.

		We note that :math:`A\circ I = I\circ A = A`
	* If the transform is **onto**, then we can define a unique composition inverse :math:`A^{-1}:U\mapsto U` such that

		.. math:: (A\circ A^{-1})(\mathbf{u}) = (A^{-1}\circ A)(\mathbf{u}) = I(\mathbf{u}) = \mathbf{u}

Examples
================================================================================
Scalar Multiplication as a Linear Transform
--------------------------------------------------------------------------------
.. attention::
	* For every scalar :math:`\alpha\in\mathbb{R}`, we can define a unique linear operator in :math:`L(\mathbb{R})` with its already defined multiplication operator as :math:`\alpha:\mathbb{R}\mapsto\mathbb{R}` where :math:`\forall x\in\mathbb{R}, \alpha(x)=\alpha\cdot x`.
	* We note that

		* :math:`\forall u,v\in \mathbb{R}, \alpha(u+v)=\alpha(u)+\alpha(v)`
		* :math:`\forall c\in\mathbb{R},\forall u\in \mathbb{R}, \alpha(c\cdot u)=c\cdot\alpha(u)`

Differentiation as a Linear Transform
--------------------------------------------------------------------------------

Integration as a Linear Transform
--------------------------------------------------------------------------------
