################################################################################
Vector Space and Linear Transform
################################################################################

********************************************************************************
Algebraic Structures
********************************************************************************
.. note::
	* Any set :math:`X` can be endowed with rich albegraic structures.
	* In the following, we assume that all the elements are in :math:`X`.
	* In each of the following operations, we assume that it creates a closure, i.e. it maps to some other element in :math:`X` itself.

Group
================================================================================
.. note::
	* We have an addition :math:`+` defined for :math:`X` which follows **associativity**, i.e.

		* :math:`x+(y+z)=(x+y)+z`.
	* There is a unique identity element :math:`0` w.r.t :math:`+` such that

		* :math:`x+0=0+x=x`
	* For every :math:`x`, there is a unique inverse element w.r.t :math:`+`, :math:`-x` such that

		* :math:`x+(-x)=(-x)+x=0`

Abelian Group
--------------------------------------------------------------------------------
.. note::
	* It is a group.
	* The addition operator has to be **commutative** so that

		* :math:`x+y=y+x`

Ring
================================================================================
.. note::
	* It is an Abelian group w.r.t the addition operator :math:`+`.
	* It also has a multiplication :math:`\cdot` defined so that

		* It is associative, :math:`x\cdot (y\cdot z)=(x\cdot y)\cdot z`
		* There is a unique identity element :math:`1` w.r.t :math:`\cdot` such that

			* :math:`x\cdot 1=1\cdot x=x`
		* For every :math:`x`, there is a unique inverse element w.r.t :math:`\cdot`, :math:`x^{-1}` such that

			* :math:`x\cdot x^{-1}=x^{-1}\cdot x=1`

Field
================================================================================
.. note::
	* It is a ring.
	* It is an Abelian group w.r.t the addition operator :math:`+` as well as the multiplication opeartor :math:`\cdot`.

********************************************************************************
Vector Space
********************************************************************************
.. note::
	* Let :math:`\mathcal{F}` be a scalar **field** (such as :math:`\mathbb{R}` or :math:`\mathbb{C}`).

		* Field refers to the algebraic definition with properly defined addition and multiplication operators on them. 
		* Not to be confused with **scalar fields** which represents functionals that maps vectors into scalers.
	* Then :math:`V_\mathcal{F}` is a vector space over :math:`\mathcal{F}` if we have **scalar multiplication** and a **commutative addition** defined as follows:

		* **Scalar Multiplication**: 

			* For :math:`\mathbf{u}\in V_\mathcal{F}\implies\forall a\in \mathcal{F}, a\cdot\mathbf{u}\in V_\mathcal{F}`
		* **Commutative Vector Addition**: 

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

Space of Linear Transforms
================================================================================
.. tip::
	The set of all linear transforms :math:`T:U\mapsto W` is represented as :math:`L(U,W)`.

As a Vector Space over Addition? Always.
--------------------------------------------------------------------------------
.. seealso::
	* Let's consider :math:`A,B\in L(U,W)`.
	* We can define a commutative addition in :math:`L(U,V)` with the same scalar multiplication of :math:`W`.

		* Let :math:`C=(a\cdot A+b\cdot B)` where for any :math:`a,b\in\mathcal{F}` we have

			.. math:: \forall\mathbf{u}\in U, C(\mathbf{u})=(a\cdot A+b\cdot B)(\mathbf{u})=a\cdot A(\mathbf{u})+b\cdot B(\mathbf{u})
		* We note that :math:`C\in L(U,W)`.
	* We also define an identity operator :math:`0_L\in L(U,W)` such that :math:`\forall \mathbf{u}, 0_L(\mathbf{u})=\mathbf{0}`.

		* We note that :math:`A+0_L=0_L+A=A`.
	* We can define a unique additive inverse :math:`-A:U\mapsto W`.

		.. math:: A(\mathbf{u})+-A(\mathbf{u})=-A(\mathbf{u})+A(\mathbf{u})=0_L(\mathbf{u})=\mathbf{0}

Composition of Linear Transforms
================================================================================
.. note::
	* We can define composite linear transforms in the usual way.
	* Let :math:`A:U\mapsto V` and :math:`B:V\mapsto W`.
	* Then :math:`(B\circ A)\in L(U,W)` where :math:`\forall\mathbf{u}\in U, (B\circ A)(\mathbf{u})=B(A(\mathbf{u}))`.

As a Vector Space over Composition? Not Guaranteed.
--------------------------------------------------------------------------------
.. seealso::
	* Let's consider :math:`A,B\in L(U)`.
	* We can consider :math:`\circ` as an "addition" in :math:`L(U,V)` with the same scalar multiplication of :math:`U`.

		* Let :math:`C=((b\cdot B)\circ (a\cdot A))\in L(U)` where for any :math:`a,b\in\mathcal{F}` we have

			.. math:: \forall\mathbf{u}\in U, C(\mathbf{u})=((b\cdot B)\circ (a\cdot A))(\mathbf{u})=ab\cdot B(A(\mathbf{u}))
	* We define the identity operator :math:`I:U\mapsto U` such that :math:`\forall \mathbf{u}, I(\mathbf{u})=\mathbf{u}`.

		We note that :math:`A\circ I = I\circ A = A`
	* If the transform is **onto**, then we can define a unique composition inverse :math:`A^{-1}:U\mapsto U` such that

		.. math:: (A\circ A^{-1})(\mathbf{u}) = (A^{-1}\circ A)(\mathbf{u}) = I(\mathbf{u}) = \mathbf{u}

.. warning::
	* HOWEVER, The composition operator is not always **commutative**.
	
		* It is generally NOT true that :math:`(A\circ B)\mathbf{u}=(B\circ A)\mathbf{u}`.
	* Example where it IS commutative:

		* Let :math:`\mathbf{A}` and :math:`\mathbf{B}` be matrices with the same eigenvectors and possibly different eigenvalues.
		* In this case, the composition is commutative.
		* We note that this is a sufficient but not a necessary condition.

.. attention::
	* The composition operator, therefore, is better thought of as a **multiplication**.
	* Together with the addition and multiplication, the space of linear operators follows the structure of a **ring**.

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
