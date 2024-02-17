################################################################################
Functional Analysis
################################################################################

********************************************************************************
Vector Space of Functions
********************************************************************************
Functions are vectors : Intuition
================================================================================
.. tip::
	* We usually think of vectors in finite dimensional case, e.g. :math:`\mathbf{y}\in\mathbb{R}^4`, as a list

		.. math:: \mathbf{y}=\begin{bmatrix}0.3426 \\1.3258 \\6.8943 \\8.3878 \\\end{bmatrix}\\
	* An alternate way to look at this is as a list of tuples, which binds the dimension (integer index in this case) and the value of that dimension

		.. math:: \mathbf{y}=\left((0,0.3426),(1,1.3258),(2,6.8943),(3,8.387)\right)
	* This can be represented by a function

		.. math:: f:[0,1,2,3]\mapsto[0.3426,1.3258,6.8943,8.387]
	* If we extend the index set to have real values, then, theoretically, we end up with infinite lists

		.. math:: \mathbf{y}=\{(x,y)|x,y\in\mathbb{R}\}
	* This is the way functions are defined in set theory.

Addition and Scalar Multiplication
--------------------------------------------------------------------------------
.. tip::
	* For finite dimensional vectors :math:`\mathbf{u},\mathbf{v}\in V_{\mathcal{F}}`, where :math:`\mathcal{F}` is the underlying field:

		* [Vector Addition]: We add the corresponding values for each dimension.
		* [Scalar Multiplication]: For any :math:`\alpha\in\mathcal{F}`, we multiply :math:`\alpha\in\mathcal{F}` to the value of each dimension.
	* For functions :math:`f:\mathcal{X}\mapsto\mathcal{Y}` and :math:`g:\mathcal{X}\mapsto\mathcal{Y}`, the dimensions are :math:`x\in\mathcal{X}`. 

		* [Vector Addition]: We can define vector addition of a function for each "dimension" :math:`x`

			.. math:: (f + g)(x) = f(x) + g(x)
		* [Scalar Multiplication]: We can define scalar multiplication for each "dimension" :math:`x`

			.. math:: (\alpha\cdot f)(x) = \alpha\cdot f(x)
	* We don't need to restrict :math:`\mathcal{X}` and :math:`\mathcal{Y}` to reals.

		* As long as :math:`+` is well-defined in :math:`\mathcal{Y}`, we can define vector addition for functions.
		* As long as elements in :math:`\mathcal{Y}` satisfy scalar multiplication for some underlying field, we can also define scalar multiplication for functions.

Inner Product
--------------------------------------------------------------------------------
.. tip::
	* For finite dimensional vectors :math:`\mathbf{u},\mathbf{v}\in V_{\mathcal{F}}`, to compute the inner (dot) product

		* We multiply the corresponding values for each dimension.
		* We sum the results across all dimensions.

			.. math:: \langle\mathbf{u},\mathbf{v}\rangle=\sum_{i=1}^n u_i\cdot v_i
	* For functions :math:`f:\mathcal{X}\mapsto\mathcal{Y}` and :math:`g:\mathcal{X}\mapsto\mathcal{Y}`

		* We can do the multiplication for each dimension :math:`x`
		* However, since :math:`\mathcal{X}` is uncountable, we replace the sum with integration

			.. math:: \langle f,g\rangle=\int_{\mathcal{X}}f(x)\cdot g(x)\mathop{dx}
		* We note that we need to have multiplication between elements, :math:`\cdot`, well defined in :math:`\mathcal{Y}`.

Norm
================================================================================
Lp Space
--------------------------------------------------------------------------------
.. note::
	* The inner product for finite vectors induces a norm (:math:`l_2`)

		.. math:: ||\mathbf{u}||_2^2=\langle \mathbf{u},\mathbf{u}\rangle=\sum_{i=1}^n|u_i|^2
	* The inner product defined above induces a norm

		.. math:: ||f||_2^2=\langle f,f\rangle=\int_{\mathcal{X}}|f(x)|^2\mathop{dx}
	* More generally, we can have

		.. math:: ||f||_{L_p}=\left(\int_{\mathcal{X}}|f(x)|^p\mathop{dx}\right)^{1/p}
	* For more general measurable spaces where we have a measure :math:`\mu(x)` defined

		.. math:: ||f||_{L_p(\mathcal{X},\mu)}=\left(\int_{\mathcal{X}}|f(x)|^p\mathop{d\mu}(x)\right)^{1/p}
	* For :math:`p=\infty`

		.. math:: ||f||_{L_\infty(\mathcal{X},\mu)}=\text{ess}\sup_\limits{x\in\mathcal{X}}|f(x)|
	* We write the function space as :math:`L^p(\mathcal{X},\mathcal{Y})=\{f|f:\mathcal{X}\mapsto\mathcal{Y};\text{such that }L_p(\mathcal{X,\mu})\text{ exists}\}`

		* Example: :math:`L^2([0,1],\mathbb{R})`

Sobolev Space
--------------------------------------------------------------------------------
.. note::
	* TODO

Metric
================================================================================
.. note::
	* The :math:`l_p` norm for finite vectors induces a metric 

		.. math:: d(\mathbf{u}, \mathbf{v})=||\mathbf{u}-\mathbf{v}||_2=\left(\sum_{i=1}^n|u_i-v_i|^p\right)^{1/p}
	* We can define, similarly, for functions

		.. math:: d(f, g)=||f-g||_{L_p(\mathcal{X},\mu)}=\left(\int_\limits{i=1}^n|f(x)-g(x)|^p\mathop{d\mu}(x)\right)^{1/p}

		* If :math:`d(f, g)=0`, then the functions are the same "almost everywhere".
		* In this case, they are different for **at most** finitely many "dimensions".

Function Basis
================================================================================
Fourier Basis
--------------------------------------------------------------------------------
.. note::
	* We can have an orthonormal set of basis vectors (not necessarily unit-vectors) for a finite dimensional vector space :math:`V_{\mathcal{F}}` as

		.. math:: \{\mathbf{b}_1,\cdots\mathbf{b}_n\}
	* Any vector :math:`\mathbf{u}\in V_{\mathcal{F}}` then can be expressed in terms of inner product with the basis vectors and then taking a finite sum

		.. math:: \mathbf{u}=\langle \mathbf{u},\mathbf{b}_1\rangle\cdot||\mathbf{b}_1||_2+\cdots\langle \mathbf{u},\mathbf{b}_n\rangle\cdot||\mathbf{b}_n||_2=\sum_{i=1}^n \langle \mathbf{u},\mathbf{b}_i\rangle\cdot||\mathbf{b}_i||_2
	* For "well-behaved" (i.e. square-integrable so that one can define :math:`L_2` norm as per above) periodic functions, we can have `basis functions of odd and even frequencies <https://math.stackexchange.com/a/32663>`_.
	* `Schauder basis <https://en.wikipedia.org/wiki/Schauder_basis>`_ (allows for infinite sum over basis):

		* A basis for functions in :math:`L^2([0,1],\mathbb{R})` can be defined in terms of an infinite set of orthonormal functions`

			.. math:: \{1, (\sqrt{2}\sin(2\pi nx))_{n=1}^\infty, (\sqrt{2}\cos(2\pi nx))_{n=1}^\infty\}
		* The :math:`\sin` functions account for odd-frequencies and the :math:`\cos` functions account for even-frequencies.
		* We define the inner products as

Integral Transforms
--------------------------------------------------------------------------------

.. seealso::
	* `Functions are vectors <https://www.youtube.com/watch?v=LSbpQawNzU8>`_
	* `THE GEOMETRY OF MATHEMATICAL METHODS <https://books.physics.oregonstate.edu/GMM/complete.html>`_
