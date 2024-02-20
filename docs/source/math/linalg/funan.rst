################################################################################
Functional Analysis
################################################################################

********************************************************************************
Vector Space of Functions
********************************************************************************
Functions are vectors : Intuition
================================================================================
.. tip::
	* We usually think of vectors in finite dimensional case, e.g. :math:`\mathbf{u}\in\mathbb{R}^4`, as a list

		.. math:: \mathbf{u}=\begin{bmatrix}0.3426 \\1.3258 \\6.8943 \\8.3878 \\\end{bmatrix}\\
	* An alternate way to look at this is as a list of tuples, which binds the dimension (integer index in this case) and the value of that dimension

		.. math:: \mathbf{u}=\left((0,0.3426),(1,1.3258),(2,6.8943),(3,8.387)\right)
	* This can be represented by a function

		.. math:: f:[0,1,2,3]\mapsto[0.3426,1.3258,6.8943,8.387]
	* If we extend the index set to have real values, then, theoretically, we end up with infinite lists

		.. math:: \mathbf{u}=\{(x,y)|x,y\in\mathbb{R}\}
	* This is the way functions are defined in set theory.

Cardinality of Function Space
--------------------------------------------------------------------------------
.. warning::
	* We might be tempted by the tuple definition to conclude that the cardinality of function space is :math:`|\mathcal{X}|\times|\mathcal{Y}|`.
	* However, this is incorrect, as functions allow multiple points to map to the same element in the domain.

.. note::
	* The space of functions :math:`f:\mathcal{X}\mapsto\mathcal{Y}` is often denoted by :math:`\mathcal{Y}^{\mathcal{X}}`.

		* For every element :math:`x\in\mathcal{X}`, we have a choice to make its image map to some :math:`y\in\mathcal{Y}`.
		* Therefore, the size of this choice for each :math:`x\in\mathcal{X}` is :math:`|\mathcal{Y}|`.
		* The size of the choice for **all** the elements in :math:`\mathcal{X}` is therefore :math:`|\mathcal{Y}|^{|\mathcal{X}|}`.
	* To get some intuition about the notation, we can think of all possible functions

		.. math:: f:\{x_1,x_2,x_3\}\mapsto\{y_1,y_2\}

		.. csv-table:: 
			:align: center

			:math:`f_1:\{x_1\mapsto y_1;x_2\mapsto y_1;x_3\mapsto y_1\}`
			:math:`f_2:\{x_1\mapsto y_1;x_2\mapsto y_1;x_3\mapsto y_2\}`
			:math:`f_3:\{x_1\mapsto y_1;x_2\mapsto y_2;x_3\mapsto y_1\}`
			:math:`f_4:\{x_1\mapsto y_1;x_2\mapsto y_2;x_3\mapsto y_2\}`
			:math:`f_5:\{x_1\mapsto y_2;x_2\mapsto y_1;x_3\mapsto y_1\}`
			:math:`f_6:\{x_1\mapsto y_2;x_2\mapsto y_1;x_3\mapsto y_2\}`
			:math:`f_7:\{x_1\mapsto y_2;x_2\mapsto y_2;x_3\mapsto y_1\}`
			:math:`f_8:\{x_1\mapsto y_2;x_2\mapsto y_2;x_3\mapsto y_2\}`

Addition and Scalar Multiplication
--------------------------------------------------------------------------------
.. note::
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
.. note::
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
.. note::
	* We can have an orthonormal set of basis vectors (not necessarily unit-vectors) for a finite dimensional vector space :math:`V_{\mathcal{F}}` as

		.. math:: \{\mathbf{b}_1,\cdots\mathbf{b}_n\}
	
		* For any vector :math:`\mathbf{u}`, we can find the proejection of it onto the basis vectors as :math:`\langle\mathbf{u},\mathbf{b}_i\rangle`.
		* The length of the basis vectors are given by :math:`||\mathbf{b}_i||_2^2=\langle\mathbf{b}_i,\mathbf{b}_i\rangle`.
		* Let :math:`a_i=\frac{\langle\mathbf{u},\mathbf{b}_i\rangle}{\langle\mathbf{b}_i,\mathbf{b}_i\rangle}` be the projection normalised for the length of the basis vector :math:`\mathbf{b}_i`.
		* :math:`\mathbf{u}` then can be expressed as

			.. math:: \mathbf{u}=a_1\cdot\mathbf{b}_i+\cdots a_n\cdot\mathbf{b}_n=\sum_{i=1}^na_i\cdot\mathbf{b}_i
		* We note that this results in the same expression if we convert each basis to a unit vector by normalising it, :math:`\mathbf{e}_i=\frac{\mathbf{b}_i}{\langle\mathbf{b}_i,\mathbf{b}_i\rangle}`

			.. math:: \mathbf{u}=\langle\mathbf{u},\mathbf{e}_1\rangle+\cdots\langle\mathbf{u},\mathbf{e}_n\rangle=\sum_{i=1}^n\langle\mathbf{u},\mathbf{e}_i\rangle

		* [Operator view]: We can define :math:`a_i(\cdot)=\frac{\langle\cdot,\mathbf{b}_i\rangle}{\langle\mathbf{b}_i,\mathbf{b}_i\rangle}` as an operator which can take any vector :math:`\mathbf{u}` and computes the projection onto it, :math:`(a_i)(\mathbf{u})=\frac{\langle\mathbf{u},\mathbf{b}_i\rangle}{\langle\mathbf{b}_i,\mathbf{b}_i\rangle}`

			.. math:: \mathbf{u}=\sum_{i=1}^n(a_i)(\mathbf{u})

Fourier Basis
--------------------------------------------------------------------------------
.. note::
	* For "well-behaved" (i.e. square-integrable so that one can define :math:`L_2` norm as per above) periodic functions, we can have `basis functions of odd and even frequencies <https://math.stackexchange.com/a/32663>`_.
	* `Schauder basis <https://en.wikipedia.org/wiki/Schauder_basis>`_ (allows for infinite sum over basis):

		* A basis for functions in :math:`L^2([0,1],\mathbb{R})` can be defined in terms of an infinite set of orthonormal functions`

			.. math:: \{1, (\sqrt{2}\sin(2\pi nx))_{n=1}^\infty, (\sqrt{2}\cos(2\pi nx))_{n=1}^\infty\}
		* The :math:`\sin` functions account for odd-frequencies and the :math:`\cos` functions account for even-frequencies.
	* Here we have 3 sets of basis functions, so we use 3 different kinds of normalised-projection co-efficients, :math:`a_0,a_i,b_i`

		.. math:: f(x)=a_0\cdot1+\sum_{n=1}^\infty a_i\cdot\cos(2\pi nx)+\sum_{n=1}^\infty b_i\cdot\sin(2\pi nx)
	* :math:`a_0` computes the projection of :math:`f(x)` onto the constant function :math:`1`.

		.. math:: a_0=\frac{\int_\limits{[0,1]}1\cdot f(x)\mathop{dx}}{\int_\limits{[0,1]}1\cdot 1\mathop{dx}}=\int_\limits{[0,1]}f(x)\mathop{dx}
	* For each :math:`k>0`, :math:`a_k` computes the projection of :math:`f(x)` onto the even frequencies, :math:`\sqrt{2}\cos(2\pi nx)`.

		.. math:: a_k=\frac{\int_\limits{[0,1]}f(x)\cdot\sqrt{2}\cos(2\pi kx)\mathop{dx}}{\int_\limits{[0,1]}\sqrt{2}\cos(2\pi kx)\cdot\sqrt{2}\cos(2\pi kx)\mathop{dx}}
	* Similarly, for :math:`b_k`.

More Basis - Integral Transforms
--------------------------------------------------------------------------------

.. seealso::
	* `Functions are vectors <https://www.youtube.com/watch?v=LSbpQawNzU8>`_
	* `THE GEOMETRY OF MATHEMATICAL METHODS <https://books.physics.oregonstate.edu/GMM/complete.html>`_
	* `Math 353 Lecture Notes Fourier series <https://services.math.duke.edu/~jtwong/math353-2020/lectures/Lec12-Fourier.pdf>`_
	* `[MIT] 9.520 Math Camp 2010 Functional Analysis Review <https://www.mit.edu/~9.520/spring10/Classes/mathcamp2010-fa-notes.pdf>`_
	* `SO post about Dirac delta being a generalized function instead of norma function <https://math.stackexchange.com/a/285643>`_
