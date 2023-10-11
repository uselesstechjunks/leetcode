################################################################
Single Variable Calculus
################################################################
.. attention::
	All the variables are real and all the functions are real valued functions.

****************************************************************
Convergence and Continuity
****************************************************************
Convergent sequence
================================================================
Let :math:`(x_n)_{n=1}^\infty` be a sequence such that :math:`\forall x_n\in S\subset\mathbb{R}`. 

.. note::
	The sequence is said to be convergent to a limit :math:`x\in S` iff

	* :math:`\forall\delta > 0`
	* :math:`\exists N_\delta\in\mathbb{N}^{+}` such that
	* :math:`n \geq N_\delta\implies |x_n-x|\leq\delta`

Cauchy convergence
================================================================
.. note::
	The sequence is said to be Cauchy convergent iff

	* :math:`\forall\delta > 0`
	* :math:`\exists N_\delta\in\mathbb{N}^{+}` such that
	* :math:`m, n\geq N_\delta\implies |x_m-x_n|\leq\delta`

.. attention::
	* For a sequence to be Cauchy convergent, the limit value doesn't need to be in :math:`S`.
	* Example: We can imagine a sequence in rationals

		.. math:: 1,1.4,1.41,1.414,1.4142,1.41421,1.414213,1.4142135,1.41421356,1.414213562,1.4142135623,1.41421356237,\cdots

	* This sequence is Cauchy convergent as it tends to :math:`\sqrt(2)` but it's not convergent in :math:`\mathbb{Q}`.

Continuity
================================================================
Let :math:`f:X\subset\mathbb{R}\mapsto Y\subset\mathbb{R}`.

.. note::
	The set :math:`U=\{f(x)\mathop{|}x\in S\}=f(S)` is called the image of :math:`S` under :math:`f`.

.. note::
	The function :math:`f:X\mapsto Y` is said to be continuous at a point :math:`x\in X` iff

	* :math:`\forall\epsilon > 0`
	* :math:`\exists\delta_{\epsilon, x} > 0` (depends on :math:`\epsilon` as well as :math:`x` and can be arbitrarily small) such that
	* :math:`\forall y\in X, |x-y|\leq\delta_{\epsilon, x}\implies |f(x)-f(y)|\leq\epsilon`

.. seealso::
	* If we're only able to take extremely small :math:`\delta_{\epsilon, x}` to push the image inside the :math:`\epsilon` ball in :math:`Y`, then we can say that the function varies quite drastically.
	* If we're allowed to take larger :math:`\delta`, then the function is considered smoother.

.. tip::
	:math:`\lim\limits_{n\to\infty} x_n=x\in X\implies \lim\limits_{n\to\infty} f(x_n)=f(x)\in Y`

Properties
----------------------------------------------------------------
.. note::
	* If :math:`f` and :math:`g` are continuous at :math:`x`, so is :math:`f\cdot g`.
	* If :math:`f` and :math:`g` are continuous at :math:`x`, so is :math:`f\circ g`.

Continuous Everywhere
----------------------------------------------------------------
.. note::
	If the function is continuous :math:`\forall\in X`, then it is said to be continuous everywhere.

Uniform Continuity
----------------------------------------------------------------
This is a stricter form of continuity.

.. note::
	The function :math:`f:X\mapsto Y` is said to be uniformly continuous in :math:`X` iff

	* :math:`\forall\epsilon > 0`
	* :math:`\exists\delta_\epsilon > 0` (a universal one, as it doesn't depend on :math:`x` anymore, however can still be arbitrarily small) such that
	* :math:`\forall x, y\in X, |x-y|\leq\delta_\epsilon\implies |f(x)-f(y)|\leq\epsilon`

Lipschitz Continuity
----------------------------------------------------------------
This is an even stricter form of continuity.

.. note::
	The function :math:`f:X\mapsto Y` is said to be Lipschitz continuous in :math:`X` with Lipschitz constant :math:`K` iff

	* :math:`\exists K\geq 0` such that
	* :math:`\forall x,y\in X, \frac{|f(x)-f(x)|}{|x-y|}\leq K`

.. seealso::
	This means

	* :math:`\forall\epsilon > 0`
	* we can choose :math:`\delta=\epsilon/K` (able to take larger values now) such that
	* :math:`\forall x, y\in X, |x-y|\leq\delta\implies |f(x)-f(y)|\leq\epsilon`

****************************************************************
Differentiation
****************************************************************
.. warning::
	Let :math:`f:(a,b)\subset\mathbb{R}\mapsto \mathbb{R}` be a continuous function at some :math:`x\in(a,b)`.

.. note::
	The derivative of :math:`f` at :math:`x\in(a,b)` is defined to be (assuming that the limit exists),

		.. math:: f'(x)=\lim\limits_{h\to 0}\frac{f(x+h)-f(x)}{h}

.. warning::
	We need the point to be inside the open interval because we should be able to create an open ball around it for which the function is defined.

Properties
================================================================
.. note::
	* **Sum Rule**: :math:`(f+g)'=f'+g'`
	* **Product Rule**: :math:`(f\cdot g)'=f\cdot g'+f'\cdot g`
	* **Chain Rule**: :math:`(f\circ g)'=(f'\circ g)\cdot g'`

****************************************************************
Integration
****************************************************************

Integration of step functions
================================================================
.. warning::
	Let :math:`f:[a,b]\subset\mathbb{R}\mapsto \mathbb{R}` be a function.

Integration of general function
================================================================
.. note::
	

Properties
================================================================

****************************************************************
Important Theorems
****************************************************************
Bolzano's theorem
================================================================

Intermediate value theorem
================================================================

Mean value theorem
================================================================

****************************************************************
Important Formulae
****************************************************************
