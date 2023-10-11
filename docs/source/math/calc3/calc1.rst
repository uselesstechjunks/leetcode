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
Let :math:`f:X\subset\mathbb{R}\mapsto\Y\subset\mathbb{R}`.

.. note::
	The set :math:`U=\{f(x)\mathop{|}x\in S\}=f(S)` is called the image of :math:`S` under :math:`f`.

.. note::
	The function :math:`f:X\mapsto Y` is said to be continuous at a point :math:`x\in X` iff

	* :math:`\forall\epsilon > 0`
	* :math:`\exists\delta_\epsilon > 0` such that
	* :math:`\forall y\in X, |x-y|\leq\delta_epsilon\implies |f(x)-f(y)|\leq\epsilon`

.. tip::
	:math:`\lim\limits_{n\to\infty} x_n=x\in X\implies \lim\limits_{n\to\infty} f(x_n)=f(x)`

****************************************************************
Differentiation
****************************************************************
Properties
================================================================

****************************************************************
Integration
****************************************************************
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
