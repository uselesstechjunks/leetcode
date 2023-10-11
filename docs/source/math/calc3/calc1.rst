################################################################
Single Variable Calculus
################################################################
.. attention::
	* All the variables are real and all the functions are real valued functions.
	* We will be using **point** instead of **elements** since :math:`\mathbb{R}` is a metric space with a distance function defined in terms of absolute function.
	* The set :math:`U=\{f(x)\mathop{:}x\in S\}=f(S)` is called the image of :math:`S` under :math:`f`.

****************************************************************
Convergence and Continuity
****************************************************************
Open Ball
================================================================
Around any point we can create an open :math:`\epsilon`-ball

.. math:: B_\epsilon(x)=\{y\mathop{|} |x-y|< \epsilon\}

Convergent sequence
================================================================
Let :math:`(x_n)_{n=1}^\infty` be a sequence such that :math:`\forall x_n\in S\subset\mathbb{R}`. 

.. note::
	The sequence is said to be convergent to a limit :math:`x\in S` iff

	* :math:`\forall\delta > 0`
	* :math:`\exists N_\delta\in\mathbb{N}^{+}` (depends on how small of a :math:`\delta` we're given) such that
	* if we skip :math:`N_\delta` number of terms in that sequence, the remaining values are guaranteed to be inside :math:`B_\delta(x)`.
		
		* Formally, :math:`n \geq N_\delta\implies |x_n-x|\leq\delta`

Cauchy convergence
================================================================
.. note::
	The sequence is said to be Cauchy convergent iff

	* :math:`\forall\delta > 0`
	* :math:`\exists N_\delta\in\mathbb{N}^{+}` such that
	* if we skip :math:`N_\delta` number of terms in that sequence, any two values fall under a :math:`\delta`-ball around one another.
	
		* Formally, :math:`m, n\geq N_\delta\implies |x_m-x_n|\leq\delta`

.. attention::
	* For a sequence to be Cauchy convergent, the limit value doesn't need to be in :math:`S`.
	* Example: We can imagine a sequence in rationals

		.. math:: 1,1.4,1.41,1.414,1.4142,1.41421,1.414213,\cdots

	* This sequence is Cauchy convergent as it tends to :math:`\sqrt{2}` but it's not convergent in :math:`\mathbb{Q}`.

Continuity
================================================================
Let :math:`f:X\subset\mathbb{R}\mapsto Y\subset\mathbb{R}`.

.. note::
	The function :math:`f:X\mapsto Y` is said to be continuous at a point :math:`p\in X` iff

	* :math:`\forall\epsilon > 0`
	* we can create an open ball around :math:`p` in the domain with some :math:`\delta_{\epsilon, p} > 0` such that

		* (note: the size depends on :math:`\epsilon` as well as :math:`p` and can be arbitrarily small)
	* if we force :math:`x` to be in :math:`B_{\delta_{\epsilon, p}}(p)`, then the image :math:`f(x)` is guaranteed to be in :math:`B_\epsilon(f(p))`.
	
		* Formally, :math:`\forall x\in X, |p-x|\leq\delta_{\epsilon, p}\implies |f(p)-f(x)|\leq\epsilon`

.. seealso::
	* If we're only able to take extremely small :math:`\delta_{\epsilon, p}` to push the image inside the :math:`\epsilon`-ball in :math:`Y`, then we can say that the function varies quite drastically.
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
	* we can create an open ball around :math:`p` in the domain with some :math:`\exists\delta_\epsilon > 0` such that

		* (note: a universal one as it doesn't depend on :math:`p` anymore, however can still be arbitrarily small)
	* if we force :math:`x` to be in :math:`B_{\delta_\epsilon}(p)` around **any** :math:`p`, the image :math:`f(x)` is guaranteed to be in :math:`B_\epsilon(f(p))`.

		* Formally, :math:`\forall p, x\in X, |p-x|\leq\delta_\epsilon\implies |f(p)-f(x)|\leq\epsilon`

.. tip::
	* The same :math:`\delta` works for every :math:`\epsilon`, hence the term **uniform**.

Lipschitz Continuity
----------------------------------------------------------------
This is an even stricter form of continuity.

.. note::
	The function :math:`f:X\mapsto Y` is said to be Lipschitz continuous in :math:`X` with Lipschitz constant :math:`K` iff

	* :math:`\exists K\geq 0` such that :math:`\forall x,y\in X, \frac{|f(x)-f(x)|}{|x-y|}\leq K`

.. seealso::
	* For the image to be in a :math:`\epsilon`-ball around any :math:`p`, we can afford to be in a :math:`\epsilon/K`-ball in the domain.
	* These functions are a lot smoother.

****************************************************************
Differentiation
****************************************************************
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
Let :math:`f:[a,b]\subset\mathbb{R}\mapsto \mathbb{R}` be a step-function defined on a partition :math:`P=\{x_0,\cdots,x_n\}` such that within each open interval :math:`(x_{k-1},x_k)`, the function takes a constant value :math:`s_k`.

.. note::
	The integral of such function is defined as

		.. math:: \int\limits_a^b f(x)\mathop{dx}=\sum_{k=1}^n s_k\cdot(x_k-x_{k-1})

Properties
----------------------------------------------------------------
.. note::
	* If :math:`f(x)<g(x)` for all :math:`x\in[a,b]`, then :math:`\int\limits_a^b f(x)\mathop{dx}<\int\limits_a^b g(x)\mathop{dx}`.

Integration of general function
================================================================
Let :math:`f:[a,b]\subset\mathbb{R}\mapsto \mathbb{R}` be a general function.

.. note::
	TODO

Properties
----------------------------------------------------------------

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