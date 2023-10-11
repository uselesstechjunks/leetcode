################################################################
Single Variable Calculus
################################################################
.. attention::
	* All the variables are real and all the functions are real valued functions.
	* We will be using **point** instead of **elements** since :math:`\mathbb{R}` is a metric space with a distance function defined in terms of the absolute value function :math:`|\cdot|`.
	* The set :math:`U=\{f(x)\mathop{:}x\in S\}=f(S)` is called the image of :math:`S` under :math:`f`.

****************************************************************
Convergence and Continuity
****************************************************************
Open and Closed Balls and Intervals
================================================================
.. note::
	* For any :math:`\epsilon > 0`, we can create an open :math:`\epsilon`-ball around any point :math:`x` as

		.. math:: B_\epsilon(x)=\{y\mathop{:} |x-y|< \epsilon\}
	* Closed ball is defined similarly as 

		.. math:: \bar{B}_\epsilon(x)=\{y\mathop{:} |x-y|\leq \epsilon\}
	* Open interval: :math:`(a,b)=\{x\mathop{:} a < x < b\}`
	* Closed interval: :math:`[a,b]=\{x\mathop{:} a \leq x \leq b\}`

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
	* we can create an open ball around :math:`p` with some :math:`\delta_{\epsilon, p} > 0` such that

		* (note: the size depends on :math:`\epsilon` as well as :math:`p` and can be arbitrarily small)
	* if we force :math:`x` to be in :math:`B_{\delta_{\epsilon, p}}(p)`, then the image :math:`f(x)` is guaranteed to be in :math:`B_\epsilon(f(p))`.
	
		* Formally, :math:`\forall x\in X, |p-x|\leq\delta_{\epsilon, p}\implies |f(p)-f(x)|\leq\epsilon`

.. seealso::
	* If we're only able to take extremely small :math:`\delta_{\epsilon, p}` to push the image inside the :math:`\epsilon`-ball in :math:`Y`, then we can say that the function varies quite drastically.
	* If we're allowed to take larger :math:`\delta`, then the function is considered smoother.

.. tip::
	Under a continuous function :math:`f`, :math:`\lim\limits_{n\to\infty} x_n=x\in X\implies \lim\limits_{n\to\infty} f(x_n)=f(x)\in Y`

Properties
----------------------------------------------------------------
.. note::
	* If :math:`f` and :math:`g` are continuous at :math:`x`, so is :math:`f\cdot g`.
	* If :math:`f` and :math:`g` are continuous at :math:`x`, so is :math:`f\circ g`.

.. attention::
	* **Boundedness Theorem**: Let :math:`f:[a,b]\mapsto\mathbb{R}` is continuous :math:`\forall x\in[a,b]`. then it is bounded.

Continuous Everywhere
----------------------------------------------------------------
.. note::
	If the function is continuous :math:`\forall p\in X`, then it is said to be continuous everywhere.

Uniform Continuity
----------------------------------------------------------------
This is a stricter form of continuity.

.. note::
	The function :math:`f:X\mapsto Y` is said to be uniformly continuous in :math:`X` iff

	* :math:`\forall\epsilon > 0`
	* we can create an open ball around :math:`p` with some :math:`\exists\delta_\epsilon > 0` such that

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
.. warning::
	* We try to approximate the integral :math:`I=\int\limits_a^b f(x)\mathop{dx}` by 2 step functions :math:`s` and :math:`t`, one above and one below, that is

		.. math:: s(x)\leq f(x)\leq t(x)
	* This is not possible if the function :math:`f` is unbounded (such as :math:`f(x)=1/x`).

Let :math:`f:[a,b]\subset\mathbb{R}\mapsto \mathbb{R}` be any bounded function.

.. note::
	* Let :math:`s` and :math:`t` be arbitrary step functions such that :math:`s(x)\leq f(x)\leq t(x)`.
	* We define :math:`S=\left\{\int\limits_a^b s(x)\mathop{dx}\mathop{:}\forall s\leq f\right\}` and :math:`T=\left\{\int\limits_a^b t(x)\mathop{dx}\mathop{:}\forall f\leq t\right\}`.
	* It is in general true that :math:`\int\limits_a^b s(x)\mathop{dx}\leq\sup_s S\leq I\leq\inf_t T\leq \int\limits_a^b t(x)\mathop{dx}`
	* The integral :math:`I` exists when :math:`\sup_s S=\inf_t T` and takes that exact same value 

		.. math:: I=\int\limits_a^b f(x)\mathop{dx}=\sup_s S=\inf_t T

.. attention::
	* Let :math:`f:[a,b]\mapsto\mathbb{R}` is continuous :math:`\forall x\in[a,b]`.
	* Then it is integrable (follows since it is bounded).

Properties
----------------------------------------------------------------
..note::
	* :math:`\int\limits_a^b (f(x)+g(x))\mathop{dx}=\int\limits_a^b f(x)\mathop{dx}=+\int\limits_a^b g(x)\mathop{dx}`
	* :math:`\int\limits_a^b c\cdot f(x)\mathop{dx}=c\cdot\int\limits_a^b f(x)\mathop{dx}`
	* :math:`\int\limits_a^b f(x)\mathop{dx}=-\int\limits_b^a f(x)\mathop{dx}`
	* :math:`\int\limits_a^c f(x)\mathop{dx}=\int\limits_a^b f(x)\mathop{dx}+\int\limits_b^c f(x)\mathop{dx}`

Indefinite Integral
================================================================
.. note::
	* For every :math:`a\leq x\leq b`, we can define a function of :math:`x` which is obtained via the integral

		.. math:: A(x)=\int\limits_a^x f(t)\mathop{dt}

		* This is known as **an** indefinite integral of :math:`f`.
	* We can define another indefinite integral with a different lower limit :math:`c\in[a,b]` as

		.. math:: C(x)=\int\limits_c^x f(t)\mathop{dt}

.. attention::
	* These two differ by only a constant as

		.. math:: A(x)-C(x)=\int\limits_a^x f(t)\mathop{dt}-\int\limits_c^x f(t)\mathop{dt}=\int\limits_a^c f(t)\mathop{dt}=k

****************************************************************
Important Theorems
****************************************************************
Bolzano's theorem
================================================================
.. note::
	* Let :math:`f:[a,b]\mapsto\mathbb{R}` is continuous :math:`\forall x\in[a,b]`.
	* Also assume that :math:`f(a)` and :math:`f(b)` have opposite signs.
	* Then :math:`\exists c\in(a,b)` such that :math:`f(c)=0`

Intermediate value theorem
================================================================
.. note::
	* Let :math:`f:[a,b]\mapsto\mathbb{R}` is continuous :math:`\forall x\in[a,b]`.
	* Let :math:`a\leq p < q\leq b` be two arbitrary points with :math:`f(p)\neq f(q)`.
	* Then :math:`f(x)` takes every possible value in :math:`(f(p), f(q))` within the interval :math:`(a,b)`.

Mean value theorem
================================================================

****************************************************************
Important Formulae
****************************************************************
