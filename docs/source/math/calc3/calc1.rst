################################################################
Single Variable Calculus
################################################################
.. attention::
	* All the variables are real and all the functions are real valued functions.
	* We will be using **points** instead of **elements** since :math:`\mathbb{R}` is a metric space with a distance function defined in terms of the absolute value function :math:`|\cdot|`.
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
		
		* Formally, :math:`n > N_\delta\implies |x_n-x|< \delta`

Cauchy convergence
================================================================
.. note::
	The sequence is said to be Cauchy convergent iff

	* :math:`\forall\delta > 0`
	* :math:`\exists N_\delta\in\mathbb{N}^{+}` such that
	* if we skip :math:`N_\delta` number of terms in that sequence, any two values from the rest of it fall under a :math:`\delta`-ball around one another.
	
		* Formally, :math:`m, n> N_\delta\implies |x_m-x_n|< \delta`

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
	
		* Formally, :math:`\forall x\in X, |p-x|< \delta_{\epsilon, p}\implies |f(p)-f(x)|< \epsilon`

.. seealso::
	* If the function varies quite drastically, we'd only able to choose extremely small :math:`\delta_{\epsilon, p}` to push the image inside :math:`B_\epsilon(f(p))`.
	* If we're allowed to take larger :math:`\delta`, then the function is considered smoother.

Sequential Continuity
----------------------------------------------------------------
.. tip::
	Under a continuous function :math:`f`, :math:`\lim\limits_{n\to\infty} x_n=x\in X\implies \lim\limits_{n\to\infty} f(x_n)=f(x)\in Y`

Properties
----------------------------------------------------------------
.. note::
	* If :math:`f` and :math:`g` are continuous at :math:`x`, so is :math:`f\cdot g`.
	* If :math:`f` and :math:`g` are continuous at :math:`x`, so is :math:`f\circ g`.

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
	* we can create an open ball around **any** :math:`p` with some :math:`\exists\delta_\epsilon > 0` such that

		* (note: a universal one as it doesn't depend on :math:`p` anymore, however can still be arbitrarily small)
	* if we force :math:`x` to be in :math:`B_{\delta_\epsilon}(p)`, the image :math:`f(x)` is guaranteed to be in :math:`B_\epsilon(f(p))`.

		* Formally, :math:`\forall p, x\in X, |p-x|< \delta_\epsilon\implies |f(p)-f(x)|< \epsilon`

.. tip::
	* The same :math:`\delta` works for every point :math:`p\in X`, hence the term **uniform**.

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
	We need the point :math:`x` to be inside the open interval because we need to be able to create an open :math:`h`-ball around it and we need the function to be well defined in that region.

Properties
================================================================
.. note::
	* **Sum Rule**: :math:`(f+g)'=f'+g'`
	* **Product Rule**: :math:`(f\cdot g)'=f\cdot g'+f'\cdot g`
	* **Chain Rule**: :math:`(f\circ g)'=(f'\circ g)\cdot g'`

Differentiation as a linear approximation
================================================================
.. note::
	* Let :math:`D` denote an operator which, when applied to a differentiable function, :math:`f`, returns :math:`D(f)=Df` which represents another function :math:`Df:\mathbf{R}\mapsto\mathbf{R}`.
	* We can consider an error term

		.. math:: E(x,h)=f(x+h)-f(x)-Df(x)\cdot h
	* We note that :math:`\lim\limits_{h\to 0)E(x,h)=0\implies Df=f'`
	* We note that :math:`Df:\mathbf{R}\mapsto\mathbf{R}` is a linear transform because of the sum rule.
	* Therefore, :math:`Df` is a linear approximation of :math:`f` with an approximation error that becomes negligible as we move closer to :math:`x`.

Application: Local extremum
================================================================
.. note::
	* Let the function be :math:`f:X\mapsto Y` and let :math:`c\in X`.
	* :math:`c` is called a relative (local) maximum iff

		.. math:: \exists\epsilon>0,x\in B_\epsilon(c)\implies f(x)\leq f(c)
.. note::
	* Relative minimum is defined in the same way.
	* This is usually defined in terms of an open interval, i.e. :math:`c\in(a,b)`.
	* Maxima and minimum are jointly called an extremum.

First derivative test
----------------------------------------------------------------
For critical points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. attention::
	Let :math:`c\in(a,b)` be a local extremum. Then :math:`f'(c)=0`.

.. tip::
	* The point :math:`c\in(a,b)` is called a **critical point**.
	* First derivative test doesn't tell us whether it's a maximum or a minimum.

For monotonic functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. attention::
	* If :math:`\forall x\in (a,b), f'(x)> 0`, then :math:`f` is strictly increasing in :math:`[a,b]`.
	* If :math:`\forall x\in (a,b), f'(x)< 0`, then :math:`f` is strictly decreasing in :math:`[a,b]`.
	* If :math:`\forall x\in (a,b), f'(x)= 0`, then :math:`f` is constant in :math:`[a,b]`.

Second derivative test
----------------------------------------------------------------
For critical points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. tip::
	Think of the slope of tangent for a convex function as it reaches a minimum.

.. attention::
	* For a minimum :math:`c\in(a,b)`, the second derivative is a strictly increasing function in :math:`[a,b]`, i.e. :math:`\forall x\in(a,b), f''(x)> 0`.
	* For a maximum :math:`c\in(a,b)`, the second derivative is a strictly decreasing function in :math:`[a,b]`, i.e. :math:`\forall x\in(a,b), f''(x)< 0`.

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
	* We define :math:`S=\left\{\int\limits_a^b s(x)\mathop{dx}\mathop{:}\forall s\leq f\right\}` and :math:`T=\left\{\int\limits_a^b t(x)\mathop{dx}\mathop{:}\forall t\geq f\right\}`.
	* It is in general true that

		.. math:: \int\limits_a^b s(x)\mathop{dx}\leq\sup_s S\leq I\leq\inf_t T\leq \int\limits_a^b t(x)\mathop{dx}
	* The integral :math:`I` exists when :math:`\sup_s S=\inf_t T` and takes that exact same value 

		.. math:: I=\int\limits_a^b f(x)\mathop{dx}=\sup_s S=\inf_t T

.. attention::
	* Let :math:`f:[a,b]\mapsto\mathbb{R}` is continuous :math:`\forall x\in[a,b]`.
	* Then it is integrable (follows since it is bounded).

Properties
----------------------------------------------------------------
.. note::
	* :math:`\int\limits_a^b (f(x)+g(x))\mathop{dx}=\int\limits_a^b f(x)\mathop{dx}=+\int\limits_a^b g(x)\mathop{dx}`
	* :math:`\int\limits_a^b c\cdot f(x)\mathop{dx}=c\cdot\int\limits_a^b f(x)\mathop{dx}`
	* :math:`\int\limits_a^b f(x)\mathop{dx}=-\int\limits_b^a f(x)\mathop{dx}`
	* :math:`\int\limits_a^c f(x)\mathop{dx}=\int\limits_a^b f(x)\mathop{dx}+\int\limits_b^c f(x)\mathop{dx}`
	* :math:`\int\limits_a^a f(x)\mathop{dx}=0`

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
Boundedness theorem
================================================================
.. note::
	* Let :math:`f:[a,b]\mapsto\mathbb{R}` is continuous :math:`\forall x\in[a,b]`. Then it is bounded.

		* There exists :math:`m, M\in\mathbb{R}` such that :math:`m\leq f(x)\leq M`.

EVT: Extreme value theorem
================================================================
.. note::
	* Let :math:`f:[a,b]\mapsto\mathbb{R}` is continuous :math:`\forall x\in[a,b]`. Then the function achives a min and a max.

		* There exists :math:`c,d\in[a,b]` such that :math:`f(c)\leq f(x)\leq f(d)`.

Fundamental theorem of calculus
================================================================
.. note::
	* Let :math:`f:[a,b]\mapsto\mathbb{R}` be a function that is integrable for every :math:`[a,x]`. Let :math:`c\in[a,b]`.
	* Let :math:`F(x)` be an indefinite integral of :math:`f`

		.. math:: F(x)=\int\limits_c^x f(t)\mathop{dt}
	* Then the derivative of :math:`F` exists at all :math:`x\in(a,b)` wherever :math:`f(x)` is continuous and

		.. math:: F'(x)=f(x)

.. tip::
	* :math:`F` is called an **antiderivative** of :math:`f`.
	* Any other antiderivative differs only by a constant.
	* **Leibniz Notation**: Therefore, we can use the notation

		.. math:: \int f(x)\mathop{dx}=F(x)+C

.. attention::
	.. math:: \int\limits_a^b f(x)\mathop{dx}=F(b)-F(a)

Integration by parts
================================================================
Let :math:`u(x)` and :math:`v(x)` be two integrable functions. We want to find out the integral of the product, :math:`\int u(x)\cdot v(x) \mathop{dx}`.

.. note::
	* To derive this formula, it becomes easier if we consider :math:`w(x)=\int v(x) \mathop{dx}` (such that :math:`w'(x)=v(x)`) and consider :math:`g(x)=u(x)\cdot w(x)`.
	* Taking derivatives on both sides :math:`g'(x)=u'(x)\cdot w(x)+u(x)\cdot w'(x)` which gives

		.. math:: u(x)\cdot w'(x)=g'(x)-u'(x)\cdot w(x)
	* Taking integration on both sides and ignoring the constant

		.. math:: \int u(x)\cdot w'(x)\mathop{dx}=\int g'(x)\mathop{dx}-\int u'(x)\cdot w(x)\mathop{dx}=u(x)\cdot w(x)-\int u'(x)\cdot w(x)\mathop{dx}
	* Replacing :math:`w(x)`

		.. math:: \int u(x)\cdot v(x)\mathop{dx}=u(x)\cdot \int v(x)\mathop{dx}-\int u'(x)\left(\int v(x)\mathop{dx}) \right)\mathop{dx}

.. tip::
	* ILATE: Dictates the order in which the functions should be chosen to be :math:`u` or :math:`v`. 
	* ILATE: Acronym for Inverse > Logarithmic > Algebraic > Trigonometric > Exponential. Choose left of the two as :math:`u`.

Bolzano's theorem
================================================================
.. note::
	* Let :math:`f:[a,b]\mapsto\mathbb{R}` is continuous :math:`\forall x\in[a,b]`.
	* Also assume that :math:`f(a)` and :math:`f(b)` have opposite signs.
	* Then :math:`\exists c\in(a,b)` such that :math:`f(c)=0`

IVT: Intermediate value theorem
================================================================
.. note::
	* Let :math:`f:[a,b]\mapsto\mathbb{R}` is continuous :math:`\forall x\in[a,b]`.
	* Let :math:`a\leq p < q\leq b` be two arbitrary points with :math:`f(p)\neq f(q)`.
	* Then :math:`f(x)` takes every possible value in :math:`(f(p), f(q))` within the interval :math:`(a,b)`.

MVT: Mean value theorem
================================================================
.. note::
	* Let :math:`f:[a,b]\mapsto\mathbb{R}` is continuous :math:`\forall x\in[a,b]`.
	* Then :math:`\exists c\in[a,b]` such that :math:`f(c)` acts as the mean value of the integral :math:`\int\limits_a^b f(x)\mathop{dx}`.
	* Formally, :math:`\int\limits_a^b f(x)\mathop{dx}=f(c)\cdot(b-a)`

.. seealso::
	* This can also be stated using derivatives as :math:`\frac{F(b)-F(a)}{b-a}=f(c)` or :math:`\frac{g(b)-g(a)}{b-a}=g'(c)`

Rolle's theorem
----------------------------------------------------------------
.. note::
	* Special case of MVT.
	* Assuming that all the MVT conditions are satisfied, if :math:`f(a)=f(b)`, then :math:`\exists c\in(a,b)` such that :math:`f'(c)=0`.

****************************************************************
Useful Resources
****************************************************************
.. seealso::
	* Calculus cheatsheet: `Notes at tutorial.math.lamar.edu <https://tutorial.math.lamar.edu/pdf/calculus_cheat_sheet_all.pdf>`_.
	* Different ways for evaluating the Gaussian integral: `YouTube video playlist by Dr Peyam <https://www.youtube.com/watch?v=HcneBkidSDQ&list=PLJb1qAQIrmmCgLyHWMXGZnioRHLqOk2bW>`_.

		* Hints (one way): Let :math:`I=\int\limits_{-\infty}^\infty e^{x^2}\mathop{dx}`. Try to compute :math:`I^2`, convert this into a double integral using Fubini's theorem, and then use polar co-ordinate transform.
