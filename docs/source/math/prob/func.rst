##########################################################################################
Functions of Random Variable
##########################################################################################

******************************************************************************************
Mass/density of a function of a rv
******************************************************************************************
Let :math:`Y=g(X)` be a function of an rv :math:`X`.

.. note::
	* If :math:`X` is discrete, this is discussed in the random variable section (TODO: add hyperlink)
	* If :math:`X` is continuous with a PDF :math:`f_X(x)`, then the process for finding :math:`f_Y(y)` is as follows:

		* Compute the CDF :math:`F_Y(y)=\mathbb{P}(Y\leq y)=\mathbb{P}(g(X)\leq y)=\int\limits_{\{x|g(x)\leq y\}}f_X(x) dx`.
		* Compute the PDF as :math:`f_Y(y)=F'_Y(y)`.

.. tip::
	Some effort is required to compute the set :math:`\{x|g(x)\leq y\}`.

Special cases
========================================================================
Linear functions
------------------------------------------------------------------------
Let :math:`Y=g(X)=aX+b` with :math:`a\neq 0`. Therefore we have 

.. math::
	F_Y(y)=\mathbb{P}(Y\leq y)=\mathbb{P}(g(X)\leq y)=\mathbb{P}(aX+b\leq y)=\begin{cases}\mathbb{P}\left(X\leq\frac{y-b}{a}\right) & \text{if $a>0$} \\ \mathbb{P}\left(X\geq\frac{y-b}{a}\right) & \text{if $a<0$}\end{cases}=\begin{cases}f_X(\frac{y-b}{a}) & \text{if $a>0$} \\ 1-f_X(\frac{y-b}{a}) & \text{if $a<0$}\end{cases}

.. tip::
	The PDF in this case has the format

		.. math:: f_Y(y)=\begin{cases}\frac{1}{a}f_X(\frac{y-b}{a}) & \text{if $a>0$} \\ -\frac{1}{a}f_X(\frac{y-b}{a}) & \text{if $a<0$}\end{cases}=\frac{1}{\left| a \right|}f_X(\frac{y-b}{a})

Monotonic functions
------------------------------------------------------------------------

Sum of rvs: Convolution
========================================================================
.. tip::
	* We want the PDF (or PMF) of the sum of two rvs, :math:`X` and :math:`Y`, :math:`Z=X+Y`.
	* Let the joint density be :math:`f_{X,Y}(x,y)` (alternatively, joint mass :math:`p_{X,Y}(x,y)`).

		.. math::
			\begin{cases}
				p_Z(z)=\sum_{x=-\infty}^\infty p_X(x) p_Y(z-x)=(p_X \ast p_Y)[z] & \text{if $X$ and $Y$ are discrete with PMF $p_X$ and $p_Y$}\\
				f_Z(z)=\int\limits_{-\infty}^\infty f_X(x) f_Y(z-x) dx=(f_X \ast f_Y)[z] & \text{if $X$ and $Y$ are continuous with PDF $f_X$ and $f_Y$}\\
			\end{cases}

******************************************************************************************
Covariance and correlation
******************************************************************************************

******************************************************************************************
Fundamentals of estimations using conditional expectation
******************************************************************************************

Law of iterated expectation
========================================================================

Conditional variance
========================================================================

Law of iterated variance
========================================================================

******************************************************************************************
Transforms of rv
******************************************************************************************

Moment Generating Functions
========================================================================


