##########################################################################################
Functions of Random Variable
##########################################################################################

******************************************************************************************
Density of a function of a rv
******************************************************************************************
Let :math:`Y=g(X)` be a function of an rv :math:`X`.

.. note::
	* If :math:`X` is discrete, this is discussed in the random variable section (TODO: add hyperlink)
	* If :math:`X` is continuous with a PDF :math:`f_X(x)`, then the process for finding :math:`f_Y(y)` is as follows:

		* Compute the CDF as

			.. math:: F_Y(y)=\mathbb{P}(Y\leq y)=\mathbb{P}(g(X)\leq y)=\int\limits_{\{x|g(x)\leq y\}}f_X(x) dx
		* Compute the PDF as :math:`f_Y(y)=F'_Y(y)`.

.. tip::
	Some effort is required to compute the set :math:`\{x|g(x)\leq y\}`.

Special cases
========================================================================
Linear functions
------------------------------------------------------------------------
Let :math:`Y=g(X)=aX+b` with :math:`a\neq 0`. Therefore we have 

.. tip::
	* If :math:`a=0`, then :math:`Y=b` with probability 1 and it's no longer a continuous rv.
	* If :math:`a\neq 0`, then we have

		.. math:: F_Y(y)=\mathbb{P}(Y\leq y)=\mathbb{P}(g(X)\leq y)=\mathbb{P}(aX+b\leq y)=\begin{cases}\mathbb{P}\left(X\leq\frac{y-b}{a}\right) & \text{if $a>0$} \\ \mathbb{P}\left(X\geq\frac{y-b}{a}\right) & \text{if $a<0$}\end{cases}=\begin{cases}F_X(\frac{y-b}{a}) & \text{if $a>0$} \\ 1-F_X(\frac{y-b}{a}) & \text{if $a<0$}\end{cases}
	* We can recover the PDF in both cases as

		.. math:: f_Y(y)=\begin{cases}\frac{1}{a}f_X(\frac{y-b}{a}) & \text{if $a>0$} \\ -\frac{1}{a}f_X(\frac{y-b}{a}) & \text{if $a<0$}\end{cases}=\frac{1}{\left| a \right|}f_X(\frac{y-b}{a})

Monotonic functions
------------------------------------------------------------------------
.. note::
	* If :math:`g(Y)=X` is a monotonic function, then it has an inverse, :math:`g^{-1}(Y)`.
	* Therefore, we have

		.. math:: F_Y(y)=\mathbb{P}(Y\leq y)=\mathbb{P}(g(X)\leq y)=\begin{cases}\mathbb{P}(X\leq g^{-1}(y)) & \text{if $g(X)$ is monotonic increasing}\\\mathbb{P}(X\geq g^{-1}(y)) & \text{if $g(X)$ is monotonic decreasing}\end{cases}=\begin{cases}F_X(g^{-1}(y)) & \text{if $g(X)$ is monotonic increasing}\\1-F_X(g^{-1}(y)) & \text{if $g(X)$ is monotonic decreasing}\end{cases}
	* We can recover the PDF in both cases as

		.. math:: f_Y(y)=\left| f_X(g^{-1}(y)) \right|\cdot\frac{d}{dy}\left[g^{-1}(y)\right]

******************************************************************************************
Density of a function of multiple jointly distributed rvs
******************************************************************************************
Let :math:`Z=g(X,Y)` be a function of 2 jointly distributed rvs, :math:`X` and :math:`Y`. In this case, we follow the same process as before.

.. tip::
	* Compute the CDF as

		.. math:: F_Z(z)=\mathbb{P}(Z\leq z)=\mathbb{P}(g(X,Y)\leq z)=\iint\limits_{\{(x,y)|g(x,y)\leq z\}}f_{X,Y}(x,y)dxdy
	* Compute the PDF as :math:`f_Z(z)=F'_Z(z)`.

Special cases
========================================================================
Sum of rvs: Convolution
------------------------------------------------------------------------
We want the PDF (or PMF) of the sum of two rvs, :math:`X` and :math:`Y`, :math:`Z=X+Y`.

Discrete case
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. tip::
	* We note that

		.. math:: p_{Z|X}(z|x)=\mathbb{P}(Z=z|X=x)=\mathbb{P}(X+Y=z|X=x)=\mathbb{P}(x+Y=z)=\mathbb{P}(Y=z-x)=p_{Y}(z-x)
	* Therefore, the joint mass between :math:`X` and :math:`Z` factorises as

		.. math:: p_{X,Z}(x,z)=p_X(x)p_{Z|X}(z|x)=p_X(x)p_{Y}(z-x)
	* Let the joint density be :math:`f_{X,Y}(x,y)` (alternatively, joint mass :math:`p_{X,Y}(x,y)`).

		.. math:: p_Z(z)=\sum_{x=-\infty}^\infty p_X(x) p_Y(z-x)=(p_X \ast p_Y)[z]
	* Marginalising, we obtain

		.. math:: p_Z(z)=\sum_x p_{X,Z}(x,z)=\sum_x p_X(x)p_{Y}(z-x)dx=(p_X \ast p_Y)[z]

Continuous case
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. tip::
	* We note that

		.. math:: F_{Z|X}(z|x)=\mathbb{P}(Z\leq z|X=x)=\mathbb{P}(X+Y\leq z|X=x)=\mathbb{P}(x+Y\leq z)=\mathbb{P}(Y\leq z-x)=F_{Y}(z-x)
	* Differentiating both sides, :math:`f_{Z|X}(z|x)=f_{Y}(z-x)`.
	* Therefore, the joint density between :math:`X` and :math:`Z` factorises as

		.. math:: p_{X,Z}(x,z)=p_X(x)p_{Z|X}(z|x)=p_X(x)p_{Y}(z-x)
	* Marginalising, we obtain

		.. math:: f_Z(z)=\int\limits_{-\infty}^\infty f_{X,Z}(x,z)dx=\int\limits_{-\infty}^\infty f_X(x)f_{Y}(z-x)dx=(f_X \ast f_Y)[z]

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


