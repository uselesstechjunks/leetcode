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

			.. math:: F_Y(y)=\mathbb{P}(Y\leq y)=\mathbb{P}(g(X)\leq y)=\int\limits_{\{x|g(x)\leq y\}}f_X(x) \mathop{dx}
		* Compute the PDF as :math:`f_Y(y)=F'_Y(y)`.

.. tip::
	* Some effort is required to compute the set :math:`\{x|g(x)\leq y\}`.
	* Find :math:`f_Y(y)` where :math:`Y=X^2`.

Special cases
========================================================================
Linear functions
------------------------------------------------------------------------
Let :math:`Y=g(X)=aX+b`.

.. tip::
	* If :math:`a=0`, then :math:`Y=b` with probability 1 and it's no longer a continuous rv.
	* If :math:`a\neq 0`, then we have

		.. math:: F_Y(y)=\mathbb{P}(Y\leq y)=\mathbb{P}(g(X)\leq y)=\mathbb{P}(aX+b\leq y)=\begin{cases}\mathbb{P}\left(X\leq\frac{y-b}{a}\right) & \text{if $a>0$} \\ \mathbb{P}\left(X\geq\frac{y-b}{a}\right) & \text{if $a<0$}\end{cases}=\begin{cases}F_X(\frac{y-b}{a}) & \text{if $a>0$} \\ 1-F_X(\frac{y-b}{a}) & \text{if $a<0$}\end{cases}
	* We can recover the PDF in both cases as

		.. math:: f_Y(y)=\begin{cases}\frac{1}{a}f_X(\frac{y-b}{a}) & \text{if $a>0$} \\ -\frac{1}{a}f_X(\frac{y-b}{a}) & \text{if $a<0$}\end{cases}=\frac{1}{\left| a \right|}f_X(\frac{y-b}{a})

Monotonic functions
------------------------------------------------------------------------
.. note::
	* If :math:`g(y)=x` is a monotonic function, then it has an inverse, :math:`x=g^{-1}(y)`.
	* Therefore, we have

		.. math:: F_Y(y)=\mathbb{P}(Y\leq y)=\mathbb{P}(g(X)\leq y)=\begin{cases}\mathbb{P}(X\leq g^{-1}(y)) & \text{if $g(X)$ is monotonic increasing}\\\mathbb{P}(X\geq g^{-1}(y)) & \text{if $g(X)$ is monotonic decreasing}\end{cases}=\begin{cases}F_X(g^{-1}(y)) & \text{if $g(X)$ is monotonic increasing}\\1-F_X(g^{-1}(y)) & \text{if $g(X)$ is monotonic decreasing}\end{cases}
	* We can recover the PDF in both cases as

		.. math:: f_Y(y)=|f_X(g^{-1}(y))|\cdot\frac{\mathop{d}}{\mathop{dy}}\left[g^{-1}(y)\right]
	* We note that the linear case is a special case of monotonic functions.

******************************************************************************************
Density of a function of multiple jointly distributed rvs
******************************************************************************************
Let :math:`Z=g(X,Y)` be a function of 2 jointly distributed rvs, :math:`X` and :math:`Y`. In this case, we follow the same process as before.

.. tip::
	* Compute the CDF as

		.. math:: F_Z(z)=\mathbb{P}(Z\leq z)=\mathbb{P}(g(X,Y)\leq z)=\iint\limits_{\{(x,y)|g(x,y)\leq z\}}f_{X,Y}(x,y)\mathop{dx}\mathop{dy}
	* Compute the PDF as :math:`f_Z(z)=F'_Z(z)`.
	* Extends naturally for more than 2 rvs.

.. seealso::
	* Find the PDF of :math:`Z=X/Y`, where :math:`X` and :math:`Y` are independent and uniformly distributed in :math:`[0,1]`.
	* Two people join a call but they are late by an amount, independent of the other, that follows an exponential distribution with parameter :math:`\lambda`. Find the PDF of the difference in their joining time.

Special cases
========================================================================
Sum of independent rvs: Convolution
------------------------------------------------------------------------
We want the PDF (or PMF) of the sum of two independent rvs, :math:`X` and :math:`Y`, :math:`Z=X+Y`.

Discrete case
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. tip::
	* We note that

		.. math:: p_{Z|X}(z|x)=\mathbb{P}(Z=z|X=x)=\mathbb{P}(X+Y=z|X=x)=\mathbb{P}(x+Y=z)=\mathbb{P}(Y=z-x)=p_{Y}(z-x)
	* Therefore, the joint mass between :math:`X` and :math:`Z` factorises as

		.. math:: p_{X,Z}(x,z)=p_X(x)p_{Z|X}(z|x)=p_X(x)p_{Y}(z-x)
	* Marginalising, we obtain

		.. math:: p_Z(z)=\sum_x p_{X,Z}(x,z)=\sum_x p_X(x)p_{Y}(z-x)=(p_X \ast p_Y)[z]

Continuous case
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. tip::
	* We note that

		.. math:: F_{Z|X}(z|x)=\mathbb{P}(Z\leq z|X=x)=\mathbb{P}(X+Y\leq z|X=x)=\mathbb{P}(x+Y\leq z)=\mathbb{P}(Y\leq z-x)=F_{Y}(z-x)
	* Differentiating both sides, :math:`f_{Z|X}(z|x)=f_{Y}(z-x)`.
	* Therefore, the joint density between :math:`X` and :math:`Z` factorises as

		.. math:: f_{X,Z}(x,z)=f_X(x)f_{Z|X}(z|x)=f_X(x)f_{Y}(z-x)
	* Marginalising, we obtain

		.. math:: f_Z(z)=\int\limits_{-\infty}^\infty f_{X,Z}(x,z)\mathop{dx}=\int\limits_{-\infty}^\infty f_X(x)f_{Y}(z-x)\mathop{dx}=(f_X \ast f_Y)[z]

.. seealso::
	* Find the PDF of the sum of two independent normals.

******************************************************************************************
Covariance and correlation
******************************************************************************************
Covariance is defined between two rvs as :math:`\mathrm{Cov}(X,Y)=\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])]`.

.. note::
	* :math:`\mathrm{Cov}(X,Y)=\mathbb{E}[XY]-\mathbb{E}[X]\mathbb{E}[Y]`.

		* Proof follows from expanding the expression in definition.
	* :math:`\mathrm{Cov}(X,X)=\mathbb{V}(X)`.
	* :math:`\mathrm{Cov}(X,aY+b)=a\cdot\mathrm{Cov}(X,Y)`.
	* :math:`\mathrm{Cov}(X,Y+Z)=\mathrm{Cov}(X,Y)+\mathrm{Cov}(X,Z)`.
	* :math:`\mathbb{V}(X+Y)=\mathbb{V}(X)+\mathbb{V}(Y)+\mathrm{Cov}(X,Y)`.
	* In general

		.. math:: \mathbb{V}\left(\sum_{i=1}^n X_i\right)=\sum_{i=1}^n \mathbb{V}(X_i)+\sum_{i=1}^n\sum_{j=1, i\neq j}^n\mathrm{Cov}(X_i,Y_j)

.. note::
	* Correlation is defined as the normalised version of covariance

		.. math:: \rho(X,Y)=\frac{\mathrm{Cov}(X,Y)}{\sqrt{\mathbb{V}(X)\mathbb{V}(Y)}}.
	* We have :math:`|\rho(X,Y)|\leq 1`.

		* Let :math:`\tilde{X}=X-\mathbb{E}[X]` and :math:`\tilde{Y}=Y-\mathbb{E}[Y]` be the centered rvs.
		* The correlation coefficient then becomes

			.. math:: \rho(X,Y)=\frac{\mathbb{E}[\tilde{X}\tilde{Y}]}{\sqrt{\mathbb{E}[\tilde{X}^2]\cdot \mathbb{E}[\tilde{Y}^2]}}
		* The proof follows from Cauchy-Schwarz inequality.
	* The equality holds only when :math:`\tilde{X}=c\cdot \tilde{Y}` for some :math:`c`.

.. seealso::
	* We can solve the hat problem using covariance.

******************************************************************************************
Fundamentals of Estimation
******************************************************************************************
.. note::
	* **Estimate**: If we do not know the exact value of a rv :math:`Y`, we can use a **guess** (estimate). 
	
		* The **guess** is another rv which can be observed or calculated based on other rvs.
	* **Estimator**: The rv which takes estimates as values is known as the **estimator**.

		* Estimator for :math:`Y` is usually written as :math:`\hat{Y}`.
		* Estimates are the values that this rv can take, :math:`\hat{Y}=\hat{y}`.
		* **Standard error**: :math:`\text{se}(\hat{Y})=\sqrt{\mathbb{V}_Y(\hat{Y})}`.
	* **Estimation error**: :math:`\tilde{Y}=\hat{Y}-Y`.

		* **Bias of an estimator**: :math:`\text{bias}(\hat{Y})=\mathbb{E}_Y[\tilde{Y}]`.
		* **Mean squared error**: :math:`\text{mse}(\hat{Y})=\mathbb{E}_Y[\tilde{Y}^2]`.

			* We note that :math:`\mathbb{V}_Y(\tilde{Y})=\mathbb{E}_Y[\tilde{Y}^2]-\left(\mathbb{E}_Y[\tilde{Y}]\right)^2=\text{mse}(\hat{Y})-\text{bias}(\hat{Y})^2`.
			* This can be rewritten as :math:`\text{mse}(\hat{Y})=\text{bias}(\hat{Y})^2+\mathbb{V}_Y(\tilde{Y})`.
			* If the quantity we're estimating is an unknown constant :math:`c` instead of being a rv (as in classical statistical estimation of an unknown parameter),

				.. math:: \text{mse}(\hat{Y})=\text{bias}(\hat{Y})^2+\mathbb{V}_Y(\hat{Y}-c)=\text{bias}(\hat{Y})^2+\mathbb{V}_Y(\hat{Y})=\text{bias}(\hat{Y})^2+\text{se}(\hat{Y})^2

Estimation using conditional expectation
==========================================================================================
.. note::
	* We assume that knowing :math:`X`, we can estimate :math:`Y`.

		* We assume that conditional density :math:`f_{Y|X}(y|x)` is known.
	
			* [**Discriminative**] We might have access to the conditional density directly.
			* [**Generative**] We might have access to the joint density :math:`f_{X,Y}(x,y)` and we can compute the conditional with Bayes theorem. 
	* From law of iterated expectation, we have :math:`\mathbb{E}[Y]=\mathbb{E}[\mathbb{E}[Y|X]]`
	* Therefore

		* Estimator: :math:`\hat{Y}=\mathbb{E}[Y|X]` can be thought of as an estimator of :math:`X` as their expected values are the same.

			* For a given value of :math:`X=x`, the estimation is :math:`\hat{y}=\mathbb{E}[Y|X=x]=r(x)`.
			* The function :math:`r(x)` is known called **regression function**.
		* Bias: Since :math:`\tilde{Y}` is expected to be 0

			.. math:: \text{bias}(\hat{Y})=\mathbb{E}[\tilde{Y}]=\mathbb{E}[\mathbb{E}[Y|X]]-\mathbb{E}[Y]=0\implies\text{mse}(\hat{Y})=\text{se}(\hat{Y})^2
		* This error is uncorrelated with the estimator.

			* We note that

				.. math:: \mathrm{Cov}(\hat{Y},\tilde{Y})=\mathbb{E}[\hat{Y}\tilde{Y}]-\mathbb{E}[\hat{Y}]\mathbb{E}[\tilde{Y}]=\mathbb{E}[\hat{Y}\tilde{Y}]
			* Invoking law of iterated expectation

				.. math:: \mathbb{E}[\hat{Y}\tilde{Y}]=\mathbb{E}[\mathbb{E}[\hat{Y}\tilde{Y}|X]]
			* Given :math:`X`, :math:`\hat{Y}` is constant.

				.. math:: \mathbb{E}[\mathbb{E}[\hat{Y}\tilde{Y}|X]]=\mathbb{E}[\hat{Y}\cdot\mathbb{E}[\tilde{Y}|X]]=\mathbb{E}[\hat{Y}\cdot\mathbb{E}[(\hat{Y}-Y)|X]]=\mathbb{E}[\hat{Y}\cdot\mathbb{E}[\hat{Y}|X]]-\mathbb{E}[\hat{Y}\cdot\mathbb{E}[Y|X]]=\mathbb{E}[\hat{Y}^2]-\mathbb{E}[\hat{Y}^2]=0
		* Therefore, we have :math:`\mathbb{V}(Y)=\mathbb{V}(\hat{Y})+\mathbb{V}(\tilde{Y})=\text{se}(\hat{Y})^2+\text{mse}(\hat{Y})`.

Conditional variance
========================================================================
.. note::
	We can define conditional variance as :math:`\mathbb{V}(X|Y)=\mathbb{E}[(X-\mathbb{E}[X|Y])^2|Y]` such that
	
		.. math:: \mathbb{E}[\mathbb{V}(X|Y)]=\mathbb{E}[\mathbb{E}[(X-\mathbb{E}[X|Y])^2|Y]]=\mathbb{E}[(X-\mathbb{E}[X|Y])^2]=\mathrm{E}[\tilde{X}^2]=\mathbb{V}(\tilde{X})

Law of iterated variance
========================================================================

.. note::
	We can rewrite the variance relation using this new notation

		.. math:: \mathbb{V}(X)=\mathbb{V}(\mathbb{E}[X|Y])+\mathbb{E}[\mathbb{V}(X|Y)]

.. tip::
	The iterated law of expectation and variance allows us to tackle complicated cases by taking help in conditioning.

.. seealso::
	* A coin with unknown probability of head is tossed :math:`n` times. The probability is known to be uniform in :math:`[0,1]`. Let :math:`X` is the total number of heads. Find :math:`\mathbb{E}[X]` and :math:`\mathbb{V}(X)`.

******************************************************************************************
Transforms of rv
******************************************************************************************
Moment Generating Function
========================================================================
.. note::
	* Moment generating function (MGF) of a rv is defined as a function of another parameter :math:`s`

		.. math:: M_X(s)=\mathbb{E}[e^{sX}]
	* This closely relates to the **Laplace Transform** (see stat stackexchange post `here <https://stats.stackexchange.com/questions/238776/how-would-you-explain-moment-generating-functionmgf-in-laymans-terms>`_)
	* We note that

		.. math:: M_X(s)=\mathbb{E}[e^{sX}]=\int\left(1+sx+\frac{s^2x^2}{2!}+\cdots\right)\mathop{dx}=1+s\cdot\mathbb{E}[X]+\frac{s^2}{2!}\cdot\mathbb{E}[X^2]+\cdots

		* From this, we establish that :math:`\frac{\mathop{d}^n}{\mathop{ds}^n}\left(M_X(s)\right)|_{s=0}=\mathbb{E}[X^n]`.
	* Extends to the multivariate case as

		.. math:: M_{X_1,X_2,\cdots,X_n}(s_1,s_2,\cdots,s_n)=\mathbb{E}[e^{\sum_{i=1}^n s_i X_i}]
	* For two independent rvs :math:`X` and :math:`Y`, the MGF of their sum :math:`Z=X+Y` is given by 

		.. math:: M_{Z}(s)=\mathbb{E}[e^{sX+sY}]=\mathbb{E}[e^{sX}e^{sY}]=\mathbb{E}[e^{sX}]\mathbb{E}[e^{sY}]=M_{X}(s)\cdot M_{Y}(s)
	* The above extends for multiple independent rvs.

.. attention::
	MGFs completely determines the CDFs and densities/mass functions.

.. tip::
	* Knowing MGF often helps us find the moments easier than direct approach.
	* Find the expectation and variance of exponential distribution in normal way and using MGF.

.. seealso::
	Find the expectation, variance and the transform of the sum of independent rvs where the number of terms is also a rv.
