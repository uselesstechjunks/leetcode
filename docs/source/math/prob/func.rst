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
	Some effort is required to compute the set :math:`\{x|g(x)\leq y\}`.

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

******************************************************************************************
Covariance and correlation
******************************************************************************************
Covariance is defined between two rvs as :math:`\mathrm{Cov}(X,Y)=\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])]`.

.. note::
	* :math:`\mathrm{Cov}(X,Y)=\mathbb{E}[XY]-\mathbb{E}[X]\mathbb{E}[Y]`.

		* Proof follows from expanding the expression in definition.
	* :math:`\mathrm{Cov}(X,X)=\mathrm{Var}(X)`.
	* :math:`\mathrm{Cov}(X,aY+b)=a\cdot\mathrm{Cov}(X,Y)`.
	* :math:`\mathrm{Cov}(X,Y+Z)=\mathrm{Cov}(X,Y)+\mathrm{Cov}(X,Z)`.
	* :math:`\mathrm{Var}(X+Y)=\mathrm{Var}(X)+\mathrm{Var}(Y)+\mathrm{Cov}(X,Y)`.
	* In general

		.. math:: \mathrm{Var}\left(\sum_{i=1}^n X_i\right)=\sum_{i=1}^n \mathrm{Var}(X_i)+\sum_{i=1}^n\sum_{j=1, i\neq j}^n\mathrm{Cov}(X_i,Y_j)

.. note::
	* Correlation is defined as the normalised version of covariance

		.. math:: \rho(X,Y)=\frac{\mathrm{Cov}(X,Y)}{\sqrt{\mathrm{Var}(X)\mathrm{Var}(Y)}}.
	* We have :math:`|\rho(X,Y)|\leq 1`.

		* Let :math:`\tilde{X}=X-\mathbb{E}[X]` and :math:`\tilde{Y}=Y-\mathbb{E}[Y]` be the centered rvs.
		* The correlation coefficient then becomes

			.. math:: \rho(X,Y)=\frac{\mathbb{E}[\tilde{X}\tilde{Y}]}{\sqrt{\mathbb{E}[\tilde{X}^2]\cdot \mathbb{E}[\tilde{Y}^2]}}
		* The proof follows from Cauchy-Schwarz inequality.
	* The equality holds only when :math:`\tilde{X}=c\cdot \tilde{Y}` for some :math:`c`.

******************************************************************************************
Estimation using conditional expectation
******************************************************************************************
.. note::
	* We assume that knowing :math:`Y`, we can estimate :math:`X`.
	* We assume that conditional density :math:`f_{X|Y}(x|y)` is known.

		* [**Discriminative**] We might have access to the conditional density directly.
		* [**Generative**] We might have access to the joint density :math:`f_{X,Y}(x,y)` and we can compute the conditional with Bayes theorem. 
	* From law of iterated expectation, we have :math:`\mathbb{E}[X]=\mathbb{E}[\mathbb{E}[X|Y]]`
	* Therefore

		* Estimator: :math:`\hat{X}=\mathbb{E}[X|Y]` can be thought of as an estimator of :math:`X` as their expected values are the same.

			* For a given value of :math:`Y=y`, the estimation is :math:`\hat{x}=\mathbb{E}[X|Y=y]=r(y)`.
			* The function :math:`r(y)` is known called **regression function**.
		* Estimation error: :math:`\tilde{X}=\hat{X}-X`.

			* This error is expected to be 0, as :math:`\mathbb{E}[\tilde{X}]=\mathbb{E}[\mathbb{E}[X|Y]]-\mathbb{E}[X]=0`.
			* Variance of this error is the same as "Mean-Squared Error" (MSE).

				.. math:: \mathrm{Var}(\tilde{X})=\mathbb{E}[\tilde{X}^2]-\left(\mathbb{E}[\tilde{X}]\right)^2=\mathbb{E}[\tilde{X}^2]=\mathbb{E}[(\hat{X}-X)^2]
			* This error is uncorrelated with the estimator.

				* We note that

					.. math:: \mathrm{Cov}(\hat{X},\tilde{X})=\mathbb{E}[\hat{X}\tilde{X}]-\mathbb{E}[\hat{X}]\mathbb{E}[\tilde{X}]=\mathbb{E}[\hat{X}\tilde{X}]
				* Invoking law of iterated expectation

					.. math:: \mathbb{E}[\hat{X}\tilde{X}]=\mathbb{E}[\mathbb{E}[\hat{X}\tilde{X}|Y]]
				* Given :math:`Y`, :math:`\hat{X}` is constant.

					.. math:: \mathbb{E}[\mathbb{E}[\hat{X}\tilde{X}|Y]]=\mathbb{E}[\hat{X}\cdot\mathbb{E}[\tilde{X}|Y]]=\mathbb{E}[\hat{X}\cdot\mathbb{E}[(\hat{X}-X)|Y]]=\mathbb{E}[\hat{X}\cdot\mathbb{E}[\hat{X}|Y]]-\mathbb{E}[\hat{X}\cdot\mathbb{E}[X|Y]]=\mathbb{E}[\hat{X}^2]-\mathbb{E}[\hat{X}^2]=0
			* Therefore, we have :math:`\mathrm{Var}(X)=\mathrm{Var}(\hat{X})+\mathrm{Var}(\tilde{X})`.

Conditional variance
========================================================================

.. note::
	We can define conditional variance as :math:`\mathrm{Var}(X|Y)=\mathbb{E}[(X-\mathbb{E}[X|Y])^2|Y]` such that
	
		.. math:: \mathbb{E}[\mathrm{Var}(X|Y)]=\mathbb{E}[\mathbb{E}[(X-\mathbb{E}[X|Y])^2|Y]]=\mathbb{E}[(X-\mathbb{E}[X|Y])^2]=\mathrm{E}[\tilde{X}^2]=\mathrm{Var}(\tilde{X})

Law of iterated variance
========================================================================

.. note::
	We can rewrite the variance relation using this new notation

		.. math:: \mathrm{Var}(X)=\mathrm{Var}(\mathbb{E}[X|Y])+\mathbb{E}[\mathrm{Var}(X|Y)]

.. tip::
	* The iterated law of expectation and variance allows us to tackle complicated cases by taking help in conditioning.
	* Example: A coin with unknown probability of head is tossed :math:`n` times. The probability is known to be uniform in :math:`[0,1]`. Let :math:`X` is the total number of heads. Find :math:`\mathbb{E}[X]` and :math:`\mathrm{Var}(X)`.

******************************************************************************************
Transforms of rv
******************************************************************************************

Moment Generating Functions
========================================================================


