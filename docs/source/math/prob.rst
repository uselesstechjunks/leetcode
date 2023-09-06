Probability
#######################################################################################
(Adapted from Bertsekas and Tsitsiklis - Introduction to Probability)

Probability Axioms (Kolmogorov Axioms)
===================================================================

Let the set of all possible outcomes of an experiment be :math:`\Omega`, and let **events** be defined as subsets, :math:`\omega\subset\Omega`. Then a measure :math:`\mu:2^{|\Omega|}\mapsto\mathbb{R}` is called a **probability measure** iff

#. :math:`\mu(\emptyset)=0`.
#. :math:`\mu(\Omega)=1`.
#. For :math:`A_1,A_2,\cdots\subset\Omega` such that :math:`A_i\cap A_j=\emptyset` for :math:`i\neq j`

.. math::
	\mu(A_1\cup A_2\cup\cdots)=\sum_{i=1}^\infty \mu(A_i).

.. tip::
	It is customary to represent probability measure as :math:`\mathbb{P}`.

Random Variable
===================================================================

Random variables (rvs) are real-valued functions of the outcome of an experiment.

.. note::
	* A function of a rv is another rv.
	* We can associate certain *central measures*/*averages* (such as **mean**, **variance**) with each rv, subject to certain condition on their existence.
	* We can *condition* an rv on an event or another rv.
	* We can define the notion of *independence* of an rv w.r.t an event or another rv.

Discrete Random Variable
------------------------------

Discrete = values are from a finite/countably infinite set.

.. note::
	For a discrete rv, :math:`X`:

	* We can define a probability mass function (PMF), :math:`p_X(x)`, associated with :math:`X`, as follows: For each value :math:`x` of :math:`X`

		#. Collect all possible outcomes that give rise to the event :math:`\{X=x\}`.
		#. Add their probabilities to obtain the mass :math:`p_X(x)=\mathbb{P}(\{X=x\})`.

	* A function :math:`g(X)` of :math:`X` is another rv, :math:`Y`, whose PMF can be obtained as follows: For each value :math:`y` of :math:`Y`

		#. Collect all possible values for which :math:`\{x | g(x)=y\}`
		#. Utilize axiom 3 :math:`p_Y(y)=\sum_{\{x | g(x)=y\}} p_X(x)`

.. note::
	Expectation and Variance:

	* We can define **Expectation** of :math:`X` as :math:`\mathbb{E}[X]=\sum_x x p_X(x)` (assuming that the sum exists).
	* We can define **Variance** of :math:`X` as :math:`\mathrm{Var}(X)=\mathbb{E}[(X-\mathbb{E}[X])^2]`.

.. note::
	Law of The Unconscious Statistician (LOTUS):

	* For expectation of :math:`Y=g(X)`, we can get away without having to compute PMF explicitly for :math:`Y`, as it can be shown that
	.. math::
		\mathbb{E}[g(X)]=\sum_x g(x)p_X(x)
	* With the help of LOTUS, :math:`\mathrm{Var}(X)=\sum_x (x-\mathbb{E}[X])^2 p_X(x)`.

.. note::
	* The *n-th moment* of :math:`X` is defined as :math:`\mathbb{E}[X^n]`.
	* Variance in terms of moments: :math:`\mathrm{Var}(X)=\mathbb{E}[X^2]-(\mathbb{E}[X])^2`.

.. note::
	For linear functions of :math:`X`, :math:`g(X)=aX+b`

	* :math:`\mathbb{E}[aX+b]=a\mathbb{E}[X]+b`.
	* :math:`\mathrm{Var}(aX+b)=a^2\mathrm{Var}(X)`.

	For non-linear functions, it is generally **not** true that :math:`\mathbb{E}[g(X)]=g(\mathbb{E}[X])`.

.. note::
	Multiple random variables:

	* We can define the joint-probability mass function for 2 rvs as :math:`p_{X,Y}(x,y)=\mathbb{P}(\{X=x\}\cap\{Y=y\})=\mathbb{P}(X=x,Y=y)`.
	* The **marginal probability** is defined as :math:`p_X(x)=\sum_y p_{X,Y}(x,y)`.
	* LOTUS holds, i.e. for :math:`g(X,Y)`, :math:`\mathbb{E}[g(X,Y)]=\sum_{x,y} g(x,y) p_{X,Y}(x,y)`.
	* Linearity of expectation holds, i.e. :math:`\mathbb{E}[aX+bY+c]=a\mathbb{E}[X]+b\mathbb{E}[Y]+c`.
	* Extends naturally for more than 2 rvs.

.. note::
	Conditioning:

	* An rv can be conditioned on an event :math:`A` and its conditional PMF is defined as :math:`p_{X|A}(x)=\mathbb{P}(X=x|A)`.
	* Extends to the case when the event is defined in terms of another rv, i.e. :math:`A=\{Y=y\}`.

Continuous Random Variable
----------------------------------------

Continuous = values are from an uncountably infinite set.

Functions of Random Variable
--------------------------------------

Moment Generating Functions
----------------------------------------------

#. Distributions
	#. Bernoulli
	#. Binomial
	#. Poisson
	#. Geometric
	#. Multinoulli
	#. Multinomial
	#. Gaussian
	#. Multivariate Gaussian
	#. Exponential
	#. Laplace
	#. Beta
	#. Dirichlet
	#. Dirac
	#. Empirical
	#. Mixture

#. Inequalities
	#. Markov
	#. Chebyshev
	#. Hoeffding
	#. Mill (Gaussian)
	#. Cauchy-Schwarz

#. Convergence
	#. Convergence in probability
	#. Convergence in distribution
	#. Convergence in quadratic mean

#. Information Theory
	#. Shanon Entropy
	#. KL Divergence
	#. Cross Entropy

#. Graphical Models
	#. Bayes Net
	#. Markov Random Factor Model
