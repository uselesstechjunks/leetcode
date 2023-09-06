Probability
#######################################################################################
(Adapted from Bertsekas and Tsitsiklis - Introduction to Probability)

Probability Axioms (Kolmogorov Axioms)
===================================================================

#. TODO

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

	* We can define **Expectation** of :math:`X` as :math:`\mathbb{E}[X]=\sum_x x p_X(x)`.
	* We can define **Variance** of :math:`X` as :math:`\Var(X)=\mathbb{E}[(X-\mathbb{E}[X])^2]`.

.. note::
	Law of The Unconscious Statistician (LOTUS):

	* For expectation of :math:`Y=g(X)`, we can get away without having to compute PMF explicitly for :math:`Y`, as it can be shown that
		.. math::
			\mathbb{E}[g(X)]=\sum_x g(x)p_X(x)
	* With the help of LOTUS, :math:`Var(X)=\sum_x (x-\mathbb{E}[X])^2 p_X(x)`.

.. note::
	* The *n-th moment* of :math:`X` is defined as :math:`\mathbb{E}[X^n]`.
	* Variance in terms of moments: :math:`Var(X)=\mathbb{E}[X^2]-(\mathbb{E}[X])^2`.

Continuous Random Variable
----------------------------------------

Continuous = values are from an uncountably infinite set.

Functions of Random Variable
--------------------------------------

Expectation
----------------------------------------------

Variance
----------------------------------------------

	#. Law Of The Unconscious Statistician (LOTUS)
	#. Covariance
	#. Moment Generating Functions

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
