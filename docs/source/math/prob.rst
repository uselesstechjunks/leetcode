Probability
#######################################################################################
(Adapted from Bertsekas and Tsitsiklis - Introduction to Probability)

Probability Axioms (Kolmogorov Axioms)
----------------------------------------------

#. TODO

Random Variable
------------------------------

Random variables (rvs) are real-valued functions of the outcome of an experiment.

.. note::
	* A function of a rv is another rv.
	* We can associate certain *central measures*/*averages* with each rv, subject to certain condition on their existence.
	* We can *condition* an rv on another rv or an event (this allows us to work with rvs of different numeric types).
	* We can define the notion of *independence* of an rv w.r.t another rv or an event.

Discrete Random Variable
------------------------------

Discrete - values are from a finite/countably infinite set

.. note::
	* We can define a probability mass function (PMF), :math:`p_X(x)`, associated with a discrete rv, :math:`X`, as follows: For each value :math:`x` of :math:`X`,
		#. Collect all possible outcomes that give rise to the event :math:`{X=x}`.
		#. Add their probabilibities to obtain the mass :math:`p_X(x)`.
	* A function of a discrete rv is another rv, whose PMF can be obtained as follows:

Continuous Random Variable
----------------------------------------

Functions of Random Variable
--------------------------------------

	#. Expectation
	#. Variance
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
