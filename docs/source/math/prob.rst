#######################################################################################
Probability
#######################################################################################
(Adapted from Bertsekas and Tsitsiklis - Introduction to Probability)

******************************************************
Probability Axioms (Kolmogorov Axioms)
******************************************************

Let the set of all possible outcomes of an experiment be :math:`\Omega`, and let **events** be defined as subsets, :math:`\omega\subset\Omega`. Then a measure :math:`\mu:2^{|\Omega|}\mapsto\mathbb{R}` is called a **probability measure** iff

#. **Non-negativity**: :math:`\mu(\omega)\ge 0` for any :math:`\omega\subset\Omega`.
#. **Unitarity**: :math:`\mu(\Omega)=1`.
#. :math:`\sigma`-**Additivity**: For :math:`A_1,A_2,\cdots\subset\Omega` such that :math:`A_i\cap A_j=\emptyset` for :math:`i\neq j`

	.. math::
		\mu(\bigcup_{i=1}^\infty A_i)=\sum_{i=1}^\infty \mu(A_i).

.. tip::
	It is customary to represent probability measure as :math:`\mathbb{P}`.

*********************************************
Random Variable
*********************************************

Random variables (rvs) are real-valued functions of the outcome of an experiment.

.. note::
	* A function of a rv is another rv.
	* We can associate certain *central measures*/*averages* (such as **mean**, **variance**) with each rv, subject to certain condition on their existence.
	* We can *condition* an rv on an event or another rv.
	* We can define the notion of *independence* of an rv w.r.t an event or another rv.

Discrete Random Variable
====================================

Discrete = values are from a finite/countably infinite set.

.. note::
	For a discrete rv, :math:`X`:

	* We can define a probability mass function (PMF), :math:`p_X(x)`, associated with :math:`X`, as follows: For each value :math:`x` of :math:`X`

		#. Collect all possible outcomes that give rise to the event :math:`\{X=x\}`.
		#. Add their probabilities to obtain the mass :math:`p_X(x)=\mathbb{P}(\{X=x\})`.

	* A function :math:`g(X)` of :math:`X` is another rv, :math:`Y`, whose PMF can be obtained as follows: For each value :math:`y` of :math:`Y`

		#. Collect all possible values for which :math:`\{x | g(x)=y\}`
		#. Utilize axiom 3 to obtain :math:`p_Y(y)=\sum_{\{x | g(x)=y\}} p_X(x)`

Expectation and Variance:
------------------------------------
.. note::
	* We can define **Expectation** of :math:`X` as :math:`\mathbb{E}[X]=\sum_x x p_X(x)` (assuming that the sum exists).
	* Elementary properties of expectation:

		* If :math:`X>0`, then :math:`\mathbf{E}[X]>0`.
		* If :math:`a\leq X\leq b`, then :math:`a\leq \mathbf{E}[X]\leq b`.
		* If :math:`X=c`, then :math:`\mathbf{E}[X]=c`.
	* We can define **Variance** of :math:`X` as :math:`\mathrm{Var}(X)=\mathbb{E}[(X-\mathbb{E}[X])^2]`.

.. tip::
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

..  warning::
	For non-linear functions, it is generally **not** true that :math:`\mathbb{E}[g(X)]=g(\mathbb{E}[X])`.

Multiple random variables:
------------------------------------
.. note::
	* We can define the joint-probability mass function for 2 rvs as 

		.. math::
			p_{X,Y}(x,y)=\mathbb{P}(\{X=x\}\cap\{Y=y\})=\mathbb{P}(X=x,Y=y).

	* The **marginal probability** is defined as :math:`p_X(x)=\sum_y p_{X,Y}(x,y)`.
	* LOTUS holds, i.e. for :math:`g(X,Y)`, :math:`\mathbb{E}[g(X,Y)]=\sum_{x,y} g(x,y) p_{X,Y}(x,y)`.
	* Linearity of expectation holds, i.e. :math:`\mathbb{E}[aX+bY+c]=a\mathbb{E}[X]+b\mathbb{E}[Y]+c`.
	* Extends naturally for more than 2 rvs.

Conditioning:
------------------------------------
.. note::
	* An rv can be conditioned on an event :math:`A` (when :math:`\mathbb{P}(A)>0`) and its conditional PMF is defined as 

		.. math::
			p_{X|A}(x)=\mathbb{P}(X=x|A).

	* Extends to the case when the event is defined in terms of another rv, i.e. :math:`A=\{Y=y\}` (:math:`p_Y(y)>0`) and is written as

		.. math::
			p_{X|Y}(x|y)=\mathbb{P}(X=x|Y=y)=\frac{p_{X,Y}(x,y)}{p_Y(y)}

	* Connects to the joint PMF as :math:`p_{X,Y}(x,y)=p_Y(y)p_{X|Y}(x|y)`	

.. tip::
	* **Bayes theorem**: For :math:`p_Y(y)>0`, :math:`p_{Y|X}(y|x)=\frac{p_Y(y)p_{X|Y}(x|y)}{\sum_y p_Y(y)p_{X|Y}(x|y)}`
	* :math:`p_Y(y)` is known as **prior**, :math:`p_{Y|X}(y|x)` is called **posterior**, and :math:`p_{X|Y}(x|y)` is known as **likelihood**. 
	* The demoninator :math:`Z=\sum_y p_Y(y)p_{X|Y}(x|y)` is the probability normalisation factor (i.e. it ensures that the sum is 1).
	* We can often work with unnormalised probabilities when exact values are not required, as :math:`p_{Y|X}(y|x)\propto p_Y(y)p_{X|Y}(x|y)`.

.. tip::
	**Total law of probability:** Let :math:`A_1,A_2,\cdots,A_n` be disjoints events such that :math:`\bigcup_{i=1}^n A_i=\Omega` (i.e. they define a partition).

	*  If :math:`\mathbb{P}(A_i)>0` for all :math:`i`, then 
	
		.. math::
			p_X(x)=\sum_{i=1}^n\mathbb{P}(A_i)p_{X|A_i}(x)

	* This also works if :math:`A` is defined in terms of a rv (i.e. :math:`A=\{X=x\}`), even when the cardinality of :math:`X` is countably infinite.

		.. math::
			p_{X}(x)=\sum_y p_Y(y)p_{X|Y}(x|y)

	* This allows us to compute the probability of events in a complicated probability model by utilising events from a simpler model, i.e. let's us use the divide-and-conquer technique. We just need to ensure that the events from the simpler model in fact exhausts the entirety of sample space of the original probability model.
	* For any other event :math:`B` where :math:`\mathbb{P}(A_i\cap B)>0` for all :math:`i`

		.. math::
			p_{X|B}(x)=\sum_{i=1}^n\mathbb{P}(A_i|B)p_{X|A_i\cap B}(x)

.. note::
	Conditional expectation:

	* Defined in terms of the conditional PMF, such as :math:`\mathbb{E}[X|A]=\sum_x x p_{X|A}(x)` and :math:`\mathbb{E}[X|Y=y]=\sum_x x p_{X|Y}(x|y)`.
	* LOTUS holds, i.e. :math:`\mathbb{E}[g(X)|A]=\sum_x g(x)p_{X|A}(x)`.

.. tip::
	From total law of probability:

	* For partitions :math:`A_1,A_2,\cdots,A_n`

		.. math::
			\mathbb{E}[X]=\sum_x x p_X(x)=\sum_{i=1}^n \mathbb{P}(A_i)\sum_x x p_{X|A_i}(x)=\sum_{i=1}^n \mathbb{P}(A_i)\mathbb{E}[X|A_i]
	
	* For any other event :math:`B` where :math:`\mathbb{P}(A_i\cap B)>0` for all :math:`i`

		.. math::
			\mathbb{E}[X|B]=\sum_{i=1}^n \mathbb{P}(A_i|B)\mathbb{E}[X|A_i\cap B]

	* **Law of iterated expectation:** For the rv version of the first, :math:`\mathbb{E}[X]=\sum_y p_Y(y)\mathbb{E}[X|Y]=\mathbb{E}[\mathbb{E}[X|Y]]`

Notion of Independence:
------------------------------------
.. note::
	* :math:`X` is independent of an event :math:`A` iff :math:`p_{X|A}(x)=p_X(x)` for all :math:`x`.
	* For two independent rvs, :math:`p_{X,Y}(x,y)=p_X(x)p_Y(y)` for all :math:`x` and :math:`y`.

.. note::
	Expectation and variance for independent rvs:

	* :math:`\mathbb{E}[XY]=\mathbb{E}[X]\mathbb{E}[Y]`
	* :math:`\mathrm{Var}(X+Y)=\mathrm{Var}(X)+\mathrm{Var}(Y)`
	* Extends naturally to more than 2 rvs.

Some discrete random variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simple rvs:
""""""""""""""""""""""""""""""""""""
* Bernoulli:

Any experiment that deals with a binary outcome (e.g. **success** or **failure**) can be represented by a Bernoulli rv. 

.. note::
	* We can define a rv :math:`X=1` which represents success and :math:`X=0` which represents failure.
	* We only need to know about one of the probability values, :math:`\mathbb{P}(X=1)=p`, as :math:`\mathbb{P}(X=0)=1-p`.
	* Therefore, a Bernoulli rv is parameterised with just 1 parameter, :math:`p`.
	* [Derive] For :math:`X\sim\mathrm{Ber}(p)`, :math:`\mathbb{E}[X]=p` and :math:`\mathrm{Var}(X)=p(1-p)`.

* Uniform:

Composite rvs:
""""""""""""""""""""""""""""""""""""
* Binomial

In a repeated (:math:`n`-times) Bernoulli trial with parameter :math:`p`, let :math:`X` denote the total number of **successes**. Then :math:`X\sim\mathrm{Bin}(n,p)`.

.. math::
	p_X(x)={n \choose x} p^x(1-p)^{n-x}

.. note::
	We can write a Binomially distributed rv as a sum of independent, Bernoulli rvs. 

	* Let's denote each of the trials with a different Bernoulli rv, :math:`X_i\sim\mathrm{Ber}(p)` for :math:`i`-th trial. 
	* Then :math:`Y=X_1+\cdots+X_n` represents the total number of successes where :math:`X_i\bot X_j` for :math:`i\neq j`.
	* [Derive] For :math:`X\sim\mathrm{Bin}(n,p)`, :math:`\mathbb{E}[X]=np` and :math:`\mathrm{Var}(X)=np(1-p)`.

..  tip::
	Solving a problem with an exisitng framework often requires us to think of a process with which the experiment takes place. With the right process description, seemingly difficult problems often become easy.

..  attention::
	[The Birthday Problem] In a party of :math:`500` guests, what is the probability that you share your birthday with :math:`5` other people?

	* All birthdays are equally likely (assumption of the underlying probability model).
	* Person A's birthday is independent of person B's birthday.
	* To find out the number of people who share their birthday with me I can
		* pick a person in random and ask their birthday
		* I consider it a success if their birthday is the same as mine, failure otherwise
		* repeat for all :math:`n`
	* Total number of successes represents the total number of people who share their birthday with me.

* Geometric
* Multinomial

Limiting rvs:
""""""""""""""""""""""""""""""""""""
* Poisson

Continuous Random Variable
======================================================

Continuous = values are from an uncountably infinite set.

Functions of Random Variable
=============================================
.. tip::
	Sum of independent rvs - Convolution:

	* Let :math:`X\sim p_X` and :math:`Y\sim p_Y` be two independent discrete rvs. Then their sum :math:`Z=X+Y` has the distribution

		.. math::
			p_Z(z)=\sum_{x=-\infty}^\infty p_X(x) p_Y(z-x)=(p_X \ast p_Y)[z]


Moment Generating Functions
=============================================

#. Distributions
	#. Multinoulli
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
