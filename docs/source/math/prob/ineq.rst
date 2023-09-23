################################################################
Tail Bounds, Convergence and Limit Theorems
################################################################

*********************************************
Tail Bounds : Inequalities
*********************************************
Why do we need inequalities?

.. note::
	* If we know the density directly, then, theoretically, we can compute exactly how much probability lie on the "tails".
	* However, often we have to work with scenarios where either the density is not known or the exact calculation is cumbersome.
	* These inequalities bound the probability on the tails based on how much we know about the underlying distribution.

Mean known: Markov
====================================
.. note::
	* Let :math:`X` be a non-negative rv with well defined :math:`\mathbb{E}[X]=\mu`.
	* Markov's inequality states that the tail probability goes down inversely as we move further right.

		.. math:: \mathbb{P}(X\geq t)\leq \frac{\mu}{t}

Mean and variance known: Chebyshev
====================================
.. note::
	* Let :math:`X` be **any** rv with well defined :math:`\mathbb{E}[X]=\mu` and :math:`\mathrm{Var}(X)=\sigma^2`.
	* Chebyshev's inequality states that the tail probability of :math:`t` being away from :math:`\mu` goes down quadratically.

		.. math:: \mathbb{P}(|X-\mu|\geq t)\leq \frac{\sigma^2}{t^2}
	* This is a tighter bound than Markov's.

MGF known: Chernoff's Bound
====================================
.. note::
	* Let :math:`X` be any rv with well defined MGF, :math:`M_X(s)`.
	* Chernoff's bound states that the probability of the tails goes down exponentially as we move further right. So, for any :math:`s`,

		.. math:: \mathbb{P}(X\geq t)\leq \frac{M_X(s)}{e^{st}}
	* We can recover Markov's and Chebyshev's from this one.

Distribution known (Gaussian): Mill 
====================================
.. note::
	* Let :math:`Z` be a standard normal rv.
	* Mills equality calculates directly from density how much probability lies on the tails.

		.. math:: \mathbb{P}(|Z|\geq t)\leq \frac{2e^{-t^2/2}}{t}

*********************************************
Convergence
*********************************************

We consider a sequence of 

Convergence in distribution
====================================

Convergence in probability
====================================

Convergence in quadratic mean
====================================

Almost surely convergence
====================================

*********************************************
Limit Theorems
*********************************************

Weak Law of Large Number
====================================

Special case: bounded rvs
------------------------------------

Hoeffding's inequality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Strong Law of Large Number
====================================

Central Limit Theorem
====================================
