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

The concept of convergence of sequences involving rvs is more subtle than convergence of normal sequences. We consider a sequence of rvs, :math:`X_1,X_2,\cdots` and :math:`X` as the limiting rv. There are multiple notions of convergence for such sequences, some being stronger than others (as in, convergence in one sense might imply convergence in another sense, but not vice versa).

Convergence in distribution
====================================
.. note::
	* :math:`(X_n)_{n=1}^\infty` is said to be converging to :math:`X` in distribution, :math:`X_n\xrightarrow[]{D}X`, if

		.. math:: \lim\limits_{n\to\infty}\mathbb{P}(X_n\leq t)=\mathbb{P}(X\leq t)
	* If :math:`X_i\sim F_i` and :math:`X\sim F`, then the above can be written in terms of CDF as

		.. math:: \lim\limits_{n\to\infty}F_n(t)=F(t)

Convergence in probability
====================================
.. note::
	* :math:`(X_n)_{n=1}^\infty` is said to be converging to :math:`X` in probability, :math:`X_n\xrightarrow[]{P}X`, if

		.. math:: \lim\limits_{n\to\infty}\mathbb{P}(|X_n-X|\geq\epsilon)=0
	* It can be restated using notions similar to convergence from calculus as follows: for a given **accuracy level** :math:`\epsilon>0` and a given **confidence level** :math:`\delta>0`,

		.. math:: \exists N_{\epsilon,\delta} . n>N\implies\mathbb{P}(|X_n-X|\geq\epsilon)\leq\delta
	* Convergence in probability implies convergence in distribution.

Convergence in :math:`L_1`
====================================
.. note::
	* :math:`(X_n)_{n=1}^\infty` is said to be converging to :math:`X` in :math:`L_1`, :math:`X_n\xrightarrow[]{L_1}X`, if

		.. math:: \lim\limits_{n\to\infty}\mathbb{E}[|X_n-X|]=0
	* Convergence in :math:`L_1` implies convergence in probability.

Convergence in quadratic mean
====================================
.. note::
	* :math:`(X_n)_{n=1}^\infty` is said to be converging to :math:`X` in quadratic mean, :math:`X_n\xrightarrow[]{qm}X`, if

		.. math:: \lim\limits_{n\to\infty}\mathbb{E}[(X_n-X)^2]=0
	* Convergence in quadratic mean implies convergence in :math:`L_1`.

Almost surely convergence
====================================
.. note::
	* :math:`(X_n)_{n=1}^\infty` is said to be converging to :math:`X` almost surely (with probability 1), :math:`X_n\xrightarrow[]{as}X`, if

		.. math:: \mathbb{P}(\lim\limits_{n\to\infty} X_n=X)=1
	* This can be restated as follows: for any :math:`\epsilon>0`

		.. math:: \mathbb{P}(\lim\limits_{n\to\infty}|X_n-X|\geq\epsilon)=0
	* Note that the limit is inside.
	* Interpretation:

		* We can think that the sample space is represented as the set of sequences :math:`\{(x_n)_{n=1}^\infty\}`.
		* In this case, almost surely convergence would mean that there are only finite number of elements in this set where the limit doesn't converge to the value of the rv :math:`X`.
	* Almost surely convergence implies convergence in quadratic mean.

*********************************************
Limit Theorems
*********************************************

Here we deal with rvs of 3 special kind for a given sequence of rvs :math:`(X_n)_{n=1}^\infty`. Let the rvs be iid and has well defined mean :math:`\mu` and variance :math:`\sigma^2`.
.. note::
	* Let the sum rv be :math:`S_n=\sum_{i=1}^n X_i` for a given :math:`n`. We can think of a sequence of this as :math:`(S_n)_{n=1}^\infty`.

		* We note that :math:`\mathbb{E}[S_n]=n\mu` and :math:`\mathrm{Var}(S_n)\to\infty`.
	* Let the mean rv be :math:`M_n=\frac{S_n}{n}` for a given :math:`n`. We can think of a sequence of this as :math:`(M_n)_{n=1}^\infty`.

		* We note that :math:`\mathbb{E}[M_n]=\mu` and :math:`\mathrm{Var}(M_n)=\sigma^2/n`.
	* Let the standardised rv be :math:`Z_n=\frac{S_n-n\mu}{\sigma\sqrt{n}}` for a given :math:`n`. We can think of a sequence of this as :math:`(Z_n)_{n=1}^\infty`.

		* We note that :math:`\mathbb{E}[Z_n]=0` and :math:`\mathrm{Var}(M_n)=1`.

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
