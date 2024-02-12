######################################################################################
Linear Methods for Classification
######################################################################################
.. note::
	* We note that the output here is a categorical rv :math:`G\in\mathcal{G}` where :math:`|\mathcal{G}|=K`. 
	* We therefore associate each observed output :math:`g_i=k` where :math:`k=1,2,\cdots,K`.
	* For classification, we can always assign a different colour to each datapoint :math:`x_i` in the input space :math:`\mathcal{X}` as per the output class :math:`k` it belongs to.
	* The decision boundary in that case is the **partition boundary** in the input space between different coloured inputs.
	* A classifier is **linear** iff the boundary can be expressed as a (set of) linear equations involving :math:`\mathbf{x}_j`.

**************************************************************************************
Probabilistic Classifiers
**************************************************************************************
.. note::
	* We can define a discriminator function :math:`\delta_k(x)` for each class :math:`k`.
	* For each :math:`x\in\mathcal{X}`, the classification prediction then becomes

		.. math:: g^*=\underset{k}{\arg\max}\delta_k(x)
	* [WHY??] For a linear classifier, we need some monotone transform of :math:`\delta_k` to be linear.
	* For probabilistic classifiers, the discriminator function is usually defined as the posterior probability.

		.. math:: \delta_k(x_i)=\mathbb{P}(G=k|X=x_i)
	* The monotone linear transform here is often the logit function

		.. math:: \log\frac{\mathbb{P}(G=1|X=x_i)}{\mathbb{P}(G=K|X=x_i)}=\log\delta_1(x_i)-\log\delta_K(x_i)

Generative Models
======================================================================================
.. note::
	* It follows from Bayes theorem that

		.. math:: \mathbb{P}(G=k|X=x_i)\propto\mathbb{P}(G=k)\times\mathbb{P}(X=x_i|G=k)=\pi_k\times f_k(x_i)

		* :math:`\pi_k=\mathbb{P}(G=k)` is the **class prior** probability.
		* :math:`f_k(x_i)=\mathbb{P}(X=x_i|G=k)` is the density of the data under a particular class :math:`k`.
	* We note that since we're interested in the arg max, we won't be needing to compute the normalisation constant in the denominator as that's the same for all classes.
	* If we assume that the in-class data density is Gaussian, then we have LDA and QDA classifiers.

Quadratic Discriminator Analysis
--------------------------------------------------------------------------------------

Linear Discriminator Analysis
--------------------------------------------------------------------------------------

Discriminative Models
======================================================================================
.. note::
	* Here, instead of invoking Bayes theorem, we can directly focus on modeling the logit as a linear function of :math:`x`.
	* For each class :math:`k=1,2,\cdots,K-1`, we can define the logits in terms of a set of linear equations

		.. math:: \log\frac{\mathbb{P}(G=k|X=x_i)}{\mathbb{P}(G=K|X=x_i)}=\beta_{0,k}+\beta_{1:,k}^Tx_i

		* Here, each :math:`\beta_{0,k}\in\mathbb{R}` is the bias (intercept) term and :math:`\beta_{1:,k}\in\mathbb{R}^d` is the weight vector.
		* We can use the notation :math:`\beta_k=(\beta_{0,k}, \beta_{1:,k})\in\mathbb{R}^{d+1}`.
	* This can be achieved if we define the density as the softmax, i.e. for :math:`k=1,2,\cdots,K-1`

		.. math:: \mathbb{P}(G=k|X=x_i)=\frac{\exp(\beta_{0,k}+\beta_{1:,k}^Tx_i)}{1+\sum_{j=1}^{K-1}\exp(\beta_{0,j}+\beta_{1:,j}^Tx_i)}
	* The final probability can just be defined in terms of others

		.. math:: \mathbb{P}(G=K|X=x_i)=\frac{1}{1+\sum_{j=1}^{K-1}\exp(\beta_{0,j}+\beta_{1:,j}^Tx_i)}
	* This formulation defines a multinoulli probability distribution for the output variable.
	* If we use the notation where :math:`\theta=(\beta_0,\cdots,\beta_{K-1})` represents the param vector, then this multinoulli density can be written as

		.. math:: \mathbb{P}(G=k|X=x_i)=p_G(k|x_i;\theta)

**************************************************************************************
Hyperplane Classifiers
**************************************************************************************
.. note::
	* Here, instead of relying on a discriminator function, we directly model the separation boundary as a piece-wise hyperplane between classes.
