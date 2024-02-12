######################################################################################
Linear Methods for Classification
######################################################################################
.. note::
	* We note that the output here is a categorical rv :math:`G\in\mathcal{G}` where :math:`|\mathcal{G}|=K`. 
	* We therefore associate each observed output :math:`g_i=k` where :math:`k=1,2,\cdots,K`.
	* For classification, we can always assign a different colour to each datapoint :math:`x_i` in the input space :math:`\mathcal{X}` as per the output class :math:`k` it belongs to.
	* The decision boundary in that case is the **partition boundary** in the input space between different coloured inputs.
	* A classifier is **linear** if the boundary can be expressed as linear equations involving :math:`\mathbf{x}_j`.

**************************************************************************************
Probabilistic Classifiers
**************************************************************************************
.. note::
	* We can define a discriminator function :math:`\delta_k(x)` for each class :math:`k`.
	* For each :math:`x\in\mathcal{X}`, the classification prediction then becomes

		.. math:: g^*=\underset{k}{\arg\max}\delta_k(x)
	* [WHY??] For a linear classifier, we need some monotone transform of :math:`\delta_k` to be linear.
	* For probabilistic classifiers, the discriminator function is usually defined as the posterior probability.

		.. math:: \delta_k(x)=\mathbb{P}(G=k|X=x)
	* The monotone linear transform here is often the logit function

		.. math:: \log\frac{\mathbb{P}(G=1|X=x)}{\mathbb{P}(G=K|X=x)}=\log\delta_1(x)-\log\delta_K(x)

Generative Models
======================================================================================
.. note::
	* It follows from Bayes theorem that

		.. math:: \mathbb{P}(G=k|X=x)\propto\mathbb{P}(G=k)\times\mathbb{P}(X=x|G=k)=\pi_k\times f_k(x)

		* :math:`\pi_k=\mathbb{P}(G=k)` is the **class prior** probability.
		* :math:`f_k(x)=\mathbb{P}(X=x|G=k)` is the density of the data under a particular class :math:`k`.
	* We note that since we're interested in the arg max, we won't be needing to compute the normalisation constant in the denominator as that's the same for all classes.
	* If we assume that the in-class data density is Gaussian, then we have LDA and QDA classifiers.

Inference
--------------------------------------------------------------------------------------
.. warning::
	* For generative models, we usually consider the joint likelihood

		.. math:: \mathbb{P}(X_1=x_1,\cdots,X_N=x_N,G_1=g_i,\cdots,G_N=g_N)=\prod_{i=1}^{N}\mathbb{P}(G_i=g_i)\times\mathbb{P}(X=x_i|G_i=g_i)=\prod_{i=1}^{N}\pi_{g_i}\times f_{g_i}(x_i)
	* We use MLE to estimate the parameters of :math:`f_k`.

Quadratic Discriminator Analysis
--------------------------------------------------------------------------------------

Linear Discriminator Analysis
--------------------------------------------------------------------------------------

Discriminative Models
======================================================================================
.. note::
	* Here, instead of invoking Bayes theorem, we can directly focus on modeling the logit as a linear function of :math:`x`.
	* For each class :math:`k=1,2,\cdots,K-1`, we can define the logits in terms of a set of linear equations

		.. math:: \log\frac{\mathbb{P}(G=k|X=x)}{\mathbb{P}(G=K|X=x)}=\beta_{0,k}+\beta_{1:,k}^Tx

		* Here, each :math:`\beta_{0,k}\in\mathbb{R}` is the bias (intercept) term and :math:`\beta_{1:,k}\in\mathbb{R}^d` is the weight vector.
		* We can use the notation :math:`\beta_k=(\beta_{0,k}, \beta_{1:,k})\in\mathbb{R}^{d+1}`.
	* This can be achieved if we define the density as the softmax, i.e. for :math:`k=1,2,\cdots,K-1`

		.. math:: \mathbb{P}(G=k|X=x)=\frac{\exp(\beta_{0,k}+\beta_{1:,k}^Tx)}{1+\sum_{j=1}^{K-1}\exp(\beta_{0,j}+\beta_{1:,j}^Tx)}
	* The final probability can just be defined in terms of others

		.. math:: \mathbb{P}(G=K|X=x)=\frac{1}{1+\sum_{j=1}^{K-1}\exp(\beta_{0,j}+\beta_{1:,j}^Tx)}
	* This formulation defines a multinoulli probability distribution for the output variable

		.. math:: G\sim\mathrm{Multinoulli}(p_1,\cdots,p_k)
	* If we use the notation where :math:`\theta=(\beta_0,\cdots,\beta_{K-1})` represents the param vector, then this multinoulli density can be parameterised in terms of

		.. math:: p_k=p_G(k|x;\theta)=\mathbb{P}(G=k|X=x)

Inference
--------------------------------------------------------------------------------------
.. warning::
	* For discriminative models, we usually consider the conditional likelihood

		.. math:: \mathbb{P}(G_1=g_i,\cdots,G_N=g_N|X_1=x_1,\cdots,X_N=x_N)=\prod_{i=1}^{N}\mathbb{P}(G_i=g_i|X=x_i)=\prod_{i=1}^{N}p_G(g_i|x_i;\theta)
	* We use MLE to estimate the parameters :math:`\theta`.

**************************************************************************************
Hyperplane Classifiers
**************************************************************************************
.. note::
	* Here, instead of relying on a discriminator function, we directly model the separation boundary as a piece-wise hyperplane between classes.
