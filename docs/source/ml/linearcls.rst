######################################################################################
Linear Methods for Classification
######################################################################################
.. note::
	* We note that the output here is a categorical rv :math:`G\in\mathcal{G}` where :math:`|\mathcal{G}|=K`. 
	* We therefore associate each observed output :math:`g_i=k` where :math:`k=1,2,\cdots,K`.
	* For classification, we can always assign a different colour to each datapoint :math:`x_i` in the input space :math:`\mathcal{X}` as per the output class :math:`k` it belongs to.
	* The decision boundary in that case is the **partition boundary** in the input space between different coloured inputs.
	* A classifier is **linear** if the boundary can be expressed as linear equations involving :math:`\mathbf{x}_j`.
	* We can extend linear classifier to create non-linear decision boundary in the original input space by using transforms, such as basis expansion.

**************************************************************************************
Probabilistic Classifiers
**************************************************************************************
.. note::
	* We can define a **discriminant function** :math:`\delta_k(x)` for each class :math:`k`.
	* For each :math:`x\in\mathcal{X}`, the classification prediction then becomes

		.. math:: g^*=\underset{k}{\arg\max}\delta_k(x)
	* For a linear classifier, we need some monotone transform :math:`h` of :math:`\delta_k` to be linear.

		* :math:`h` can very well be just the identity function.
		* The decision boundary between :math:`k=1` and :math:`k=K` is given by the surface where

			.. math:: h(\delta_1(x))-h(\delta_K(x))=0
	* For probabilistic classifiers, the discriminant function is usually defined as the posterior probability.

		.. math:: \delta_k(x)=\mathbb{P}(G=k|X=x)
	* If :math:`\log` is used as the monotone transform, then the decision boundary forms the logit function

		.. math:: \log\frac{\mathbb{P}(G=1|X=x)}{\mathbb{P}(G=K|X=x)}=\log\delta_1(x)-\log\delta_K(x)

		* At the decision boundary, the posterior probabilities are equal.

			.. math:: \log\delta_1(x)-\log\delta_K(x)=0

Generative Models
======================================================================================
.. note::
	* It follows from Bayes theorem that

		.. math:: \mathbb{P}(G=k|X=x)\propto\mathbb{P}(G=k)\times\mathbb{P}(X=x|G=k)=\pi_k\times f_k(x)

		* :math:`\pi_k=\mathbb{P}(G=k)` is the **class prior** and it parameterises a :math:`\mathrm{Multinoulli}(\pi_1,\cdots,\pi_k)` over the classes.
		* :math:`f_k(x)=\mathbb{P}(X=x|G=k)` is the **conditional data-density per class** :math:`k`.

.. tip::
	* We note that since we're interested in the arg max, we won't be needing to compute the normalisation constant in the denominator as that's the same for all classes.
	* If we assume that the in-class data density is Gaussian, then we have LDA and QDA classifiers.

Inference
--------------------------------------------------------------------------------------
Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. warning::
	* For generative models, we usually consider the joint likelihood

		.. math:: \mathbb{P}(X_1=x_1,\cdots,X_N=x_N,G_1=g_i,\cdots,G_N=g_N)=\prod_{i=1}^{N}\mathbb{P}(G_i=g_i)\times\mathbb{P}(X=x_i|G_i=g_i)=\prod_{i=1}^{N}\pi_{g_i}\times f_{g_i}(x_i)	
	* If :math:`f_k` is parametric in :math:`\theta`, we use MLE to estimate those parameters.

		.. math:: \hat{f}_k(x;\theta)=f_k(x;\hat{\theta}_{\text{MLE}})
	* Otherwise. we resort to non-parametric density estimation methods to estimate :math:`\hat{f}_k(x)`.

Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. tip::
	.. math:: \hat{g}=\underset{k}{\arg\max}\left(\hat{\pi}_k\times \hat{f}_k(x)\right) 

Quadratic Discriminator Analysis
--------------------------------------------------------------------------------------
.. note::
	* We assume the conditional data density to be Gaussian for each class

		.. math:: f_k(x)=\frac{1}{|\boldsymbol{\Sigma}_k|^{1/2}\left(2\pi\right)^{d/2}}\exp(-\frac{1}{2}(x-\mu_k)^T\boldsymbol{\Sigma}_k^{-1}(x-\mu_k))
	* We note that

		.. math:: \log(\pi_k\times f_k(x))=\log(\pi_k)-\frac{1}{2}\log(|\boldsymbol{\Sigma}_k|)-\frac{d}{2}\log(2\pi)-\frac{1}{2}(x-\mu_k)^T\boldsymbol{\Sigma}_k^{-1}(x-\mu_k)
	* We can define :math:`\delta_k(x)=\log(\pi_k)-\frac{1}{2}\log(|\boldsymbol{\Sigma}_k|)-\frac{1}{2}(x-\mu_k)^T\boldsymbol{\Sigma}_k^{-1}(x-\mu_k)`
	* The decision boundary between :math:`k=1` and :math:`k=2` is given by the surface

		.. math:: \log\frac{\delta_1(x)}{\delta_2(x)}=\log\frac{\pi_1}{\pi_2}-\log\frac{|\boldsymbol{\Sigma}_1|}{|\boldsymbol{\Sigma}_2|}-\frac{1}{2}(x-\mu_1)^T\boldsymbol{\Sigma}_1^{-1}(x-\mu_1)+\frac{1}{2}(x-\mu_2)^T\boldsymbol{\Sigma}_2^{-1}(x-\mu_2)=0
	* We note that this is quadratic in :math:`x`.

Linear Discriminator Analysis
--------------------------------------------------------------------------------------
.. note::
	* If we model the conditional density in a way such that they all share the covariance (:math:`\boldsymbol{\Sigma}`), then the equation simplifies to a linear one in :math:`x` as the quadratic term :math:`x^T\boldsymbol{\Sigma}^{-1}x` cancels.

		.. math:: x^T\boldsymbol{\Sigma}^{-1}x-\mu_1^T\boldsymbol{\Sigma}^{-1}x-x^T\boldsymbol{\Sigma}^{-1}\mu_1+\mu_1^T\boldsymbol{\Sigma}^{-1}\mu_1-x^T\boldsymbol{\Sigma}^{-1}x+\mu_2^T\boldsymbol{\Sigma}^{-1}x+x^T\boldsymbol{\Sigma}^{-1}\mu_2-\mu_2^T\boldsymbol{\Sigma}^{-1}\mu_2=2x^T\boldsymbol{\Sigma}^{-1}(\mu_2-\mu_1)+\left(\mu_1^T\boldsymbol{\Sigma}^{-1}\mu_1-\mu_2^T\boldsymbol{\Sigma}^{-1}\mu_2\right)
	* The decision boundary between :math:`k=1` and :math:`k=2` is given by the hyperplane

		.. math:: \log\frac{\delta_1(x)}{\delta_2(x)}=\log\frac{\pi_1}{\pi_2}+x^T\boldsymbol{\Sigma}^{-1}(\mu_1-\mu_2)-\frac{1}{2}(\mu_1-\mu_2)^T\boldsymbol{\Sigma}^{-1}(\mu_1-\mu_2)=0
	* We note that this is linear in :math:`x`.

.. tip::
	* Let :math:`N_k=\sum_{i=1}^N\mathbb{I}_{g_i=k}` be the number of labels belonging to a class :math:`k`.
	* We estimate the priors using MLE

		.. math:: \hat{\pi}_k=\frac{N_k}{N}
	* The conditional density parameters are also estimated using MLE.
		
		* Mean

			.. math:: \hat{\mu}_k=\frac{\sum_{g_i=k}x_i}{N_k}
		* Covariance
		
			.. math:: \hat{\boldsymbol{\Sigma}}=\frac{1}{N-K}\sum_{k=1}^K\sum_{g_i=k} (x_i-\hat{\mu}_k)(x_i-\hat{\mu}_k)^T

Regularised Discriminator Analysis
--------------------------------------------------------------------------------------
.. note::
	* As a compromise between QDA and LDA, we can decompose each of the class-covariance matrix into a pooled (shared) matrix and a class-specific matrix.

		.. math:: \hat{\boldsymbol{\Sigma}}_k(\alpha)=\alpha\hat{\boldsymbol{\Sigma}}_k+(1-\alpha)\hat{\boldsymbol{\Sigma}}
	* The shared-covariance matrix can be further decomposed into a diagonal one (uncorrelated covariates) and one which contains the correlations.

		.. math:: \hat{\boldsymbol{\Sigma}}(\gamma)=\gamma\hat{\boldsymbol{\Sigma}}+(1-\gamma)\hat{\sigma}^2\mathbf{I}
	* Both these versions form a regularised version of the QDA with :math:`\alpha` and :math:`\gamma` as hyperparameters.

Discriminative Models
======================================================================================
.. note::
	* Here, instead of invoking Bayes theorem, we can directly focus on modeling the logit as a linear function of :math:`x`.
	* For each class :math:`k=1,2,\cdots,K-1`, we can define the logits in terms of a set of linear equations

		.. math:: \log\frac{\mathbb{P}(G=k|X=x)}{\mathbb{P}(G=K|X=x)}=\beta_{0,k}+\beta_{1:,k}^Tx

		* Here, each :math:`\beta_{0,k}\in\mathbb{R}` is the bias (intercept) term and :math:`\beta_{1:,k}\in\mathbb{R}^d` is the weight vector.
		* We can use the notation :math:`\beta_k=(\beta_{0,k}, \beta_{1:,k})^T\in\mathbb{R}^{d+1}`.
	* This can be achieved if we define the density as the softmax, i.e. for :math:`k=1,2,\cdots,K-1`

		.. math:: \mathbb{P}(G=k|X=x)=\frac{\exp(\beta_{0,k}+\beta_{1:,k}^Tx)}{1+\sum_{j=1}^{K-1}\exp(\beta_{0,j}+\beta_{1:,j}^Tx)}
	* The final probability can just be defined in terms of others

		.. math:: \mathbb{P}(G=K|X=x)=\frac{1}{1+\sum_{j=1}^{K-1}\exp(\beta_{0,j}+\beta_{1:,j}^Tx)}
	* This formulation too defines a multinoulli probability distribution for the output variable once we observe :math:`x`

		.. math:: G\sim\mathrm{Multinoulli}(p_1,\cdots,p_k)
	* If we use the notation where :math:`\theta=(\beta_0,\cdots,\beta_{K-1})` represents the param vector, then this multinoulli density can be parameterised in terms of

		.. math:: p_k=p_G(k|x;\theta)=\mathbb{P}(G=k|X=x)

Inference
--------------------------------------------------------------------------------------
Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. warning::
	* For discriminative models, we usually consider the conditional likelihood

		.. math:: \mathbb{P}(G_1=g_i,\cdots,G_N=g_N|X_1=x_1,\cdots,X_N=x_N)=\prod_{i=1}^{N}\mathbb{P}(G_i=g_i|X=x_i)=\prod_{i=1}^{N}p_G(g_i|x_i;\theta)
	* We use MLE to estimate the parameters :math:`\theta`.

Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. tip::
	.. math:: \hat{g}=\underset{k}{\arg\max}\left(\hat{p}_k\right) 

Logistic Regression
--------------------------------------------------------------------------------------
.. note::
	* For :math:`|\mathcal{G}|=2` (binary classification), :math:`G\sim\text{Bernoulli}(p_x(\beta))` with :math:`p_G(1|x;\theta)=p_x(\beta)` and :math:`p_G(2|x;\theta)=1-p_x(\beta)` where 

		* :math:`\beta=(\beta_{0,1},\beta_{1:,1})^T` and
		* :math:`p_x(\beta)=\frac{\exp(\beta^Tx)}{1+\exp(\beta^Tx)}` is the **sigmoid function**.
	* We introduce a dummy output variable :math:`y` such that

		* :math:`y_i=1\iff g_i=1`
		* :math:`y_i=0\iff g_i=2`
	* The log likelihood in this case can be written as

		.. math:: l(\theta)=\sum_{i=1}^{N}\log(p_G(g_i|x_i;\theta))=\sum_{i=1}^{N}y_i\log(p_{x_i}(\beta))+(1-y_i)\log(1-p_{x_i}(\beta))=f(\beta)
	* This is the Binary Cross Entropy (BCE) loss.

.. tip::
	* To estimate, we need to maximise the MLE.

		.. math:: \frac{\partial f}{\partial \beta}=\frac{\partial}{\partial \beta}\left(\sum_{i=1}^{N}y_i\log(\frac{\exp(\beta^Tx_i)}{1+\exp(\beta^Tx_i)})+(1-y_i)\log(\frac{1}{1+\exp(\beta^Tx_i)})\right)=\sum_{i=1}^N x_i(y_i-p_{x_i}(\beta))
	* This can be rewritten in terms of matrix equations as :math:`\mathbf{X}^T(\mathbf{y}-\mathbf{p})`.
	* We can perform gradient descent, or even Newton's method which involves computing the second derivative

		.. math:: \frac{\partial^2 f}{\mathop{\partial\beta}\mathop{\partial\beta^T}}=-\sum_{i=1}^N x_ix_i^Tp_{x_i}(\beta)(y_i-p_{x_i}(\beta))=-\mathbf{X}^T\mathbf{W}\mathbf{X}
	* Here :math:`\mathbf{W}` is the diagonal matrix with entries of :math:`p_{x_i}(\beta)(y_i-p_{x_i}(\beta))`.

Comparison Between LDA and Logistic Regression
======================================================================================
.. tip::
	* LDA and LR performs similar and they both evaluate to linear logits. But the way we estimate the parameters of this linear model separates them.
	* LDA assumes additional structure for the marginal data distribution because of Gaussian nature of the class-conditional density.
	* On the other hand, LR assumes no structure for the marginal data distribution. We can think of it as if we're free to fit any non-parametric density for the marginal, such as empirical distribution.
	* Since LDA makes additional assumption about the sturcture, we can estimate them with lower variance if the underlying data density indeed follows a Gaussian.
	* However, since outliers play a significant role in how the covariance is estimated, it is not robust to outliers/mislabelled examples.
	* If the data is perfectly separable by a hyperplane, the MLE formulation for LR is ill-defined.

**************************************************************************************
Hyperplane Classifiers
**************************************************************************************
.. note::
	* Here, instead of relying on a discriminator function, we directly model the separation boundary as a piece-wise hyperplane between classes.

Perceptron
======================================================================================

Optimal Hyperplane Classifier
======================================================================================
Separable Case
--------------------------------------------------------------------------------------
Non-Separable Case
--------------------------------------------------------------------------------------
Support Vector Machines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
