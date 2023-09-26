##########################################################################################
Classical Statistical Inference
##########################################################################################
We have a sample of size :math:`n`, :math:`X_1,\cdots X_n` from an unknown CDF :math:`F`.

.. note::
	* The task for statistical inference is to infer :math:`F`, or some function of :math:`F`, or some other quantity that depends on :math:`F`, that best explains the data, for some criteria of **best** chosen beforehand.
	* The functions of this form, :math:`T(F)`, are known as **statistical functionals**, such as 

		* density :math:`T(F)=f=F'`
		* expectation :math:`T(F)=\mathbb{E}[X]=\int x \mathop{dF}`
		* variance :math:`T(F)=\text{Var}(X)=\mathbb{E}[(\mathbb{E}[X]-X)^2]`
		* median: :math:`T(F)=F^{-1}(1/2)`   
	* The **inferred values** are called **estimates** of the quantities of interest. 
	* The **expression** that computes these estimates from samples is called an **estimator**.

.. attention::
	Estimates are rv as their values may change subject to a different sample.

******************************************************************************************
Machine Learning as a Statistical Inference
******************************************************************************************
.. note::
	* We have iid samples from an unknown joint CDF, e.g. :math:`(X_i,Y_i)_{i=1}^n\sim F_{X,Y}`.
	* **Model inference**: Model inference means estmating the conditional expectation corresponding to :math:`F_{Y|X}` with a **regression function** :math:`r(X)` such that

		.. math::
		    T(F_{Y|X})=\mathbb{E}[Y|X]=r(X)+\epsilon

	  where :math:`\mathbb{E}[\epsilon]=0`. 

		* This inference is known as **learning** in Machine Learning and **curve estimation** in statistics.
	* **Variable inference**: In the above case, a variable inference means estimating an unseen :math:`Y|X=x` by :math:`\hat{Y}=\hat{y}=r(x)` for a given :math:`X=x`. 

		* This is known as **inference** in Machine Learning and **prediction** in statistics.

.. note::
	Dependent and Independent Variable: 

	* :math:`X` is called the independent variable (**features**) and :math:`Y` called as dependent variable (**target**). 
	* Independent variables are often multidimensional vectors :math:`X=\mathbf{x}\in\mathbb{R}^d` for some :math:`d>1`.	

******************************************************************************************
Statistical Model
******************************************************************************************
.. note::
	A statistical model :math:`\mathcal{F}` is set of distributions (or other statistical functionals of interest). 

.. attention::
    * The process that decides the model, such as choice of function-class or number of parameters, is independent of the inference and is performed separately beforehand. In ML, these are called **hyper-parameters**. 
    * Since there are multiple items to choose before performing inference, it is useful to clarify the sequence:

        #. A metric of goodness of an estimator is chosen first.
        #. A model is chosen (such as, hyperparameters).
        #. Inference is performed using computation involving the samples.
        #. Quality of model is judged by evaluating the model on the inference data.
        #. (Optional) A different model is chosen and the process repeats.

Types of Statistical Model
==========================================================================================
The following categories of models are based on the dimensionality of this set.

Parametric Model
------------------------------------------------------------------------------------------
.. note::
	This set can be spanned by a finitely many parameters.

.. seealso::
	* A regression model is defined by the set of affine functions

		.. math:: \mathcal{F}=\{r(x)=mx+c; m,c\in\mathbb{R}\}
	* If the regression model is a set of feed-forward networks (FFN) of a given size, then it is also parametric and the parameters of this model are the weights and biases in each layer.
	* If the task is to estimate densities, then a parametric model could be 

		.. math::
			\mathcal{F}=\{f_X(x;\mu,\sigma)=\frac{1}{\sigma\sqrt{2\pi}}\exp\{\frac{1}{2\sigma}(x-\mu)^2);\mu\in\mathbb{R},\sigma\in\mathbb{R}^+\}

Non-parametric Model
------------------------------------------------------------------------------------------
.. note::
	This set cannot be spanned by a finitely many parameters.

.. seealso::
	A non-parametric model for distributions can be the set of all possible cdfs.

Empirical distribution function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The estimator for :math:`F` is :math:`\hat{F_n}` which assigns a mass :math:`1/n` to every point in sample :math:`\{X_i\}_{i=1}^n`.

.. note::		
	For a given :math:`x`,

		* :math:`\mathbb{E}[\hat{F_n}(x)]=F(x)`
		* :math:`\text{Var}(\hat{F_n})=\frac{F(x)(1-F(x))}{n}`

Plug-in Estimator
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
.. note::
	The plug-in estimator for any statistical functional :math:`T(F)` can be obtained by replacing it with :math:`\hat{F_n}` as :math:`T(\hat{F_n})`.

******************************************************************************************
Types of Inference
******************************************************************************************

Point Estimation
==========================================================================================
.. note::
	* A single **best** estimate (point) for the fixed, unknown qualtity of interest :math:`\theta` within the model.
	* This estimate of :math:`\theta` is expressed as a function of the data :math:`\hat{\theta_n}=g(x_1,\cdots,x_n)`.
	* The estimator :math:`\hat{\Theta_n}` is a rv (i.e. with a different sample, it evaluates to a different value :math:`\hat{\theta_n}`).
	* Examples: 

		#. a single distribution/density function (parameterised/non-parameterised)
		#. a single regression function
		#. a single value for expectation/variance/other moments
		#. a single prediction for a dependent variable with a given independent variable. etc. 

Some useful terminology
-------------------------------------------------------------------------------------------
.. note::
	* **Sampling Distribution**: The distribution of :math:`\hat{\Theta_n}` over different samples.
	* **Estimation Error**: :math:`\tilde{\Theta_n}=\hat{\Theta_n}-\theta`.
	* **Bias**: :math:`\text{b}_\theta(\hat{\Theta_n})=\mathbb{E}_{\theta}[\tilde{\Theta_n}]=\mathbb{E}_{\theta}[\hat{\theta_n}]-\theta`. 
	* **Standard Error**: :math:`\text{se}_\theta(\hat{\Theta_n})=\sqrt{\mathbb{V}_{\theta}(\hat{\Theta_n})}`.
	* If the variance in above is also an estimate (as it often is), then we estimate SE as :math:`\sqrt{\hat{\mathbb{V}}_{\theta}(\hat{\Theta_n})}`.

.. note::
	* **Unbiased Estimator**: If :math:`\text{b}_\theta(\hat{\Theta_n})=0`.
	* **Asymptotically Unbiased Estimator**: If :math:`\hat{\Theta_n}\xrightarrow[]{L_1}\theta`.
	* **Consistent Estimator**: If :math:`\hat{\Theta_n}\xrightarrow[]{P}\theta`.
	* **Mean-Squared Error**: :math:`\text{mse}_\theta(\hat{\Theta}_n)=\mathbb{E}_{\theta}[\tilde{\Theta_n}^2]=\mathbb{E}_{\theta}[(\hat{\Theta_n}-\theta)^2]=\text{b}_\theta^2(\hat{\Theta_n})+\mathbb{V}_{\theta}(\hat{\Theta_n})`.

.. attention::
	Theorem: If :math:`\lim\limits_{n\to\infty}\text{b}_\theta(\hat{\Theta_n})=0` and :math:`\lim\limits_{n\to\infty}\text{se}_\theta(\hat{\Theta_n})=0` then :math:`\hat{\Theta_n}` is consistent.

.. note::
    * **Asymptotically Normal Estimator**: :math:`\hat{\Theta_n}\xrightarrow[]{D}\mathcal{N}(\theta,\hat{\text{se}_\theta}^2(\hat{\Theta_n}))`.
    * Empirical distribution function is a consistent estimator for any distribution.

Confidence Set Estimation
==========================================================================================
.. note::
	An estimated set which traps the fixed, unknown value of our quality of interest with a pre-determined probability. 

.. attention::
	#. A :math:`1-\alpha` confidence interval (CI) for a real qualtity of interest :math:`\theta` is defined as :math:`\hat{C_n}=(a,b)` where :math:`\mathbb{P}(\theta\in\hat{C_n})\ge 1-\alpha`. 
	#. The task is to estimate :math:`\hat{a}=a(X_1,\cdots,X_n)` and :math:`\hat{b}=b(X_1,\cdots,X_n)` such that the above holds. 
	#. For vector quantities, this is expressed with sets instead of intervals.
	#. In regression setting, a confidence interval around the regression function can be thought of the set of functions which contains the true function with certain probabilty. However, this is usually never measured.

Some useful terminology
-------------------------------------------------------------------------------------------
.. note::
	* **Pointwise Asymptotic CI**: :math:`\forall\theta,\liminf\limits_{n\to\infty}\mathbb{P}_{\theta}(\theta\in\hat{C_n})\ge 1-\alpha`
	* **Uniform Asymptotic CI**: :math:`\liminf\limits_{n\to\infty}\inf\limits_{\theta\in\Theta}\mathbb{P}_{\theta}(\theta\in\hat{C_n})\ge 1-\alpha`

		* Uniform Asymptotic CI is stricter.
	* **Normal-based CI**: If :math:`\hat{\theta_n}` is an aysmptotically normal estimator of :math:`\theta`, then a :math:`1-\alpha` confidence interval is given by

		.. math:: (\hat{\theta_n}-z_{\alpha/2}\hat{\text{se}},\hat{\theta_n}+z_{\alpha/2}\hat{\text{se}})
	
		* The above is a pointwise asymptotic CI.

For the empirical distribution model, following are some interesting results.

.. note::
    * **Glivenko-Cantelli Theorem**: :math:`||\hat{F_n}(x)-F(x)||_\infty=\sup_{x}|\hat{F_n}(x)-F(x)|\xrightarrow[]{as} 0`.
    * **Dvoretzsky-Kiefer-Wolfowitz (DKW) Inequality**: For any :math:`\epsilon>0`,
    
        .. math::
            \mathbb{P}(\sup_x|\hat{F_n}(x)-F(x)|>\epsilon) \le 2\exp(-2n\epsilon^2)

    * It can be derived from DKW that we can form a :math:`1-\alpha` CI of width :math:`2\epsilon_n` around :math:`\hat{F_n}` where :math:`\epsilon_n=\sqrt{\frac{1}{2n}\ln(\frac{2}{\alpha})}`.

Hypothesis Testing
==========================================================================================
.. note::
	* This helps to evaluate how good a statistical model is given samples. 
	* Assuming a fixed statistical model, we compute estimates for certain quantities of interest, which can then be compared with the same quantity assuming the model is correct. 
	* The task is then to arrive at probabilistic statements about how different these two are.

.. attention::
	#. The statement about the quantity of interest assuming the model is correct is called the **Null hypothesis**.
	#. The statement where the model is incorrect is called **Alternate hypothesis**.
	#. [TODO:CHECK IF TRUE] If we create a :math:`1-\alpha` confidence set for the estimated quantity and the quantity as-per-model doesn't fall within this set, then we **reject** the null hypothesis with significance level :math:`1-\alpha`.  If it does then we **fail to reject** the null hypothesis.
