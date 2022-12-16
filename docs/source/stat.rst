Statistics
#####################

Statistical Inference
******************************

We have a sample of size :math:`n` from an unknown distribution :math:`F`.

.. math::
    X_1,\cdots X_n \sim F

The task for statistical inference is to infer :math:`F` or some function of :math:`F`, :math:`T(F)`, (also known as statistical functionals), such as 

#. density :math:`T(F)=f=F'`
#. expectation :math:`T(F)=\mathbb{E}[X]=\int_{-\infty}^{\infty} x dF`
#. variance :math:`T(F)=\text{Var}(X)=\mathbb{E}[(\mathbb{E}[X]-X)^2]`
#. median: :math:`T(F)=F^{-1}(1/2)`

that best *explains* the data (for some given definition of *best* chosen beforehand, such as *mean-squared-error*). 

Machine Learning
======================
If the rv is a tuple, e.g. :math:`(X_i,Y_i)_{i=1}^n\sim F_{X,Y}`, then inference might mean infering a *regression function* :math:`r(X)` that fits the conditional expectation corresponding to :math:`F_{Y|X}`

.. math::
    T(F_{Y|X})=\mathbb{E}[Y|X]=r(X)+\epsilon

where :math:`\mathbb{E}[\epsilon]=0`. This inference is known as *learning* in Machine Learning (achieved via *training* on a given sample set) and *curve estimation* in statistics.

In the above case, an inference might also mean an inferring an unseen :math:`Y` by :math:`\hat{y}=r(x)` for a given :math:`X=x`. This is known as *inference* in Machine Learning and *prediction* in statistics.

.. note::
    #. Dependent and Independent Variable: :math:`X` is known as the independent variable (*features* in Machine Learning) and :math:`Y` is known as dependent variable (*target* in Machine Learning). Independent variables are usually a multidimensional vectors :math:`X=\mathbf{x}\in\mathbb{R}^d` for some :math:`d>1`.
    #. It can be proven that it is always possible to write a conditional expectation in the above form if the expected :math:`\epsilon` is :math:`0`.

Statistical Model
======================

A statistical model :math:`\mathcal{F}` is set of distributions (or other statistical functionals of interest). The following categories of models are based on the dimensionality of this set.

#. Parametric Model: If this set can be spanned by a finitely many parameters.
#. Non-parametric Model: Otherwise.

Example: If a model for regression is restricted to the set of affine functions

.. math::
    \mathcal{F}=\{r(x)=mx+c; m,c\in\mathbb{R}\}

then it's parametric. Similarly, if the model is a set of FFN (feed-forward networks) of a given size, then it is also parametric and the parameters of this model are the weights and biases in each layer.

If it is for densities, then a parametric model could be 

.. math::
    \mathcal{F}=\{f_X(x;\mu,\sigma)=\frac{1}{\sigma\sqrt{2\pi}}\exp\{\frac{1}{2\sigma}(x-\mu)^2);\mu\in\mathbb{R},\sigma\in\mathbb{R}^+\}

.. note::
    #. The factors that decide the number of parameters, such as choice of function-class, is independent of the inference process and is decided separately. These are often called *hyper-parameters*. 
    #. Since there are multiple items to choose before performing inference, it is useful to clarify the sequence:

        #. A metric of goodness of an estimator is chosen first.
        #. A model is chosen (i.e. hyperparameters).
        #. Inference is performed using computation involving the samples.

A non-parametric model for distributions can be the set of all possible cdfs.

Types of Inference
=========================

#. Point Estimation: An inferred single *best* estimate (i.e. a point) for the fixed, unknown qualtity of interest within the model. This estimate for a fixed, unknown quantity of interest, :math:`\theta`, is expressed as a function of the data

    .. math::
        \hat{\theta_n}=g(X_1,\cdots,X_n)

    Example: 

        #. a single distribution/density function (parameterised/non-parameterised)
        #. a single regression function
        #. a single value for expectation/variance/other moments
        #. a single prediction for a dependent variable with a given independent variable. etc. 

#. Confidence Set Estimation: An inferred set which traps the fixed, unknown value of our quality of interest with a pre-determined probability. 

    .. note::
        #. A :math:`1-\alpha` confidence interval for a real qualtity of interest :math:`\theta` is defined as :math:`\hat{C_n}=(a,b)` where :math:`\mathbb{P}(\theta\in\hat{C_n})\ge 1-\alpha`. 
        #. The task is to estimate :math:`\hat{a}=a(X_1,\cdots,X_n)` and :math:`\hat{b}=b(X_1,\cdots,X_n)` such that the above holds. 
        #. For vector quantities, this is expressed with sets instead of intervals.
        #. In regression setting, a confidence interval around the regression function can be thought of the set of functions which contains the true function with certain probabilty. However, this is usually never measured.

#. Hypothesis Testing:

Point Estimation
---------------------------
The estimate :math:`\hat{\theta_n}` depends on data and therefore is a rv (i.e. with a different sample, it evaluates to a different value).

.. note::
    #. Sampling Distribution: The distribution of :math:`\hat{\theta_n}` over different samples.
    #. Bias: :math:`\text{bias}(\hat{\theta_n})=\mathbb{E}_{\theta}[\hat{\theta_n}]-\theta`. If :math:`\text{bias}(\hat{\theta_n})=0`, then :math:`\hat{\theta_n}` is called an *unbiased estimator* of :math:`\theta`.
    #. Standard Error: :math:`\text{se}(\hat{\theta_n})=\sqrt{\text{Var}_{\theta}(\hat{\theta_n})}`.
    #. Consistent Estimator: If :math:`\hat{\theta_n}` converges in probability to true :math:`\theta`.
    #. Mean-Squared Error: :math:`\mathbb{E}_{\theta}[(\hat{\theta_n}-\theta)^2]=\text{bias}^2(\hat{\theta_n})+\text{Var}_{\theta}(\hat{\theta_n})`
    #. Theorem: If :math:`\text{bias}\to 0` and :math:`\text{se}\to 0` as :math:`n\to \infty`, then :math:`\hat{\theta_n}` is consistent.
    #. Asymptotically Normal Estimator: :math:`\hat{\theta_n}\approx\mathcal{N}(\theta,\hat{\text{se}}^2)`.

Confidence Set Estimation
---------------------------------------
.. note::
    #. Normal-based Confidence Interval: If :math:`\hat{\theta_n}` is an aysmptotically normal estimator of :math:`\theta`, then :math:`(\hat{\theta_n}-z_{\alpha/2}\hat{\text{se}},\hat{\theta_n}+z_{\alpha/2}\hat{\text{se}})` is a :math:`1-\alpha` confidence interval.

Hypothesis Testing
---------------------------------
