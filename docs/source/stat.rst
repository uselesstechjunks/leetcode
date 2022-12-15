Statistics
#####################

Statistical Inference
******************************

We have a sample of size :math:`n` from an unknown distribution :math:`F`.

.. math::
    X_1,\cdots X_n \sim F

The task for statistical inference is to infer :math:`F` or some function of :math:`F` (also known as statistical functionals, such as density :math:`f=T(F)=F'`, expectation :math:`\mathbb{E}[X]=T(F)=\int_{-\infty}^{\infty} x dF` or variance :math:`\text{Var}(X)=T(F)=\mathbb[(\mathbb{E}[X]-X)^2]`) from the sample. If the rv is a tuple, e.g. :math:`(X_i,Y_i)_{i=1}^n`, then inference might also mean infering a *regression function* :math:`r(X)` for the conditional expectation, :math:`\mathbb{E}[Y|X]=r(X)+\epsilon`.

Statistical Model
======================

A statistical model :math:`\mathcal{F}` is set of distributions (or densities or regression functions).
