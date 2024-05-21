#########################################################################
The Big Picture of Probabilistic Machine Learning
#########################################################################

*************************************************************************
Probabilistic Framework
*************************************************************************

Defining the Framework
*************************************************************************
.. note::
	* For a given task, we can collect the stuff that we care about in a set :math:`\Omega`.
	* The goal of the probabilistic machine learning framework is to be able to define a probability measure, :math:`\mathbb{P}(\omega\in\Omega)`

		* Equivalently, we can define a (or a set of) random variable(s) :math:`S(\omega):2^{\Omega}\mapsto\mathbb{R}` and define a 

			* distribution :math:`F_S(s)=\mathbb{P}(S\leq s)` or a
			* density :math:`f_S(s)` such that 

				.. math:: F_S(s)=\int\limits_{-\infty}^s f_S(t)\mathop{d\mu(t)}
	* The types of items that can be in the set :math:`\Omega` can be quite diverse, and therefore the associated rv can have the range which can confront to different types of mathematical structures.

		* A single binary variable, :math:`S\in\{0,1\}`.
		* A categorical variable, :math:`S\in\{1,\cdots,K\}`.
		* Real number :math:`S\in\mathbb{R}`
		* Finite dimensional Euclidean vectors :math:`\mathbf{S}\in\mathbb{R}^d` with a common practice of associate each dimension with its own separate rv such that :math:`S_i\in\mathbf{R}`.
		* Infinite sequences :math:`(S)_{i=1}^\infty` where each :math:`S_i\in\mathbb{R}`.

Using the Framework
*************************************************************************
.. note::
	* We can use this framework to estimate some quantities of interest from distribution e.g. 
		
		* predict the mean :math:`\mathbb{E}_{S\sim F_S}[S]` [regression framework]
		* predict the mode :math:`s^*=\underset{s}{\arg\max} f_S(s)` [classification framework]
		* generate samples :math:`s^*=s\sim F_S` [generative framework]	
