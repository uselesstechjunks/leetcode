################################################################################################
Fundamentals of Learning
################################################################################################

************************************************************************************************
Defining the Objective
************************************************************************************************
.. note::
	* Prerequisies:

		* High School Math

			* [Loney] Trigonometry, Coordinate Geometry
			* [Strang] Calculus Volume 1, 2, 3
		* Matrix Algebra

			* [Graybill] Matrices with Applications in Statistics - Chapter 4 Geometric Interpretation
			* [Springer] Matrix Tricks for Linear Statistical Models - Chapter Introduction
			* Matrix Cookbook - Identities - All Things Multinomial and Normal
		* Matrix Calculus

			* [Minka] `Old and New Matrix Algebra Useful for Statistics <https://tminka.github.io/papers/matrix/minka-matrix.pdf>`_
			* [Dattaro] `Convex Optimization - Appendix D <https://www.cs.cmu.edu/~epxing/Class/10701-08s/recitation/mc.pdf>`_
			* [Abadir Magnus] Matrix Algebra 
		* Probability Theory - Exponential Family, Graphical Models

			* [Bishop] Pattern Recognition and Machine Learning
		* Point Estimation

			* [Lehaman] Theory of Point Estimation - Chapter 1 Preparations
		* Information Theory
	* Estimating Densities

		* Divergence

			* Discriminative Models

				* Cross Entropy and Negative Log-Likelihood
				* Regression - Bayes Estimator: Conditional Expectation Solution
				* Classification - Bayes Estimator: MAP Solution
			* Latent Generative Models

				* Variational Lower Bounds
				* Gaussian Mixture Models
				* Probabilistic PCA
				* Variational Autoencoder
				* Denoising Probabilistic Diffusion
		* Integral Probability Metrics

			* MMD
			* Wasserstein Distance
	* Minmax Theory

		* Adversarial Objective: GAN
		* Constrained Objective Formulation

************************************************************************************************
Optimisation for Optimality
************************************************************************************************
.. note::
	* Prerequisies:

		* Matrix Algebra and Calculus - Geometric View, Identities
		* Taylor Approximation
	* Unconstrained: First and Second Order Methods

		* First Order Methods 

			* Exact: Gradient Descent Variants
			* Approximate: Stochastic Gradient Descent Variants
		* Second Order Methods

			* Exact: Newton's Method
			* Approximate: Gauss-Newton's Hessian Approximation
	* Constrained

		* Lagrange Multipliers
		* KKT
