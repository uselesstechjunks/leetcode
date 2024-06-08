################################################################################
Sample Questions
################################################################################

********************************************************************************
ML Breadth
********************************************************************************
Study Framework
================================================================================
.. note::
	* Problem

		* Problem description and assumptions for simplicity.
	* Approach and Assumptions

		* Theoretical framework & motivation.
		* Mathematical derivation of training objective (loss) with boundary conditions.
		* What-if scenarios where training fails - mathematical issues.
	* Training and Validation

		* Training algorithm
		* Implementation and computational considerations including complexity.
		* What-if scenarios where training fails - computational issues.
		* How to check if algorithm converged.
	* Testing and Model Selection

		* How to check for overfitting/underfitting. Remedies?
		* Metrics to check - different choices and trade-offs.
		* How to tune hyperparameters and perform model selection.
	* Inference

		* Computational considerations.
		* Identify signs for model degradation over time. Remedies?

Key Topics
================================================================================
.. warning::
	* Feature Engineering
	* Linear Regression and variants
	* Boosted Trees, Random Forest
	* Naive Bayes
	* Logistic Regression	
	* Support Vector Machines

Esoteric Topics
================================================================================
.. warning::
	* Ordinal Regression - predicts a class label/score (check `this <https://home.ttic.edu/~nati/Publications/RennieSrebroIJCAI05.pdf>`_)
	* Learning To Rank - predicts a relative-order (MAP, DCG/NDCG, Precision@n, Recall@n, MRR)
	* Dimensionality Reduction - t-SNE, Spectral Clustering, PCA, Latent-variable models, NMF
	* Clustering & Anomaly Detection - DBSCAN, HDBSCAN, Hierarchical Clustering, Self-Organizing Maps, Isolation Forest, K-Means
	* Bayesian linear regression
	* Gaussian Processes
	* Graphical Models, Variational Inference, Belief Propagation, Deep Belief Net, LDA, CRF
	* NER, Pos-tagging, ULMFit
	* FaceNet, YOLO
	* Reinforcement learning: SARSA, explore-exploit,  bandits (eps-greedy, UCB, Thompson sampling), Q-learning, DQN - applications

********************************************************************************
ML Depth
********************************************************************************
Study Framework
================================================================================

********************************************************************************
ML Applications: Framework
********************************************************************************
Study Framework
================================================================================
.. note::
	* Step 1: Identify the key problem and formulate it as ML problem. Ensure that solving it would achieve the goal.
	* Step 2: Solve the key problem and identify subproblems.
	* Step 3: Assume simplest solution to the subproblems and solve the key problem end-to-end.
	* Step 4: Talk about metrics, practical issues and hosting.
	* Step 5: Iterate over the subproblems and identify ones that can be solved by ML.
	* Step 6: Solve the ML subproblems using step 2-6 in repeat until there are none left.

Problem Domains
================================================================================
.. warning::
	* Classification 
	* Generative modeling 
	* Regression 
	* Clustering 
	* Dimensionality reduction 
	* Density estimation 
	* Anomaly detection 
	* Data cleaning 
	* AutoML 
	* Association rules 
	* Semantic analysis 
	* Structured prediction 
	* Feature engineering 
	* Feature learning 
	* Learning to rank 
	* Grammar induction 
	* Ontology learning 
	* Multimodal learning

********************************************************************************
Theoretical Background
********************************************************************************

Statistical Learning: Probability, Statistics, Learning Theory
================================================================================
.. note::
	* You're given a sample from :math:`F_X` of size :math:`N`, and I give you an estimator :math:`\hat{x}`. Write down the MSE expression and break it down into bias and variance terms.
	* Suppose you know the underlying data distribution, :math:`F_X`. What estimator would you choose so that it minimizes MSE?
	* I explain to you the conditional mean estimator for regression. I give you two ways to approximate this by averaging.

		* For every point :math:`x`, check it's :math:`K` neighbours, check their values and average them. The average becomes your estimate.
		* You define some rectangular regions, with some homogeneaty criteria - points that fall within the same region more or less have very similar values.

	  which one of thse would have higher bias than the other? Can you explain the trends in bias vs variance if I allow you to have tiny rectangular regions, vs larger rectangular regions?
	* I explain to you the MAP estimator for conditional density for classification. Say, you have sample from two joint distributions and you want to build a MAP estimate classifier. I tell you to model the densities as Gaussian. Can you explain how do you come up with the classification rule? If those Gaussians share their covariance, does that simplify things?
	* I give you a system where you can have trees only upto 10 nodes. But you have the option to get multiple of them running in parallel. Can you use this system to do better than individual ones? What type of error would your approach reduce?

Classical ML: Optimisation
================================================================================
.. note::
	* Max-Margin classifiers

		* Constrained convex optimisation - KKT conditions
		* Separable non-separable case.
	* Linear regression - ridge, LASSO.
	* How do you move beyond linearity? Basis expansion. Infinite dimensional expansion using kernels.
	* Explain gradient descent, stochastic gradient descent, co-ordinate descent

		* How does that work for non-convex error surfaces?
		* How do you identify that you're in a local minima?

********************************************************************************
Related StackExchanges
********************************************************************************
.. note::
	* `stats.stackexchange <https://stats.stackexchange.com/>`_
	* `datascience.stackexchange <https://datascience.stackexchange.com/>`_
	* `ai.stackexchange <https://ai.stackexchange.com/>`_

********************************************************************************
Sample Questions
********************************************************************************
Feature Engineering
================================================================================
.. note::
	* When do we need to scale features?
	* How to handle categorical features for

		* categories with a small number of possible values
		* categories with a very large number of possible values
		* ordinal categories (an order associated with them)

Mathematics
================================================================================
.. note::
	* Different types of matrix factorizations. 
	* How are eigenvalues related to singular values.

Statistics
================================================================================
.. note::
	* You have 3 features, X, Y, Z. X and Y are correlated, Y and Z are correlated. Should X and Z also be correlated always?

Classical ML
================================================================================
.. note::
	* Regression

		* What are the different ways to measure performance of a linear regression model.
	* Naive Bayes

		* Some zero problem on Naive Bayes
	* Trees

		* Difference between gradient boosting and XGBoost.
