################################################################################
ML Interview Prep Guide
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
		* What-if scenarios where training fails - mathematical issues (check stack-exchange).
	* Training and Validation

		* Design the training algorithm
		* Implementation and computational considerations including complexity.
		* How to check if algorithm converged.
		* What-if scenarios where training fails - computational issues (check stack-exchange).		
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

Sample Questions
================================================================================
Feature Engineering
--------------------------------------------------------------------------------
* When do we need to scale features?
* How to handle categorical features for

	* categories with a small number of possible values
	* categories with a very large number of possible values
	* ordinal categories (an order associated with them)

Mathematics
--------------------------------------------------------------------------------
* Different types of matrix factorizations. 
* How are eigenvalues related to singular values.

Statistics
--------------------------------------------------------------------------------
* You have 3 features, X, Y, Z. X and Y are correlated, Y and Z are correlated. Should X and Z also be correlated always?

Classical ML
--------------------------------------------------------------------------------
* Regression

	* What are the different ways to measure performance of a linear regression model.
* Naive Bayes

	* Some zero problem on Naive Bayes
* Trees

	* Difference between gradient boosting and XGBoost.

Related StackExchanges
================================================================================
.. note::
	* `stats.stackexchange <https://stats.stackexchange.com/>`_
	* `datascience.stackexchange <https://datascience.stackexchange.com/>`_
	* `ai.stackexchange <https://ai.stackexchange.com/>`_

********************************************************************************
ML Depth
********************************************************************************
Study Framework
================================================================================
Sample Questions
================================================================================
Generic
--------------------------------------------------------------------------------
* Can you explain how you handle scenarios with low data availability?
* Could you elaborate on the different sampling techniques you are familiar with?
* Can you explain the teacher-student paradigm in machine learning? When is a separate teacher model needed?
* Explain a portion from your paper.

Click Prediction
--------------------------------------------------------------------------------
* Can you discuss the pros and cons of Gradient Boosting Decision Trees (GBDT) with respect to Deep Neural Networks (DNNs)?
* Can you explain the personalization aspect of your Click Prediction model? 
* Can you use a collaborative Filtering approach to solve the Click Prediction problem?
* What are the key metrics that you consider when evaluating your CP model? 
* How do you determine when it needs retraining?
* How do you identify when things fail in your model or system?
* How did you handle categorical and ordinal features in your CP problem? 
* Why did you frame online-ranking as a CP problem for ranking and not as a learning to rank problem?

Encoder
--------------------------------------------------------------------------------
* Can you explain how BERT is trained? 
* How does BERT differ from models like GPT or T5? 
* Can you use BERT for text generation?
* What are the different BERT variants that you have experimented with? 
* How do you fine-tune a BERT-based model for your specific domain?
* What is a Sentence-BERT (SBERT) model? How is it different from normal BERT?
* How is SBERT trained and how do you evaluate its quality? 
* Other than BERT, what other Encoder Models do you know of?

Multilingual
--------------------------------------------------------------------------------
* How would you approach training a multilingual model?
* What are the key challenges and why this is hard to do?

Offline Ranking
--------------------------------------------------------------------------------
* Can you discuss the simulation strategy you used for offline ranking? 
* What are the pros and cons of the marginalization you had to perform? 

Personalization
--------------------------------------------------------------------------------
* Can you discuss the pros and cons of using a similarity score between a user’s history and an item to represent user interest?

GAN
--------------------------------------------------------------------------------
* How did you use the MMD estimator as a discriminator in a GAN? 
* What are the difficulties in training and using GANs? Are there better alternatives out there?

LLM
--------------------------------------------------------------------------------
* How do you go about fine-tuning a large language model?
* How did you select which prompts to use in your model? 
* Could you share some prompts that didn’t work and how you came up with better ones?

Statistics
--------------------------------------------------------------------------------
* Can you explain what non-parametric two-sample tests are and how they differ from parametric ones? 
* Could you provide the intuition behind the Maximum Mean Discrepancy (MMD) estimator that you used? 
* Do you know about Bayesian testing? Is Bayesian the same as non-parametric?

Linear Algebra
--------------------------------------------------------------------------------
* Can you list the linear algebra algorithms you are familiar with? 
* What is a rational approximation of an operation function? 
* Can you discuss the feature selection algorithms that you implemented? 
* What are linear operators? How do they differ from non-linear operators? 
* Can you explain the estimation strategy that you used in the approximation algorithm?

********************************************************************************
ML Applications
********************************************************************************
Study Framework
================================================================================
.. note::
	* Step 1: Identify the key problem and formulate it as ML problem. Ensure that solving it would achieve the goal.
	* Step 2: Solve the key problem and identify subproblems.
	* Step 3: Assume simplest solution to the subproblems and solve the key problem end-to-end.
	* Step 4: Talk about metrics, practical issues and hosting.
	* Step 5: Subproblems

		* Step 5a: Iterate over the subproblems and identify ones that can be solved by ML.
		* Step 5b: Solve the ML subproblems using step 2-6 in repeat until there are none left.
	* Step 6: Identify model degradation over time.

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
