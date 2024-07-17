
################################################################################
ML Breadth
################################################################################
Study Framework
********************************************************************************
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

Sample Questions
================================================================================
(a) https://www.geeksforgeeks.org/machine-learning-interview-questions/
(b) https://www.turing.com/interview-questions/machine-learning
(c) https://www.interviewbit.com/machine-learning-interview-questions/
(d) https://anywhere.epam.com/en/blog/machine-learning-interview-questions
(e) https://www.mygreatlearning.com/blog/machine-learning-interview-questions/

.. warning::
	* Set 1

		1. Explain one project where you faced a challenging or ambiguous problem statement and solved it. What was the business impact?
		2. How do you decide between the model complexity vs the latency budget (I mentioned this during my explanation)?
		3. What is SFT and why it is needed?
		4. What do you understand by PPO in RLHF?
		5. What are LoRA and QLoRA?
		6. Have you worked with other types of generative models like GAN or VAE?
		7. Tell me how GANs are trained. Objective function?
		8. What are some of the problems in training GANs? Said Mode Collapse and Vanishing Gradient (too string discriminator). Asked me to explain both.
		9. How are VAEs different from vanilla autoencoders?
		10. Explain the reparameterisation trick.
		11. For classification trees, what is the splitting criteria?
		12. How are Random Forests different from normal classification trees?
		13. What is regularisation and why do we need it? Explained in RR and DNN? What type of regulariser is used in RR? What is the L1 version called?
	* Set 2

		1. Do you have experience with LLMs?
		2. Explain offline selection problem in detail.
		3. What is the difference between offline selection and online ranking?
		4. What are the inputs and outputs of your triplet BERT model?
		5. Explain triplet BERT architecture, how is it different from normal BERT? Why do you need 3 copies of the identical towers and not just concatenate text with SEP token?
		6. How do you tackle embeddings of 3 different embeddings? 
		7. What is the meaning of doing a max-pooling over the features in terms of individual dimensions? 
		8. How is max-pooling different than doing concatenation first and then projection?
		9. Walk me through the entire BERT encoding process from input sequence in natural text to final the layer.
		10. Explain how wordpiece works. Explain the Embedding matrix. What are its dimensions?
		11. Why do we need positional encodings? Which part in the transformer layer requires this positional information?
		12. Why do we need to divide QK^T by sqrt(d_k)?
		13. Why do we need softmax?
		14. Why do we need residual connection?
		15. Explain the FC layer.
		16. What are your evaluation metrics and why do you use them?
		17. Do you know about metrics that are used for generation?
		18. Tell me some shortcomings of BLEU and ROUGE. What other metric can we use? How is perplexity defined?
		19. Do you know about Reflection? How would you evaluate LLM outputs for hallucination and other mistakes?

.. note::
	1. What is convolution Operation? Code it up.
	2. What is self attention?
	3. Derive gradient descent update rule for non negative matrix factorisation.
	4. Code non negative matrix factorisation.
	5. Derive gradient descent update rule for linear/logistic regression.
	6. Code stochastic gradient descent in linear/logistic regression setting.
	7. Code AUC.
	8. Questions related to my projects/thesis.
	9. One question from statistics: was related to Bayes theorem.
	10. Bias-variance tradeoff.
	11. Design questions: i remember only one, let's say some countries don't allow showing ads for knife, gun, etc, how would you go about building a system that can classify safe queries vs unsafe queries?
	12. What's a language model?
	13. Explain the working of any click prediction model.
	14. A couple of questions related to indexing in search engine.
	15. Convolution vs feedforward.

.. seealso::
	1. `Clustering evaluation. <https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation>`_

		- `Silhouette Coefficient <https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient>`_
		- `CH Index <https://scikit-learn.org/stable/modules/clustering.html#calinski-harabasz-index>`_
		- `DB Index <https://scikit-learn.org/stable/modules/clustering.html#davies-bouldin-index>`_
		- `Rand Index <https://scikit-learn.org/stable/modules/clustering.html#rand-index>`_
		
	2. How does batch norm help in faster convergence? [`Interesting read <https://blog.paperspace.com/busting-the-myths-about-batch-normalization/>`_]
	3. Why does inference take less memory than training?

Topics
********************************************************************************
Key Topics
================================================================================
* Feature Engineering
* Linear Regression and variants
* Boosted Trees, Random Forest
* Naive Bayes
* Logistic Regression	
* Support Vector Machines

Esoteric Topics
================================================================================
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

Even more esoteric topics
================================================================================
* Causal reasoning and diagnostics
* Recommender systems
* Learning latent representations
* Neural networks
* Causal networks

Sample Questions
********************************************************************************
GPT-generated Sample Questions for Outside-of-Resume Topics
================================================================================
1. Ensemble Learning:
--------------------------------------------------------------------------------
- Explain the concept of ensemble learning and the rationale behind combining multiple weak learners to create a strong learner. Provide examples of ensemble methods and their respective advantages and disadvantages.
- Can you discuss any ensemble learning techniques you've used in your projects, such as bagging, boosting, or stacking? How do you select base learners and optimize ensemble performance in practice?
- With the increasing popularity of deep learning models, how do you see the role of ensemble methods evolving in modern machine learning pipelines, and what are the challenges and opportunities in combining deep learning with ensemble techniques?

2. Dimensionality Reduction Techniques:
--------------------------------------------------------------------------------
- Discuss the importance of dimensionality reduction techniques in machine learning, particularly in addressing the curse of dimensionality and improving model efficiency and interpretability.
- Can you explain the difference between linear and non-linear dimensionality reduction methods, and provide examples of algorithms in each category? When would you choose one method over the other?
- Given the exponential growth of data in various domains, how do you adapt dimensionality reduction techniques to handle high-dimensional datasets while preserving meaningful information and minimizing information loss?

3. Model Evaluation and Validation:
--------------------------------------------------------------------------------
- Explain the concept of model evaluation and validation, including common metrics used for assessing classification, regression, and clustering models.
- Can you discuss any strategies or best practices for cross-validation and hyperparameter tuning to ensure robust and reliable model performance estimates?
- Given the prevalence of imbalanced datasets and skewed class distributions in real-world applications, how do you adjust model evaluation metrics and techniques to account for class imbalance and minimize bias in performance estimation?

4. Statistical Hypothesis Testing:
--------------------------------------------------------------------------------
- Discuss the principles of statistical hypothesis testing and the difference between parametric and non-parametric tests. Provide examples of hypothesis tests commonly used in machine learning and statistics.
- Can you explain Type I and Type II errors in the context of hypothesis testing, and how you control for these errors when conducting multiple hypothesis tests or adjusting significance levels?
- With the increasing emphasis on reproducibility and rigor in scientific research, how do you ensure the validity and reliability of statistical hypothesis tests, and what measures do you take to mitigate the risk of false positives or spurious findings?

5. Bayesian Methods and Probabilistic Modeling:
--------------------------------------------------------------------------------
- Explain the Bayesian approach to machine learning and its advantages in handling uncertainty, incorporating prior knowledge, and facilitating decision-making under uncertainty.
- Can you discuss any Bayesian methods or probabilistic models you've applied in your work, such as Bayesian regression, Gaussian processes, or Bayesian neural networks? How do you interpret and communicate Bayesian model outputs to stakeholders?
- Given the computational challenges of Bayesian inference, how do you scale Bayesian methods to large datasets and high-dimensional parameter spaces, and what approximation techniques or sampling methods do you use to overcome these challenges?
   
6. Graph Neural Networks (GNNs):
--------------------------------------------------------------------------------
- Explain the theoretical foundations of graph neural networks (GNNs) and their applications in recommendation systems and social network analysis.
- Can you discuss any challenges or limitations in training GNNs on large-scale graphs, particularly in scenarios with heterogeneous node types or dynamic graph structures?
- With the growing interest in heterogeneous information networks and multimodal data, how do you extend traditional GNN architectures to handle diverse types of nodes and edges, and what strategies do you employ to integrate different modalities effectively?

7. Causal Inference and Counterfactual Reasoning:
--------------------------------------------------------------------------------
- Discuss the importance of causal inference in machine learning applications, particularly in domains such as personalized recommendation systems and healthcare analytics.
- Can you explain the difference between causal inference and predictive modeling, and how you incorporate causal reasoning into the design and evaluation of machine learning models?
- Given the challenges of estimating causal effects from observational data, what techniques or methodologies do you use to address confounding variables and selection bias, and what are the limitations of these approaches?

8. Federated Learning and Privacy-Preserving Techniques:
--------------------------------------------------------------------------------
- Explain the concept of federated learning and its advantages in scenarios where data privacy and security are paramount, such as healthcare or financial services.
- Can you discuss any challenges or trade-offs in implementing federated learning systems, particularly in terms of communication overhead, model aggregation, and privacy guarantees?
- With the increasing regulatory scrutiny and consumer concerns around data privacy, how do you ensure compliance with privacy regulations such as GDPR or CCPA while leveraging data for model training and inference, and what techniques do you use to anonymize or encrypt sensitive information?

9. Meta-Learning and Transfer Learning:
--------------------------------------------------------------------------------
- Discuss the principles of meta-learning and its applications in few-shot learning, domain adaptation, and model generalization across tasks and datasets.
- Can you provide examples of meta-learning algorithms or frameworks you've worked with, and how they improve the efficiency and effectiveness of model adaptation and transfer?
- With the increasing complexity and diversity of machine learning models, how do you leverage transfer learning techniques to transfer knowledge from pre-trained models to new tasks or domains, and what strategies do you employ to fine-tune model parameters and hyperparameters effectively?

10. Interpretability and Explainable AI:
--------------------------------------------------------------------------------
- Explain the importance of model interpretability and explainability in machine learning, especially in domains such as finance, healthcare, and law enforcement.
- Can you discuss any techniques or methodologies for explaining black-box models, such as LIME, SHAP, or model distillation, and their advantages and limitations in different contexts?
- Given the trade-offs between model complexity and interpretability, how do you balance model performance with the need for transparency and accountability, and what strategies do you use to communicate complex model decisions to stakeholders or end-users?

Sample Interview Questions
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

Applied ML
--------------------------------------------------------------------------------
* What metrics are used for a heavily imbalanced dataset?

Related StackExchanges
================================================================================
.. note::
	* `stats.stackexchange <https://stats.stackexchange.com/>`_
	* `datascience.stackexchange <https://datascience.stackexchange.com/>`_
	* `ai.stackexchange <https://ai.stackexchange.com/>`_
