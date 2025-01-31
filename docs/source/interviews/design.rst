################################################################################
ML Application
################################################################################
********************************************************************************
Useful References:
********************************************************************************
1. Real-World Recommendation System Series

	- `Part 1 <https://blog.fennel.ai/p/real-world-recommendation-system?s=w>`_
	- `Part 2 <https://blog.fennel.ai/p/real-world-recommendation-systems?s=r>`_
	- `Part 3 <https://blog.fennel.ai/p/real-world-recommendation-systems-21e>`_
2. Feature Engineering for Recommendations

	- `Feature Engineering Blog <https://blog.fennel.ai/p/feature-engineering-for-recommendation>`_
	- `Feature Logging <https://fennel.ai/blog/feature-engineering-for-recommendation-031/>`_
	- `Information Leakage <https://www.fennel.ai/blog/two-types-of-information-leakage/>`_

********************************************************************************
General System Design interview Tips 
********************************************************************************
	#. Start with documenting your summary/overview in Google docs/Excalidraw or Zoom whiteboard. Even if the company hasn’t provided a link and interviewer insists on the conversation to be purely verbal - Document key bullet points. 
	#. Present your interview systematically; lead the conversation and don't wait for the interviewer to ask questions. At the beginning of the interview, present the discussion's structure and ask the interviewer about their main areas of interest. 
	#. Show your understanding of the business implications by sharing insights on metrics. Understand what the product truly expects from you. 
	#. Actively listen to the interviewer. At the start, ask: "What are you primarily looking for?". Address the whole process, from collecting and labeling data to defining metrics. 
	#. Assess the importance of the modeling process. 
	#. Familiarize yourself with the nuances of ML-Ops, such as: At the start of the interview, get a feel for if the interviewer seems interested in ML-Ops. You'll mostly get a clear signal on whether or not they are interested. 

		#. Managing model versions 
		#. Training models 
		#. Using model execution engines 
	#. Keep your resume at hand and review it before starting the interview.

********************************************************************************
Recommendation System
********************************************************************************
1. Problem Setup and Label Collection
================================================================================
a. Clarifying Questions

	- Understand the problem context and objectives.
	- Identify constraints and requirements.
b. Definition of Success

	- Define key performance metrics (e.g., accuracy, precision, recall, business metrics).
c. Positive and Negative Labels

	i. Different Options to Define Labels:

		1. Joining a group.
		2. Retention after a week.
		3. Interaction with other users.
		4. Meaningful interaction (e.g., time spent, making friends in the group).
	ii. Fairness Considerations:

		- Ensure adequate data for underrepresented groups.
d. Label Generation

	i. Engagement as Proxy:

		- User click as a positive label, no click as a negative label.
	ii. Use of Labelers:

		1. Utilize semi-supervised or unsupervised methods (e.g., clustering) to enhance labeler efficiency.
		2. Consider visits in a session (e.g., Pinterest or DoorDash) as similar pins or restaurants.
e. Downsampling the Dominant Class using Negative Sampling

	- Only downsample training data while keeping validation and test distributions unchanged.
f. Bias in Training

	- Limit the number of labels per user, video, or restaurant to prevent bias towards active users or popular items.

2. Feature Engineering
================================================================================
a. User Features

	- Demographic information, historical behavior, preferences.
b. Group Features

	- Attributes of the group or community.
c. Cross Features Between Group and Users

	- Interaction-based features.
d. Contextual Features

	- Time of day, holiday, device type, network connection (WiFi vs. 4G).
e. Feature Selection Process

	- Start with basic counters/ratios and refine using Gradient Boosted Decision Trees (GBDT).

3. Modeling
================================================================================
a. Two-Tower Model

	- Separate embedding models for users and items.
b. Embedding Creation

	- Graph embeddings and learned representations.
c. Retrieval (Optimized for Recall)

	- Collaborative filtering-based approaches.
d. Diversification of Sources

	- Ensure variety in retrieved results.
e. Ranking (Optimized for Precision)

	1. Two-Tower Model.
	2. Precision-focused optimization.

4. Measurement
================================================================================
a. Offline vs. Online Evaluation

	- Offline metrics (precision, recall) vs. online business impact.
b. Key Metrics

	- NDCG (Normalized Discounted Cumulative Gain)
	- Precision\@Top-K: Measures relevance of top-K recommendations.
	- Mean Average Precision (MAP\@K): Mean of AP\@K across users.
c. Explanation of Metrics

	- Justify metric choice at each evaluation stage.
d. Online Measurement

	- Prioritize business metrics.
	- Conduct A/B testing or Multi-Armed Bandit experiments.

5. Debugging
================================================================================
a. Structured Debugging Approach

	- Maintain a clear, written log of issues and solutions.
b. Online vs. Offline Model Debugging

	- Identify discrepancies between offline validation and real-world performance.

6. Feature Logging
================================================================================
a. Training Phase

	- Ensure consistency in feature storage and retrieval.
b. Debugging

	- Log model inputs and outputs for analysis.

7. Preparing the Training Pipeline
================================================================================
	- Automate feature extraction, model training, and validation.
	- Ensure reproducibility and scalability.

8. Deployment
================================================================================
a. Novelty Effects

	- Account for temporary engagement spikes post-deployment.
b. Model Refresh Impact

	- Understand how periodic updates influence engagement.

9. Stages of a Ranking System Funnel
================================================================================
	- Retrieval: Reduce millions of candidates to thousands.
	- Filtering: Remove irrelevant or outdated candidates.
	- Feature Extraction: Ensure consistency in train-test splits.
	- Ranking: Apply advanced models to refine selections.

10. Advanced Topics
================================================================================
a. Data Pipeline & Infrastructure

	- Efficient data ingestion, storage, and processing at scale.
	- Real-time vs. batch data pipelines.
	- Feature freshness and consistency.

b. Scalability & Latency Considerations

	- Low-latency serving strategies.
	- Trade-offs between model complexity and inference speed.
	- Caching, pre-computation, and model distillation.

c. Handling Model Drift & Monitoring

	- Detection of data drift and performance degradation.
	- Automated retraining strategies.
	- Monitoring feature distribution shifts over time.

d. Fairness, Interpretability, and Ethics
	
	- Fairness-aware learning to mitigate biases.
	- Interpretability techniques like SHAP, LIME.
	- Ethical considerations in AI-driven recommendations.

********************************************************************************
Paradigms For Applications
********************************************************************************
* Classification 

	* Semantic analysis 
	* Learning to rank 
* Regression 
* Clustering 

	* Anomaly detection 
	* User understanding
* Dimensionality reduction 

	* Topic models
	* Inferred suggestions
* Generative modeling 

	* Structured prediction
* Multimodal learning

********************************************************************************
ML Design Round Framework
********************************************************************************
(a) https://www.youtube.com/watch?v=jkKAeIx7F8c

Basic Structure
================================================================================
* Problem Understanding:

	- Functional Requirements: Identify the key business problem and the KPIs for success.
	- Non-functional Requirements: Ask about the additional requirement such as

		- imposing compliance policies (geographic, demographic)
		- additional desirable features (diversity, context-awareness, ability to 
* Problem Identification:

	- Abstraction: Think about the observed data as :math:`X` and the target as :math:`Y` (can be :math:`X` itself).

		* Does 'X' have structure (sequence: language, timeseries; locality: image, graph) or is it unstructured (can be shuffled)?
		* Are there latent variables :math:`Z`?
	- Mapping: Identify ML paradigms. If you can't map to of any, create a new ML paradigm for it!
* Scale Identification:

	- Think about the scale and discuss trade-offs for using different types of ML models for that paradigm. 
	- Decide on a scale for the current problem and draw system diagram. Mark the parts involving ML.
* ML cycle for each parts:

	* Working solution:

		- Uses a SOTA/novel technique.
		- Solves at the right scale.
		- Can go live.
	* Various trade-offs:

		- Model choice (e.g. Offline: DNNs/LLMs; Online: LR, GBDT and NN).
		- Loss (e.g. Imbalanced Dataset: weighted/focal loss).
		- Hyperparameter (overfitting; convergence).
		- Metric (e.g. RecSys: NDCG/MAP for PC vs MRR for Mobile; Classification: P, ROC-AUC vs R, PR-AUC).
	* Identify shortcomings:

		- Parts that can be iterated on.

********************************************************************************
Broad Application Domains
********************************************************************************
Recommendation and Search
================================================================================
Retrieval
--------------------------------------------------------------------------------
(a) retrieval based on query - query can be text or images (image search)
(b) query-less personalised retrieval for homepage reco (Netflix/YT/Spotify/FB/Amzn homepage)
(c) item-specific recommendation for "suggested items similar to this"

Ranking
--------------------------------------------------------------------------------
(d) context-aware online ranking (CP model or some ranking model)

Policy Enforcement
--------------------------------------------------------------------------------
(e) fraud detection
(f) policy compliance models (age restriction, geo restriction, banned-item restriction)

********************************************************************************
Sample Questions
********************************************************************************
* Design a system for QA where a user would be able to search with a query and the system answers from an internal knowledge-base.
* What would you do to reduce the latency in the system further?
* How would you apply a content restriction policy in the system (not all users would be able to search through all the knowledge-base).

