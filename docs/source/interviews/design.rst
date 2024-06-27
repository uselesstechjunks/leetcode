################################################################################
ML Application
################################################################################
********************************************************************************
Benchmark
********************************************************************************
(a) https://www.youtube.com/watch?v=jkKAeIx7F8c

Sample Questions
================================================================================
.. note::
	* Design a system for QA where a user would be able to search with a query and the system answers from an internal knowledge-base.

		* What would you do to reduce the latency in the system further?
		* How would you apply a content restriction policy in the system (not all users would be able to search through all the knowledge-base).

********************************************************************************
Framework [Needs Revision]
********************************************************************************
.. note::
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
Recommendation and Search
********************************************************************************
Retrieval
================================================================================
(a) retrieval based on query - query can be text or images (image search)
(b) query-less personalised retrieval for homepage reco (Netflix/YT/Spotify/FB/Amzn homepage)
(c) item-specific recommendation for "suggested items similar to this"

Ranking
================================================================================
(d) context-aware online ranking (CP model or some ranking model)

Policy Enforcement
================================================================================
(e) fraud detection
(f) policy compliance models (age restriction, geo restriction, banned-item restriction) 

********************************************************************************
Problem Domains
********************************************************************************
.. warning::
	* Classification 

		* Semantic analysis 
		* Learning to rank 
	* Regression 
	* Clustering 

		* Anomaly detection 
	* Dimensionality reduction 
	* Generative modeling 
	
		* Structured prediction 	
	* Multimodal learning
