#######################################################################
ML Application and System Design
#######################################################################
********************************************************************************
Framework
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
