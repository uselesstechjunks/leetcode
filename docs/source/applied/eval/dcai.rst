###################################################################################
Data Centric AI
###################################################################################
.. note::
	* [youtube.com] `MIT: Introduction to Data-Centric AI <https://www.youtube.com/watch?v=ayzOzZGHZy4&list=PLnSYPjg2dHQKdig0vVbN-ZnEU0yNJ1mo5>`_

***********************************************************************************
Class Imbalance
***********************************************************************************
***********************************************************************************
Outliers
***********************************************************************************
***********************************************************************************
Observability
***********************************************************************************
.. note::

	* [arize.com] `Courses on Observability <https://courses.arize.com/courses/>`_

Distribution Shift
====================================================================================
.. note::
	* [mit.edu] `Class Imbalance, Outliers, and Distribution Shift <https://dcai.csail.mit.edu/2024/imbalance-outliers-shift/>`_	
	* [arize.com] `Drift Metrics: a Quickstart Guide <https://arize.com/blog-course/drift/>`_

Defitions
-------------------------------------------------------------------------------------
.. note::
	* Distribution shift: :math:`p_{\text{train}}(\mathbf{x},y)\neq p_{\text{test}}(\mathbf{x},y)`
	* Covariate shift: 

		* :math:`p_{\text{train}}(\mathbf{x})\neq p_{\text{test}}(\mathbf{x})`
		* :math:`p_{\text{train}}(y|\mathbf{x})=p_{\text{test}}(y|\mathbf{x})`
	* Concept shift:

		* :math:`p_{\text{train}}(\mathbf{x})=p_{\text{test}}(\mathbf{x})`
		* :math:`p_{\text{train}}(y|\mathbf{x})\neq p_{\text{test}}(y|\mathbf{x})`
	* Label shift:

		* Only in :math:`y\implies\mathbf{x}` problems.
		* :math:`p_{\text{train}}(y)\neq p_{\text{test}}(y)`
		* :math:`p_{\text{train}}(\mathbf{x}|y)=p_{\text{test}}(\mathbf{x}|y)`

Identification 
-------------------------------------------------------------------------------------
Detecting distribution shift requires monitoring both data drift (changes in input distribution) and concept drift (changes in target relationships).  

(A) Statistical & Distance-Based Methods  

	#. Kolmogorov-Smirnov (KS) Test / Jensen-Shannon Divergence (JSD)  
	
		- Measures difference in feature distributions between past data (training set) and new data (live traffic).  
		- Example: If the distribution of search queries in training data is significantly different from recent user queries, a shift is happening.  

	#. Population Stability Index (PSI)  
	
		- Tracks changes in feature distributions over time to identify shifts.  
		- Example: If a recommender systems user embeddings shift significantly, the model might be outdated.  

(B) Model Performance Monitoring  

	#. Live A/B Testing with Shadow Models  
	
		- Deploy a newer retrained model alongside the existing one, comparing engagement metrics (CTR, conversions, etc.).  
		- Example: If old models show declining CTR, while new models improve CTR, this signals distribution shift.  

	#. Error Analysis on Recent Queries  
	
		- Compare model predictions on new queries vs. actual user behavior.  
		- Example: If a search model ranks outdated news articles highly but users click on newer sources, concept drift has occurred.  

(C) Embedding-Based Drift Detection  

	#. Measuring Drift in Learned Representations (e.g., PCA, t-SNE)  
	
		- Compare embedding spaces of items and users from past vs. present data.  
		- Example: If embeddings from old user data cluster differently from new user behavior, a shift is occurring.  

	#. Contrastive Learning for Drift Detection  
	
		- Train an encoder on past interactions and compare with embeddings from new interactions.  
		- If new embeddings are significantly different, it signals a distribution shift.  
