################################################################################
ML Application
################################################################################
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
Topics for Revision
********************************************************************************
* End Goal:  

	.. note::
	
		- Can I explain the inner workings of transformers, diffusion models, and LLM fine-tuning techniques?  
		- Can I walk through the end-to-end design of an ML system confidently?  
		- Am I able to break down an ambiguous problem into structured ML components?  

* Plan Outline:  

	.. note::
	
		- Days 1-3: Build ML and deep learning foundation  
		- Day 4: Deep dive into applied ML & system design  
		- Day 5: Mock interviews & reinforcement  

Day 1: ML Fundamentals & Core Deep Learning Concepts (4-5 hours)  
================================================================================
.. note::
	Objective: Refresh fundamental ML concepts and deep learning theory, ensuring a strong foundation.  

Topics to Cover:  
--------------------------------------------------------------------------------
1. Supervised & Unsupervised Learning Basics  

	- Bias-variance tradeoff, overfitting, regularization, cross-validation  
	- Optimization techniques: SGD, Adam, Momentum  
	- Feature selection and feature engineering  
2. Deep Learning Core Concepts  

	- Neural network architectures: CNNs, RNNs, Transformers  
	- Backpropagation & optimization in deep learning  
	- Attention mechanisms & self-attention  
3. Probabilistic Thinking in ML  

	- Bayesian ML, Gaussian Processes, Uncertainty Estimation  
	- Graph-based models (e.g., Probabilistic Graphical Models)  

Suggested Readings & Materials:  
--------------------------------------------------------------------------------
Papers  

	- "Understanding Machine Learning: From Theory to Algorithms" – Shalev-Shwartz & Ben-David (Chapters 1-3)  
	- "Deep Learning" – Ian Goodfellow et al. (Chapters 6-9 for deep learning core concepts)  
Videos  

	- MIT 6.S191: Introduction to Deep Learning – Lecture 1 & 2 (YouTube)  
	- CS229: Machine Learning – Stanford (Andrew Ng’s lectures)  

Practice Questions:  
--------------------------------------------------------------------------------
	- Explain the key trade-offs in choosing different ML models (e.g., trees vs. deep learning vs. probabilistic models).  
	- Given a dataset with heavy class imbalance, what strategies would you use?  
	- What are the main challenges when optimizing deep networks? 

Day 2: Generative AI & Large Language Models (LLMs) Essentials (4-5 hours)  
================================================================================
.. note::
	Objective: Develop a deep understanding of LLMs, transformers, generative models, and diffusion models.  

Topics to Cover:  
--------------------------------------------------------------------------------
1. Transformer Models & Self-Attention  

	- Attention mechanisms, Multi-Head Attention, Positional Encoding  
	- Pretraining vs. Fine-tuning in LLMs  
2. Training and Inference Optimization  

	- Parameter-efficient fine-tuning methods (LoRA, adapters)  
	- Quantization and distillation for LLMs  
3. Diffusion Models & GANs  

	- How diffusion models work and where they are used (e.g., DALL-E, Stable Diffusion)  
	- How they compare to GANs for generative modeling  

Suggested Readings & Materials:  
--------------------------------------------------------------------------------
Papers  

	- "Attention Is All You Need" – Vaswani et al. (Transformer architecture)  
	- "Scaling Laws for Neural Language Models" – Kaplan et al. (Important for LLM scaling)  
	- "Denoising Diffusion Probabilistic Models" – Ho et al. (Key diffusion model paper)  
Videos  

	- Yannic Kilcher’s explainer on Transformers & LLMs (YouTube)  
	- Andrej Karpathy’s "State of GPT" talk  

Practice Questions:  
--------------------------------------------------------------------------------
	- How does self-attention work in transformers?  
	- Why do LLMs require large-scale pretraining, and what are some methods to reduce compute requirements?  
	- Compare GANs and diffusion models in terms of training stability and quality of generated content.  

Day 3: Applied ML in E-commerce & First-Principles Thinking  (4-5 hours)  
================================================================================
.. note::
	Objective: Understand how ML is applied in e-commerce and practice solving open-ended ML problems.  

Topics to Cover:  
--------------------------------------------------------------------------------
1. Personalization & Recommendations  

	- Collaborative filtering, Matrix Factorization, Deep Learning for Recommendations  
	- Cold start problem and hybrid approaches  
2. Fraud Detection & Marketplace Integrity  

	- Anomaly detection methods, semi-supervised learning  
	- Behavioral modeling for fraud prevention  
3. Search & Ranking in E-commerce  

	- Learning-to-Rank (LTR) approaches  
	- RAG-based models for search  
4. Conversational AI & Generative AI in E-commerce  

	- AI-powered chatbots for customer support  
	- Product image & description generation  

Suggested Readings & Materials:  
--------------------------------------------------------------------------------
Papers  

	- "Deep Learning Based Recommender System: A Survey and New Perspectives" – Zhang et al.  
	- "A Survey on Learning to Rank for Information Retrieval" – Liu et al.  
	- "BERT for E-commerce Search" – Amazon AI Paper  

Videos  

	- DeepMind’s talk on "Learning to Rank" (YouTube)  
	- Stanford CS330: Personalized AI Models  

Practice Questions:  
--------------------------------------------------------------------------------
	- How would you design a ranking algorithm for a search engine?  
	- Suppose an e-commerce company wants to detect fraud in seller transactions. What approach would you take?  
	- How can generative AI be used to automate product catalog generation?  

Day 4 (Weekend): End-to-End ML System Design & Case Studies  (8+ hours)  
================================================================================
.. note::
	Objective: Work on end-to-end ML system design, focusing on real-world case studies.  

Topics to Cover:  
--------------------------------------------------------------------------------
1. ML System Design Framework  

	- Problem formulation, data pipeline, model selection, serving infrastructure  
	- Latency vs. Accuracy trade-offs in production systems  
2. Scaling ML Systems for Millions of Users  

	- Distributed training & inference optimization  
	- Model monitoring & retraining strategies  
3. Applied ML Case Studies  

	- End-to-end design of a large-scale recommendation system  
	- ML-based fraud detection pipeline  
	- Building a generative AI-based product description generator  

Suggested Readings & Materials:  
--------------------------------------------------------------------------------
Papers  

	- "Machine Learning: The High-Interest Credit Card of Technical Debt" – Sculley et al.  
	- "TFX: A TensorFlow-Based Production-Scale Machine Learning Platform" – Baylor et al.  
Videos  

	- ML System Design - Stanford CS329S  
	- Chip Huyen’s talk on ML in Production  

Practice Questions:  
--------------------------------------------------------------------------------
	- Design a real-time personalized feed ranking system for an e-commerce company.  
	- How would you ensure that ML models in production do not degrade over time?  
	- Design a fraud detection pipeline that scales across millions of transactions.  

Day 5 (Weekend): Mock Interviews & Final Review  (8+ hours)  
================================================================================
.. note::
	Objective: Reinforce learning, work on mock interviews, and refine your explanations.  

Activities:  
--------------------------------------------------------------------------------
1. Mock Interviews (4-5 hours)  

	- Practice answering end-to-end ML system design problems out loud  
	- Get a friend or use a platform like pramp/interviewing.io  
2. Concept Review & Weak Area Focus (3-4 hours)  

	- Revise key LLM, ML, and system design concepts  
	- Solve additional case studies  
3. Behavioral & Culture Fit Preparation  

	- STAR method for answering leadership & impact questions  
	- Reflect on past projects where you applied ML in production  

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

