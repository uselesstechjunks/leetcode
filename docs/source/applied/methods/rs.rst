####################################################################################
Search & Recommendation
####################################################################################
.. contents:: Table of Contents
   :depth: 3
   :local:
   :backlinks: none

************************************************************************************
Survey Papers
************************************************************************************
* [le-wu.com][Ranking] `A Survey on Accuracy-Oriented Neural Recommendation <https://le-wu.com/files/Publications/JOURNAL/A_Survey_of_Neural_Recommender_Systems.pdf>`_
* [arxiv.org][Retrieval] `A Comprehensive Survey on Retrieval Methods in Recommender Systems <https://arxiv.org/pdf/2407.21022>`_
* [arxiv.org][CTR] `Deep Learning for Click-Through Rate Estimation <https://arxiv.org/abs/2104.10584>`_
* [mdpi.com] `A Comprehensive Survey of Recommender Systems Based on Deep Learning <https://www.mdpi.com/2076-3417/13/20/11378/pdf?version=1697524018>`_
* [arxiv.org][SSL] `Self-Supervised Learning for Recommender Systems: A Survey <https://arxiv.org/abs/2203.15876>`_

************************************************************************************
Overview
************************************************************************************
.. warning::

	* Overview of search and recsys - different stages
	* Metrics, Modelling for different stages
	* Application of LLMs at different stages
	* General problems
	* Domain specific problems

.. important::
	- Entities

		- Users, items (text, image, video, nodes), interactions, context
	- Labels

		- Supervised, semi-supervised (proxy label), self-supervised, unsupervised
	- Patterns

		- Query-Item, User-Item, Item-Item, Session, User-User
	- Objectives & metrics

		- Accuracy Precision@k, Recall@k, MAP@k, NDCG@k, MRR@k, HR@k
		- Behavioral Diversity, Novelty, Serendipity, Popularity-bias, Personalisation, Fairness
		- Monitoring Drift metrics
	- Considerations in model training

		- Training window Seasonality, Data leak
		- Deciding on labels
	- Stages

		- Retrieval, Filtering, Rerank
	- Models

		- Retrieval

			- Content-based Filtering
			- Collaborative Filtering - MF/Neural CF
			- GCN - LightGCN
			- Sequence - Transformers
		- Filtering

			- Ruled based
		- Rerank
		
			- GBDT, NN, DCN, WDN, DPP
	- Domains

		- Search Advertising
		- Music
		- Video
		- E-commerce
		- Social media
	- Issues

		- General

			#. Cold-start
			#. Diversity vs. personalization Trade-Off
			#. Popularity bias & fairness
			#. Short-term engagement vs. long-term user retention trade-off
			#. Privacy concerns & compliance (GDPR, CCPA)
			#. Distribution shift (data/input, concept/target)
		- Advanced

			#. Multi-touch Attribution
			#. Real-time personalization & latency trade-Offs
			#. Cross-device and cross-session personalization
			#. Multi-modality & cross-domain recommendation challenges
		- Domain-Specific

			#. Search Query understanding & intent disambiguation
			#. E-Commerce Balancing revenue & user satisfaction
			#. Video & Music Streaming Content-length bias in recommendations

Metrics
====================================================================================
Accuracy
------------------------------------------------------------------------------------
.. csv-table::
	:header: "Metric", "Full Name", "Formula", "Desc", "Drawback"
	:align: center
	
		HR@k, Hit-rate at k, , ,
		Recall@k, Recall at k, , ,
		NDCG@k, Normalized Discounted Cumulative Gain at k, , ,

Popularity Bias
------------------------------------------------------------------------------------
.. note::
	* :math:`U`: Set of all users
	* :math:`I`: Set of all items
	* :math:`L_u`: List of items (concatenated) impressed for user :math:`u`
	* :math:`L`: All list of items (concatenated)

.. csv-table::
	:header: "Metric", "Full Name", "Formula", "Note", "Drawback"
	:align: center
	
		ARP, Average Recommendation Popularity, :math:`\frac{1}{|U|}\sum_{u\in U}\frac{\sum_{i\in L_u}\phi(i)}{|L_u|}`, Average CTR across users, Good (low) value doesn't indicate coverage
		Agg-Div, Aggregate Diversity, :math:`\frac{|\bigcup_{u\in U}L_u|}{|I|}`, Item Coverage, Doesn't detect skew in impression
		Gini, Gini Index, :math:`1-\frac{1}{|I|-1}\sum_{k}^{|I|}(2k-|I|-1)p(i_k|L)`, :math:`p(i_k|L)` how many times :math:`i_k` occured in `L`, Ignores user preference
		UDP, User Popularity Deviation, , ,

Diversity
------------------------------------------------------------------------------------
Personalsation
------------------------------------------------------------------------------------
Issues
====================================================================================
Distribution Shift
------------------------------------------------------------------------------------
.. csv-table::
	:header: "Problem", "How to Detect", "How to Fix", "Trade-Offs"
	:align: center 

		Model Degradation, Performance drop (CTR; engagement), Frequent model retraining, Computationally expensive
		Popularity Mismatch, PSI; JSD; embeddings drift, Adaptive reweighting of historical data, Hard to balance long vs. short-term relevance
		Bias Reinforcement, Disparity in exposure metrics, Fairness-aware ranking, May hurt engagement
		Cold-Start for New Trends, Increase in unseen queries, Session-based personalization, Requires fast inference
		Intent Drift in Search, Increase in irrelevant search rankings, Online learning models, Real-time training is costly

Stages
====================================================================================
.. csv-table::
	:header: "Stage", "Goals", "Key Metrics", "Common Techniques"
	:align: center

		Retrieval, Fetch diverse candidates from multiple sources, Recall@K; Coverage; Latency, Multi-tower models; ANN; User embeddings
		Combining & Filtering, Merge candidates; remove duplicates; apply business rules, Diversity; Precision@K; Fairness, Weighted merging; Min-hashing; Rule-based filtering
		Re-Ranking, Optimize order of recommendations for engagement, CTR; NDCG; Exploration Ratio, Neural Rankers; Bandits; DPP for diversity

Patterns
====================================================================================
.. csv-table::
	:header: "Pattern", "Traditional Approach", "LLM Augmentations"
	:align: center

		Query-Item, BM25; TF-IDF; Neural Ranking, LLM-based reranking; Query expansion
		Item-Item, Co-occurrence; Similarity Matching, Semantic matching; Multimodal embeddings
		User-Item, CF; Content-Based; Deep Learning, Personalized generation; Zero-shot preferences
		Session-Based, Sequential Models; Transformers, Few-shot reasoning; Context-aware personalization
		User-User, Graph-Based; Link Prediction, Profile-text analysis; Social graph augmentation

Domains
====================================================================================
#. E-commerce (Amazon, Alibaba)
#. Music (Spotify)
#. Image (Instagram)
#. Video (Netflix, Firestick, YouTube)
#. Voice (Alexa)
#. Short-video (Tiktok)
#. Food (DoorDash, UberEats)
#. Travel (AirBnB)
#. Social (Facebook, Twitter)
#. Search (Google, Bing)
#. Search Advertising (Google, Bing)

Music
------------------------------------------------------------------------------------
.. csv-table::
	:header: "Challenge", "Why Its Important", "Trade-Offs"
	:align: center

		Personalization vs. Serendipity, Users want relevant music but also expect some new discoveries., Too much personalization  Feels repetitive. Too much exploration  Feels random.
		Repetition & Content Fatigue, Users get frustrated if the same songs appear too often., Strict anti-repetition  May exclude user favorites. Loose constraints  Risk of overplaying certain songs.
		Context & Mood Adaptation, Users listen to music differently based on mood; time; activity (workout; relaxation)., Explicit mood tagging is effective but requires manual input. Implicit context detection risks wrong assumptions.
		Balancing Popular & Niche Tracks, Highly popular songs dominate engagement; making it hard for lesser-known songs to gain exposure., Boosting niche tracks improves diversity; but may lower engagement metrics.
		Cold-Start for New Songs & Artists, Newly released songs struggle to get exposure due to lack of engagement signals., Over-boosting new music can lead to reduced user satisfaction.
		Playlist Length & Engagement Optimization, Users may not finish long playlists; leading to low engagement metrics., Shorter playlists increase completion rate; but longer ones improve session duration.

Search
------------------------------------------------------------------------------------
.. note::
	- [fennel.ai] `Feature Engineering for Personalized Search <https://fennel.ai/blog/feature-engineering-for-personalized-search/>`_

Search Advertising
------------------------------------------------------------------------------------
.. csv-table::
	:header: "Issue", "Why It Matters", "Strategic Fixes", "Trade-Offs"
	:align: center

		Relevance vs. Revenue, Showing high-bid but low-relevance ads hurts trust, Hybrid ranking (bid + quality), Too much relevance filtering lowers revenue
		Click Fraud & Ad Spam, Inflated clicks drain budgets, ML-based fraud detection, False positives can hurt advertisers
		Ad Auction Manipulation, AI-driven bid shading exploits system, Second-price auctions, Reduced ad revenue
		Ad Fatigue & Banner Blindness, Users ignore repetitive ads, Adaptive ad rotation, Frequent ad refreshing increases costs
		Query Intent Mismatch, Poor ad matching frustrates users, BERT-based intent detection, Over-restricting ads lowers monetization
		Landing Page Experience, High bounce rate = low conversion, Quality Score rules, Strict rules limit advertiser flexibility
		Multi-Touch Attribution, Last-click attribution undervalues early ad exposures, Shapley-based attribution, More complexity; slower optimization
		Ad Bias & Fairness, Favoring large advertisers hurts competition, Fairness-aware bidding, Less revenue from high bidders

************************************************************************************
Resources
************************************************************************************
Metrics & QA
====================================================================================
.. important::

	* [evidentlyai.com] `10 metrics to evaluate recommender and ranking systems <https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems>`_
	* [docs.evidentlyai.com] `Ranking metrics <https://docs.evidentlyai.com/reference/all-metrics/ranking-metrics>`_
	* [arize.com] `A Quick Survey of Drift Metrics <https://arize.com/blog-course/drift/>`_
	* [github.com] `50 Fundamental Recommendation Systems Interview Questions <https://github.com/Devinterview-io/recommendation-systems-interview-questions>`_
	* [devinterview.io] `50 Recommendation Systems interview questions <https://devinterview.io/questions/machine-learning-and-data-science/recommendation-systems-interview-questions/>`_

Videos
====================================================================================
- [youtube.com] `Stanford CS224W Machine Learning w/ Graphs I 2023 I GNNs for Recommender Systems <https://www.youtube.com/watch?v=OV2VUApLUio>`_
.. note::
	- Mapped as an edge prediction problem in a bipartite graph
	- Ranking

		- Metric Recall@k (non differentiable)
		- Other metrics HR@k, nDCG
		- Differentiable Discriminative loss - binary loss (similar to cross entropy), Bayesian prediction loss (BPR)
		- Issue with binary, BPR solves the ranking problem better
		- Trick to choose neg samples
		- Not suitable for ANN
	- Collaborative filtering

		- DNN to capture user item similarity with cosine or InfoNCE loss
		- ANN friendly
		- Doesn't consider longer than 1 hop in the bipartite graph
	- GCN

		- Smoothens the embeddings by GCN layer interactions using undirected edges to enforce similar user and similar item signals
		- Neural GCN or LightGCN
		- Application similar image recommendation in Pinterest
		- Issue doesn't have contextual awareness or session/temporal awareness

Course, Books & Papers
====================================================================================
Key Papers
------------------------------------------------------------------------------------
	- BOF = Bag of features
	- NG = N-Gram
	- CM = Causal Models (autoregressive)

.. csv-table::
	:header: "Tag", "Title"
	:align: center

		QU,`Better search through query understanding <https://queryunderstanding.com/>`_
		IR;QU,`Using Query Contexts in Information Retrieval <http://www-rali.iro.umontreal.ca/rali/sites/default/files/publis/10.1.1.409.2630.pdf>`_
		IR;Course;Stanford,`CS 276 / LING 286 Information Retrieval and Web Search <https://web.stanford.edu/class/cs276/>`_
		IR;Book,`Introduction to Information Retrieval <https://nlp.stanford.edu/IR-book/information-retrieval-book.html>`_
		Retrieval;Survey,`A Comprehensive Survey on Retrieval Methods in Recommender Systems <https://arxiv.org/pdf/2407.21022>`_
		DL;RS;Survey,`Deep Learning based Recommender System A Survey and New Perspectives <https://arxiv.org/pdf/1707.07435>`_
		Retrival;RS,`Simple but Efficient A Multi-Scenario Nearline Retrieval Framework for Recommendation on Taobao <https://arxiv.org/pdf/2408.00247v1>`_
		Retrival;Ranking;Embed+MLP,`Neural Collaborative Filtering <https://arxiv.org/abs/1708.05031>`_
		Retrival;Two Tower;BOF,`StarSpace Embed All The Things! <https://arxiv.org/abs/1709.03856>`_
		Retrival;Ranking;Two Tower;NG+BOF,`Embedding-based Retrieval in Facebook Search <https://arxiv.org/abs/2006.11632>`_
		Ranking;WDN,`Wide & Deep Learning for Recommender Systems <https://arxiv.org/abs/1606.07792>`_
		Ranking;DCN,`DCN V2 Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems <https://arxiv.org/abs/2008.13535>`_
		Ranking;L2R,`DeepRank: Learning to rank with neural networks for recommendation <http://zhouxiuze.com/pub/DeepRank.pdf>`_
		GCN,`Graph Convolutional Neural Networks for Web-Scale Recommender Systems <https://arxiv.org/abs/1806.01973>`_
		GCN,`LightGCN - Simplifying and Powering Graph Convolution Network for Recommendation <https://arxiv.org/abs/2002.02126>`_
		CM;Session,`Transformers4Rec Bridging the Gap between NLP and Sequential / Session-Based Recommendation <https://scontent.fblr25-1.fna.fbcdn.net/v/t39.8562-6/243129449_615285476133189_8760410510155369283_n.pdf?_nc_cat=104&ccb=1-7&_nc_sid=b8d81d&_nc_ohc=WDJcULkgkY8Q7kNvgHspPmM&_nc_zt=14&_nc_ht=scontent.fblr25-1.fna&_nc_gid=A_fmEzCPOHil7q9dPSpYsHS&oh=00_AYDCkVOnyZufYEGHEQORBbfI-blNODNIrePL4TaB8p_82A&oe=67A8FEDE>`_			
		Diversity;DPP,`Improving the Diversity of Top-N Recommendation via Determinantal Point Process <https://arxiv.org/abs/1709.05135v1>`_
		Diversity;DPP,`Practical Diversified Recommendations on YouTube with Determinantal Point Processes <https://jgillenw.com/cikm2018.pdf>`_
		Diversity;DPP,`Personalized Re-ranking for Improving Diversity in Live Recommender Systems <https://arxiv.org/abs/2004.06390>`_
		Diversity;DPP,`Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity <https://proceedings.neurips.cc/paper_files/paper/2018/file/dbbf603ff0e99629dda5d75b6f75f966-Paper.pdf>`_
		Diversity;Multi-Stage,`Representation Online Matters Practical End-to-End Diversification in Search and Recommender Systems <https://arxiv.org/pdf/2305.15534>`_
		Polularity Bias,`Managing Popularity Bias in Recommender Systems with Personalized Re-Ranking <https://cdn.aaai.org/ocs/18199/18199-78818-1-PB.pdf>`_
		Polularity Bias,`User-centered Evaluation of Popularity Bias in Recommender Systems <https://dl.acm.org/doi/fullHtml/10.1145/3450613.3456821>`_
		Polularity Bias,`Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System <https://arxiv.org/pdf/2010.15363>`_
		Fairness,`Fairness in Ranking Part II Learning-to-Rank and Recommender Systems <https://dl.acm.org/doi/pdf/10.1145/3533380>`_
		Fairness,`Fairness Definitions Explained <https://fairware.cs.umass.edu/papers/Verma.pdf>`_
		LLM,`A Review of Modern Recommender Systems Using Generative Models (Gen-RecSys) <https://arxiv.org/abs/2404.00579>`_
		LLM,`Collaborative Large Language Model for Recommender Systems <https://arxiv.org/abs/2311.01343>`_
		LLM,`Recommendation as Instruction Following A Large Language Model Empowered Recommendation Approach <https://arxiv.org/abs/2305.07001>`_

More Papers
------------------------------------------------------------------------------------
.. csv-table::
	:header: "Year", "Title"
	:align: center

		2001,Item-Based Collaborative Filtering Recommendation Algorithms – Sarwar et al.
		2003,Amazon.com Recommendations Item-to-Item Collaborative Filtering – Linden et al.
		2007,Link Prediction Approaches and Applications – Liben-Nowell et al.
		2008,An Introduction to Information Retrieval – Manning et al.
		2009,BM25 and Beyond – Robertson et al.
		2009,Matrix Factorization Techniques for Recommender Systems – Koren et al.
		2010,Who to Follow Recommending People in Social Networks – Twitter Research
		2014,DeepWalk Online Learning of Social Representations – Perozzi et al.
		2015,Learning Deep Representations for Content-Based Recommendation – Wang et al.
		2015,Netflix Recommendations Beyond the 5 Stars – Gomez-Uribe et al.
		2016,Deep Neural Networks for YouTube Recommendations – Covington et al.
		2016,Wide & Deep Learning for Recommender Systems – Cheng et al.
		2016,Session-Based Recommendations with Recurrent Neural Networks – Hidasi et al.
		2017,DeepRank A New Deep Architecture for Relevance Ranking in Information Retrieval – Pang et al.
		2017,Neural Collaborative Filtering – He et al.
		2017,A Guide to Neural Collaborative Filtering – He et al.
		2018,BERT Pre-training of Deep Bidirectional Transformers for Language Understanding – Devlin et al.
		2018,PinSage Graph Convolutional Neural Networks for Web-Scale Recommender Systems – Ying et al.
		2018,Neural Architecture for Session-Based Recommendations – Tang & Wang
		2018,SASRec Self-Attentive Sequential Recommendation – Kang & McAuley
		2018,Graph Convolutional Neural Networks for Web-Scale Recommender Systems – Ying et al.
		2019,Deep Learning Based Recommender System A Survey and New Perspectives – Zhang et al.
		2019,Session-Based Recommendation with Graph Neural Networks – Wu et al.
		2019,Next Item Recommendation with Self-Attention – Sun et al.
		2019,BERT4Rec Sequential Recommendation with Bidirectional Encoder Representations – Sun et al.
		2020,Dense Passage Retrieval for Open-Domain Question Answering – Karpukhin et al.
		2020,ColBERT Efficient and Effective Passage Search via Contextualized Late Interaction Over BERT – Khattab et al.
		2020,T5 for Information Retrieval – Nogueira et al.
		2021,CLIP Learning Transferable Visual Models from Natural Language Supervision – Radford et al.
		2021,Transformers4Rec Bridging the Gap Between NLP and Sequential Recommendation – De Souza et al.
		2021,Graph Neural Networks A Review of Methods and Applications – Wu et al.
		2021,Next-Item Prediction Using Pretrained Language Models – Sun et al.
		2022,Unified Vision-Language Pretraining for E-Commerce Recommendations – Wang et al.
		2022,Contextual Item Recommendation with Pretrained LLMs – Li et al.
		2023,InstructGPT for Information Retrieval – Ouyang et al.
		2023,GPT-4 for Web Search Augmentation – Bender et al.
		2023,CLIP-Recommend Multimodal Learning for E-Commerce Recommendations – Xu et al.
		2023,Semantic-Aware Item Matching with Large Language Models – Chen et al.
		2023,GPT4Rec A Generative Framework for Personalized Recommendation – Wang et al.
		2023,LLM-based Collaborative Filtering Enhancing Recommendations with Large Language Models – Liu et al.
		2023,LLM-Powered Dynamic Personalized Recommendations – Guo et al.
		2023,Real-Time Recommendation with Large Language Models – Zhang et al.
		2023,Graph Neural Networks Meet Large Language Models A Survey – Wu et al.
		2023,LLM-powered Social Graph Completion for Friend Recommendations – Huang et al.
		2023,LLM-Augmented Node Classification in Social Networks – Zhang et al.

************************************************************************************
Implicit Labels
************************************************************************************
Collaborative Filtering (CF)  
====================================================================================
- Relies on user-item interactions to recommend items. 
- Since users rarely provide explicit ratings, implicit signals are inferred from engagement behaviors.  

User Engagement-Based Labels  
------------------------------------------------------------------------------------
.. csv-table::
	:header: "Implicit Label", "Collection Method", "Assumptions & Trade-offs"
	:align: center

		Clicks, Count clicks on an item.,  Simple; scalable.  Clicking  liking (accidental clicks).
		Watch Time / Dwell Time, Measure time spent on videos/articles.,  Captures engagement depth.  Long duration  satisfaction (e.g.; passive watching).
		Purchase / Conversion, Track purchases (e-commerce; rentals; subscriptions).,  Strongest preference signal.  Sparse data (only a few items are purchased).
		Add to Cart / Wishlist, Users mark interest without purchasing.,  Softer preference signal.  Users may abandon carts.
		Scrolling & Hovering, Detect mouse hover time over items.,  Early preference signal.  May be unintentional.
		Search Queries & Item Views, Items viewed after searching for a term.,  Strong relevance signal.  Some users browse randomly.

Social & Community-Based Signals  
------------------------------------------------------------------------------------
.. csv-table::
	:header: "Implicit Label", "Collection Method", "Assumptions & Trade-offs"
	:align: center

		Likes / Upvotes, Count "likes" on posts; videos; or comments.,  Clear positive feedback.  Some users never like items.
		Shares / Retweets, Count how often users share content.,  Strong endorsement.  May share for controversy.
		Follows / Subscriptions, Followed creators or product wishlists.,  Indicates long-term interest.  Users may follow without deep engagement.

Negative Feedback & Implicit Dislikes  
------------------------------------------------------------------------------------
.. csv-table::
	:header: "Implicit Label", "Collection Method", "Assumptions & Trade-offs"
	:align: center

		Skip / Bounce Rate, Detect when a user skips a song/video quickly.,  Identifies disinterest.  May skip for reasons unrelated to content.
		Negative Actions, "Not Interested" clicks; downvotes; blocking content.,  Explicit dislike signal.  Only a subset of users take these actions.

CF Use Case Example:  
- Spotify uses play count, skip rate, and playlist additions to infer user preferences.  
- Netflix monitors watch completion rate, rewatches, and early exits for movie recommendations.  

Content-Based Filtering (CBF)  
====================================================================================
Session-Based & Short-Term Context Labels  
------------------------------------------------------------------------------------
.. csv-table::
	:header: "Implicit Label", "Collection Method", "Assumptions & Trade-offs"
	:align: center

		Recent Search Context, Track evolving search terms.,  Captures short-term needs.  Trends change quickly.
		Location-Based Preferences, Match user location with nearby content.,  Useful for local recommendations.  Privacy-sensitive.
		Time of Day / Activity Patterns, Suggest different items based on morning/evening behavior.,  Improves context relevance.  Needs continuous adaptation.

Self-Supervised Paradigm
------------------------------------------------------------------------------------
TODO

Knowledge Graphs for Hybrid Labeling
====================================================================================
- Uses entities and relationships to enhance recommendations.

************************************************************************************
Stages
************************************************************************************
Retrieval 
====================================================================================
(Fetching an initial candidate pool from multiple sources) 

Task
------------------------------------------------------------------------------------
	- Reduce a large item pool (millions of candidates) to a manageable number (thousands). 
	- Retrieve diverse candidates from multiple sources that might be relevant to the user. 
	- Balance long-term preferences vs. short-term intent. 

Comon Metrics
------------------------------------------------------------------------------------
	- Recall@K – How many relevant items are in the top-K retrieved items? 
	- Coverage – Ensuring diversity by retrieving from multiple pools. 
	- Latency – Efficient retrieval in milliseconds at large scales. 

Common Techniques
------------------------------------------------------------------------------------
.. csv-table::
	:header: "Goal", "Techniques"
	:align: center

		Heterogeneous Candidate Retrieval, Multi-tower models; Hybrid retrieval (Collaborative Filtering + Content-Based)
		Personalization, User embeddings (e.g.; Two-Tower models; Matrix Factorization)
		Exploration & Freshness, Real-time embeddings; Bandit-based exploration
		Scalability & Efficiency, Approximate Nearest Neighbors (ANN); FAISS; HNSW
		Cold-Start Handling, Content-based retrieval (TF-IDF; BERT); Popularity-based heuristics

Example - YouTube Recommendation 
------------------------------------------------------------------------------------
	- Candidate pools Watched videos, partially watched videos, topic-based videos, demographically popular videos, newly uploaded videos, videos from followed channels. 
	- Techniques used Two-Tower model for retrieval, Approximate Nearest Neighbors (ANN) for fast lookup. 

Combining & Filtering 
====================================================================================
(Merging retrieved candidates from different sources and removing low-quality items) 

Task
------------------------------------------------------------------------------------
	- Merge multiple retrieved pools and assign confidence scores to each source. 
	- Filter out irrelevant, duplicate, or low-quality candidates. 
	- Apply business rules (e.g., compliance filtering, removing expired content). 

Comon Metrics
------------------------------------------------------------------------------------
	- Diversity – Ensuring different content types are represented. 
	- Precision@K – How many retrieved items are actually relevant? 
	- Fairness & Representation – Avoiding over-exposure of popular items. 
	- Latency – Keeping the filtering process efficient. 

Common Techniques
------------------------------------------------------------------------------------
.. csv-table::
	:header: "Goal", "Techniques"
	:align: center

		Merging Multiple Candidate Pools, Weighted aggregation based on confidence scores
		Duplicate Removal, Min-hashing; Jaccard similarity; clustering-based deduplication
		Quality Filtering, Heuristic filters; Rule-based filters; Adversarial detection
		Business Constraints, Compliance rules (e.g.; sensitive content removal); Content freshness checks
		Balancing Diversity, Re-weighting based on underrepresented categories
		Scaling Up, Streaming pipelines (Kafka; Flink); Pre-filtering with Bloom Filters

Example - Newsfeed Recommendation 
------------------------------------------------------------------------------------
	- Candidate sources Text posts, image posts, video posts. 
	- Filtering techniques Removing duplicate posts, blocking low-quality content, filtering based on engagement thresholds. 

Re-Ranking 
====================================================================================
Task
------------------------------------------------------------------------------------
	- Optimize the order of candidates to maximize engagement. 
	- Balance personalization with exploration (ensuring new content gets surfaced). 
	- Ensure fairness and representation (avoid showing only highly popular items). 

Metrics
------------------------------------------------------------------------------------
	- [Offline] AUC (ROC-AUC, PR-AUC) – Measures prediction accuracy if modeled as a binary classification problem.
	- [Offline] NDCG@k, MRR@k, HR@k – Measures ranking quality.
	- [Online] CTR (Click-Through Rate) – Measures immediate engagement.
	- [Online] Long-Term Engagement – Holdout -> Measures retention and repeat interactions.
	- [?] Exploration Ratio – Tracks new content shown to users.

Techniques
------------------------------------------------------------------------------------
.. csv-table::
	:header: "Goal", "Techniques"
	:align: center

		Fast Re-Ranking, Tree-based models (GBDT); LightGBM; XGBoost
		Personalized Ranking, Embed + MLP Models (e.g.; DeepFM; Wide & Deep; Transformer-based rankers)
		Diversity Promotion, Re-ranking by category (e.g.; Round Robin); Determinantal Point Processes (DPP)
		Explore-Exploit Balance, Multi-Armed Bandits (Thompson Sampling; UCB); Randomized Ranking
		Handling Highly Popular Items, Popularity dampening; Re-ranking with popularity decay
		Fairness & Representation, Re-weighting models; Exposure-aware ranking		

Resources
------------------------------------------------------------------------------------
Ranking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Features
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	- User profile (captures long term user's preferences)
	- Item profile (captures item metadata and content understanding)
	- Contextual features (e.g, device, geolocation, temporal)
	- Interaction features

`CTR Prediction Papers <https://paperswithcode.com/task/click-through-rate-prediction>`_
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
.. csv-table::
	:header: "Technique", "Resource"
	:align: center

		LR, `Distributed training of Large-scale Logistic models <https://proceedings.mlr.press/v28/gopal13.pdf>`_
		Survey, `Click-Through Rate Prediction in Online Advertising: A Literature Review <https://arxiv.org/abs/2202.10462>`_
		Embed + MLP, `Deep Neural Networks for YouTube Recommendations <https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf>`_
		Embed + MLP, `Real-time Personalization using Embeddings for Search Ranking at Airbnb <https://dl.acm.org/doi/pdf/10.1145/3219819.3219885>`_
		Wide & Deep, `Wide & Deep Learning for Recommender Systems <https://arxiv.org/abs/1606.07792>`_
		DeepFM, `DeepFM: A Factorization-Machine based Neural Network for CTR Prediction <https://arxiv.org/abs/1703.04247>`_
		xDeepFM, `xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems <https://arxiv.org/abs/1803.05170>`_
		DCN, `Deep & Cross Network for Ad Click Predictions <https://arxiv.org/abs/1708.05123>`_
		DCNv2, `DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems <https://arxiv.org/abs/2008.13535>`_
		DIN, `Deep Interest Network for Click-Through Rate Prediction <https://arxiv.org/abs/1706.06978>`_
		BST, `Behavior Sequence Transformer for E-commerce Recommendation in Alibaba <https://arxiv.org/abs/1905.06874>`_

************************************************************************************
Patterns
************************************************************************************
Query-Item Recommendation 
====================================================================================
- Search systems
- text-to-item search
- image-to-item search
- query expansion techniques

Key Concept 
------------------------------------------------------------------------------------
- Common Approaches

	- Lexical Matching (TF-IDF, BM25, keyword-based retrieval) 
	- Semantic Matching (Word embeddings, Transformer models like BERT, CLIP for vision-text matching) 
	- Hybrid Search (Combining lexical and semantic search, e.g., BM25 + embeddings) 
	- Learning-to-Rank (LTR) models optimizing ranking performance based on user interactions) 
	- Multimodal Search (Image-to-text retrieval, video search, voice search, etc.) 
- LLM Applications

	- LLMs enhance ranking via reranking models (ColBERT, T5-based retrieval). 
	- Can be used for query expansion, understanding user intent, and handling ambiguous queries. 
	- Example use case Google Search, AI-driven Q&A search (Perplexity AI). 

Learning Objectives
------------------------------------------------------------------------------------
#. Supervised Learning 

	- Label Binary (clicked vs. not clicked) or relevance score (explicit ratings, dwell time). 
	- Data sources Search logs, query-click data, user feedback (thumbs up/down). 
	- Challenges Noisy labels (e.g., clicks may not always indicate relevance). 
#. Semi-Supervised Learning 

	- Use query expansion techniques (e.g., weak supervision from similar queries). 
	- Leverage pseudo-labeling (e.g., use a weaker ranker to generate labels). 
#. Self-Supervised Learning 

	- Contrastive learning (e.g., train embeddings by pulling query and relevant items closer). 
	- Masked query prediction (e.g., predicting missing words in search queries). 

Common Features
------------------------------------------------------------------------------------
- Query Features Term frequency, query length, part-of-speech tagging. 
- Item Features Title, description, category, metadata, embeddings. 
- Interaction Features Click history, query-to-item dwell time, CTR. 
- Contextual Features Time of query, device type, user history. 
- Embedding-Based Features Pretrained word embeddings (Word2Vec, FastText, BERT embeddings). 

Resources
------------------------------------------------------------------------------------
#. Traditional Information Retrieval 

	- "An Introduction to Information Retrieval" – Manning et al. (2008) 
	- "BM25 and Beyond" – Robertson et al. (2009) 
#. Neural Ranking Models 

	- "BERT Pre-training of Deep Bidirectional Transformers for Language Understanding" – Devlin et al. (2018) 
	- "Dense Passage Retrieval for Open-Domain Question Answering" – Karpukhin et al. (2020) 
#. Multimodal & Deep Learning-Based Search 

	- "CLIP Learning Transferable Visual Models from Natural Language Supervision" – Radford et al. (2021) 
	- "DeepRank A New Deep Architecture for Relevance Ranking in Information Retrieval" – Pang et al. (2017) 
#. LLM-Based Search Ranking 

	- "ColBERT Efficient and Effective Passage Search via Contextualized Late Interaction Over BERT" – Khattab et al. (2020) 
	- "T5 for Information Retrieval" – Nogueira et al. (2020) 
#. LLM-Augmented Search 

	- "InstructGPT for Information Retrieval" – Ouyang et al. (2023) 
	- "GPT-4 for Web Search Augmentation" – Bender et al. (2023) 

Item-Item Recommendation 
====================================================================================
- Similar Products
- Related Videos
- "Customers Who Bought This Also Bought"

Key Concept 
------------------------------------------------------------------------------------
- Item-item recommendation focuses on suggesting similar items based on user interactions. This is widely used in e-commerce, streaming platforms, and content discovery systems. 

	- Typically modeled as an item simi-larity problem. 
	- Unlike user-item recommendation, the goal is to find related items rather than predicting a user’s preferences. 
- Common Approaches

	- Item-Based Collaborative Filtering (Similarity between item interaction histories) 
	- Content-Based Filtering (Similarity using item attributes like text, image, category) 
	- Graph-Based Approaches (Item-item similarity using co-purchase graphs) 
	- Deep Learning Methods (Representation learning, embeddings) 
	- Hybrid Methods (Combining multiple approaches) 
- LLM Applications

	- LLMs improve semantic similarity scoring, identifying nuanced item relationships.
	- Multimodal LLMs (e.g., CLIP) combine text, images, and metadata to enhance recommendations.
	- Example use case E-commerce (Amazon's “similar items”), content platforms (Netflix’s related videos).

Learning Objectives
------------------------------------------------------------------------------------
#. Supervised Learning 

	- Label Binary (1 = two items are similar, 0 = not similar). 
	- Data sources Co-purchase data, co-click data, content similarity. 
	- Challenges Defining meaningful similarity when explicit labels don’t exist. 
#. Semi-Supervised Learning 

	- Clustering similar items based on embeddings or co-occurrence. 
	- Weak supervision from user-generated tags, reviews. 
#. Self-Supervised Learning 

	- Contrastive learning (e.g., learning embeddings by pushing dissimilar items apart). 
	- Masked item prediction (e.g., predicting missing related items in a session). 

Common Features
------------------------------------------------------------------------------------
- Item Features Category, brand, price, textual description, images. 
- Interaction Features Co-purchase counts, view sequences, co-engagement. 
- Graph Features Item co-occurrence in user sessions, citation networks. 
- Embedding-Based Features Learned latent item representations. 
- Contextual Features Time decay (trending vs. evergreen items).  

Resources
------------------------------------------------------------------------------------
#. Collaborative Filtering-Based Approaches 

	- "Item-Based Collaborative Filtering Recommendation Algorithms" – Sarwar et al. (2001) 
	- "Matrix Factorization Techniques for Recommender Systems" – Koren et al. (2009) 
#. Content-Based Approaches 

	- "Learning Deep Representations for Content-Based Recommendation" – Wang et al. (2015) 
	- "Deep Learning Based Recommender System A Survey and New Perspectives" – Zhang et al. (2019) 
#. Graph-Based & Hybrid Approaches 

	- "Amazon.com Recommendations Item-to-Item Collaborative Filtering" – Linden et al. (2003) 
	- "PinSage Graph Convolutional Neural Networks for Web-Scale Recommender Systems" – Ying et al. (2018) 
#. Multimodal LLMs for Recommendation 

	- "CLIP-Recommend Multimodal Learning for E-Commerce Recommendations" – Xu et al. (2023) 
	- "Unified Vision-Language Pretraining for E-Commerce Recommendations" – Wang et al. (2022) 
#. Semantic Similarity Using LLMs 

	- "Semantic-Aware Item Matching with Large Language Models" – Chen et al. (2023) 
	- "Contextual Item Recommendation with Pretrained LLMs" – Li et al. (2022) 

User-Item Recommendation 
====================================================================================
- Homepage recommendations
- product recommendations
- videos you might like, etc

Key Concept 
------------------------------------------------------------------------------------
- User-item recommendation focuses on predicting a user's preference for an item based on historical interactions. This can be framed as 

	- Explicit feedback (e.g., ratings, thumbs up/down) 
	- Implicit feedback (e.g., clicks, watch time, purchases) 
- Common Approaches

	- Collaborative Filtering (CF) (Matrix Factorization, Neural CF) 
	- Content-Based Filtering (Feature-based models) 
	- Hybrid Models (Combining CF and content-based methods) 
	- Deep Learning Approaches (Neural networks, Transformers) 
- LLM Applications

	- LLMs enhance this by learning richer user and item embeddings, capturing nuanced interactions. 
	- LLMs can generate user preferences dynamically via zero-shot/few-shot learning, improving personalization. 
	- Example use case Personalized product descriptions, interactive recommendation assistants. 

Learning Objectives
------------------------------------------------------------------------------------
#. Supervised Learning 

	- Label binary (clicked/not clicked, purchased/not purchased) or continuous (watch time, rating). 
	- Data sources user interactions, purchase logs, watch history. 
	- Challenges Class imbalance (many more non-clicked items than clicked ones). 
#. Semi-Supervised Learning 

	- Use self-training (pseudo-labeling) to expand labeled data. 
	- Graph-based methods to propagate labels across similar users/items. 
#. Self-Supervised Learning 

	- Contrastive learning (e.g., SimCLR, BERT-style masked item prediction). 
	- Learning representations via session-based modeling (e.g., predicting the next item a user interacts with). 

Common Features
------------------------------------------------------------------------------------
- User Features Past interactions, demographics, engagement signals. 
- Item Features Category, text/image embeddings, historical engagement. 
- Cross Features User-item interactions (e.g., user’s affinity to a category). 
- Contextual Features Time of day, device, location. 
- Embedding-based Features Learned latent factors from models like Word2Vec for items/users. 

Resources
------------------------------------------------------------------------------------
#. Collaborative Filtering 

	- "Matrix Factorization Techniques for Recommender Systems" – Koren et al. (2009) 
	- "Neural Collaborative Filtering" – He et al. (2017) 
#. Deep Learning Approaches 

	- "Deep Neural Networks for YouTube Recommendations" – Covington et al. (2016) 
	- "Wide & Deep Learning for Recommender Systems" – Cheng et al. (2016) 
#. Hybrid and Production Systems 

	- "Netflix Recommendations Beyond the 5 Stars" – Gomez-Uribe et al. (2015) 
#. Transformer-Based RecSys 

	- "BERT4Rec Sequential Recommendation with Bidirectional Encoder Representations" – Sun et al. (2019) 
	- "SASRec Self-Attentive Sequential Recommendation" – Kang & McAuley (2018) 
#. LLM-powered Recommendation 

	- "GPT4Rec A Generative Framework for Personalized Recommendation" – Wang et al. (2023) 
	- "LLM-based Collaborative Filtering Enhancing Recommendations with Large Language Models" – Liu et al. (2023) 

Session-Based Recommendation 
====================================================================================
- Personalized recommendations based on recent user actions
- short-term intent modeling
- sequential recommendations

Key Concept 
------------------------------------------------------------------------------------
- Session-based recommendation focuses on predicting the next relevant item for a user based on their recent interactions, rather than long-term historical data. This is useful when 

	- Users don’t have extensive histories (e.g., guest users). 
	- Preferences shift dynamically (e.g., browsing sessions in e-commerce). 
	- Recent behavior is more indicative of intent than long-term history. 
- Common Approaches

	- Rule-Based Methods (Most popular, trending, or recently viewed items) 
	- Markov Chains & Sequential Models (Predicting next item based on state transitions) 
	- Recurrent Neural Networks (RNNs, GRUs, LSTMs) (Capturing sequential dependencies) 
	- Graph-Based Approaches (Session-based Graph Neural Networks) 
	- Transformer-Based Models (Attention-based architectures for session modeling) 
- LLM Applications

	- Traditional methods use sequential models (RNNs, GRUs, Transformers) to predict next-item interactions. 
	- LLMs enhance session modeling by leveraging sequential reasoning and contextual awareness. 
	- Few-shot prompting allows LLMs to infer session preferences without explicit training. 
	- Example use case Dynamic content feeds (TikTok), real-time recommendations (Spotify session playlists). 

Learning Objectives
------------------------------------------------------------------------------------
#. Supervised Learning 

	- Label Next item in sequence (e.g., clicked/purchased item). 
	- Data sources User sessions, browsing logs, cart abandonment data. 
	- Challenges Short sessions make training harder; sparse interaction data. 
#. Semi-Supervised Learning 

	- Use self-supervised tasks like predicting masked interactions. 
	- Graph-based node propagation to learn session similarities. 
#. Self-Supervised Learning 

	- Contrastive learning (e.g., predict next item from different user sessions). 
	- Next-click prediction using masked sequence modeling (BERT-style). 

Common Features
------------------------------------------------------------------------------------
- Session Features Time spent, number of items viewed, recency of last interaction. 
- Item Features Product category, textual embeddings, popularity trends. 
- Sequence Features Click sequences, time gaps between interactions. 
- Contextual Features Device type, time of day, geographical location. 
- Embedding-Based Features Pretrained session embeddings (e.g., Word2Vec-like for items). 

Resources
------------------------------------------------------------------------------------
#. Traditional Approaches & Sequential Models 

	- "Session-Based Recommendations with Recurrent Neural Networks" – Hidasi et al. (2016) 
	- "Neural Architecture for Session-Based Recommendations" – Tang & Wang (2018) 
#. Graph-Based Methods 

	- "Session-Based Recommendation with Graph Neural Networks" – Wu et al. (2019) 
	- "Next Item Recommendation with Self-Attention" – Sun et al. (2019) 
#. Transformer-Based Methods 

	- "SASRec Self-Attentive Sequential Recommendation" – Kang & McAuley (2018) 
	- "BERT4Rec Sequential Recommendation with Bidirectional Encoder Representations" – Sun et al. (2019) 
#. LLM-Driven Dynamic Recommendation 

	- "LLM-Powered Dynamic Personalized Recommendations" – Guo et al. (2023) 
	- "Next-Item Prediction Using Pretrained Language Models" – Sun et al. (2021) 
	- "Real-Time Recommendation with Large Language Models" – Zhang et al. (2023) 

User-User Recommendation 
====================================================================================
- People You May Know
- Friend Suggestions
- Follower Recommendations

Key Concept 
------------------------------------------------------------------------------------
- User-user recommendation focuses on predicting connections between users based on their behavior, interests, or existing social networks.

	#. Typically modeled as a link prediction problem in graphs. 
	#. Used for social networks, professional connections, or matchmaking systems. 
- Common Approaches

	#. Collaborative Filtering (User-Based CF) 
	#. Graph-Based Approaches (Graph Neural Networks, PageRank, Node2Vec, etc.) 
	#. Feature-Based Matching (Demographic and behavior similarity) 
	#. Hybrid Approaches (Graph + CF + Deep Learning) 
- LLM Applications

	- Typically modeled as a graph-based link prediction problem, where users are nodes. 
	- LLMs can enhance user similarity computations by processing richer profile texts (e.g., bios, chat history). 
	- Social connections can be inferred by analyzing natural language data, rather than relying solely on structural graph features. 
	- Example use case Professional networking (LinkedIn), AI-assisted friend suggestions. 

Learning Objectives
------------------------------------------------------------------------------------
#. Supervised Learning 

	- Label Binary (1 = connection exists, 0 = no connection). 
	- Data sources Friendship graphs, follow/unfollow actions, mutual interests. 
	- Challenges Highly imbalanced data (most user pairs are not connected). 

#. Semi-Supervised Learning 

	- Graph-based label propagation (e.g., predicting missing edges in a user graph). 
	- Use unlabeled users with weak supervision from social structures. 

#. Self-Supervised Learning 

	- Contrastive learning (e.g., learning embeddings where connected users are closer in vector space). 
	- Masked edge prediction (e.g., hide some connections and train the model to reconstruct them). 

Common Features
------------------------------------------------------------------------------------
- User Features Profile attributes (age, location, industry, interests). 
- Graph Features Common neighbors, Jaccard similarity, Adamic-Adar score. 
- Interaction Features Message frequency, engagement level. 
- Embedding-Based Features Node2Vec or GNN-based embeddings. 
- Contextual Features Activity time, shared communities.

Resources
------------------------------------------------------------------------------------
#. Collaborative Filtering-Based Approaches 

	- "Item-Based Collaborative Filtering Recommendation Algorithms" – Sarwar et al. (2001) 
	- "A Guide to Neural Collaborative Filtering" – He et al. (2017) 
#. Graph-Based Approaches 

	- "DeepWalk Online Learning of Social Representations" – Perozzi et al. (2014) 
	- "Graph Convolutional Neural Networks for Web-Scale Recommender Systems" – Ying et al. (2018) 
	- "Graph Neural Networks A Review of Methods and Applications" – Wu et al. (2021) 
#. Hybrid and Large-Scale User-User Recommendation 

	- "Link Prediction Approaches and Applications" – Liben-Nowell et al. (2007) 
	- "Who to Follow Recommending People in Social Networks" – Twitter Research (2010) 
#. Graph-Based LLMs 

	- "Graph Neural Networks Meet Large Language Models A Survey" – Wu et al. (2023) 
	- "LLM-powered Social Graph Completion for Friend Recommendations" – Huang et al. (2023) 
#. Hybrid Graph and LLMs 

	- "LLM-Augmented Node Classification in Social Networks" – Zhang et al. (2023) 
	- "Graph Convolutional Neural Networks for Web-Scale Recommender Systems" – Ying et al. (2018)  

************************************************************************************
Issues in Search & Recommendation Systems
************************************************************************************
General Issues
====================================================================================
Cold-Start Problem (Users & Items) 
------------------------------------------------------------------------------------
- Why It Matters 

	- New users No interaction history makes personalization difficult. 
	- New items Struggle to get exposure due to lack of engagement signals. 

- Strategic Solutions & Trade-Offs 

	- Content-Based Methods (Text embeddings, Image/Video features) → Good for new items, but lacks user personalization. 
	- Demographic-Based Recommendations (Cluster similar users) → Generalizes well but risks oversimplification. 
	- Randomized Exploration (Show new items randomly) → Increases fairness but can reduce CTR. 

- Domain-Specific Notes 

	- E-commerce (Amazon, Etsy) → Cold-start for new sellers & niche products. 
	- Video Streaming (Netflix, YouTube) → Cold-start for newly released content. 

Popularity Bias & Feedback Loops
------------------------------------------------------------------------------------
- Why It Matters 

	- Over-recommending already popular items creates a "rich-get-richer" effect affecting fairness, novelty.
	- Reinforces biases in user engagement, making it harder to surface niche or novel content.

- Common Approaches:
	- Changing objective

		- ReGularization (RG)

			- [depaul.edu] `Controlling Popularity Bias in Learning to Rank Recommendation <https://scds.cdm.depaul.edu/wp-content/uploads/2017/05/SOCRS_2017_paper_5.pdf>`_
			- Controls the ratio of popular and less popular items via a regularizer added to the objective function
			- Penalizes lists that contain only one group of items and hence attempting to reduce the concentration on popular items
		- Discrepancy Minimization (DM)

			- [cmu.edu] `Post Processing Recommender Systems for Diversity <https://www.contrib.andrew.cmu.edu/~ravi/kdd17.pdf>`_
			- Optimizes for aggregate diversity
			- Define a target distribution of item exposure as a constraint for the objective function
			- Goal is therefore to minimize the discrepancy of the recommendation frequency for each item and the target distribution
		- FA*IR (FS)

			- [arxiv.org] `FA*IR A Fair Top-k Ranking Algorithm <https://arxiv.org/abs/1706.06368>`_
			- Creates queues of protected (long-tail) and unprotected (head) items so that protected items get more exposure
		- Personalized Long-tail Promotion (XQ)

			- [arxiv.org] `Managing Popularity Bias in Recommender Systems with Personalized Re-ranking <https://arxiv.org/abs/1901.07555>`_
			- Query result diversification
			 -The objective for a final recommendation list is a balanced ratio of popular and less popular (long-tail) items.
		- Calibrated Popularity (CP)

			- [arxiv.org] `User-centered Evaluation of Popularity Bias in Recommender Systems - Abdollahpouri et. al <https://arxiv.org/pdf/2103.06364>`_
			- Takes user's affinity towards popular, diverse and niche contents into account
	- Randomisation

		- Contextual Bandits
	- Position debiasing
- Domain-Specific Notes:

	- Social Media (TikTok, Twitter, Facebook) Celebrity overexposure (e.g., verified users dominating feeds). 
	- News Aggregators (Google News, Apple News) Same sources getting recommended (e.g., mainstream news over independent journalism). 

Diversity vs. Personalization Trade-Off 
------------------------------------------------------------------------------------
- Resources:

	- [engineering.fb.com] `On the value of diversified recommendations <https://engineering.fb.com/2020/12/17/ml-applications/diversified-recommendations/>`_
- Why It Matters:

	- Highly personalized feeds reinforce user preferences, limiting exposure to new content.
	- Leads to boredom of users in long-term which might reduce retention rate.
	- Users may get stuck in content silos (e.g., political polarization, filter bubbles).

- Understanding the issue:
	
	- Theoretical framework
	
		- Personalization
			- Polya process
			- self reinforcement
			- pros short term gains
			- cons leads to boredom and retention
		- Balancing
			- balancing process
			- Negative reinforcement
			- Pros doesn't lead to boredom
			- Cons affects short term gains
	- Complexities in real world personal preferences

		- Multidimensional (dark comedy = dark thriller + general comedy)
		- Soft (30% affinity towards comedy, 90% affinity towards sports)
		- Contextual (mood, time of day, current trends)
		- Dynamic (evolves over time)

- Heuristics on diversifying recommendation:

	- Author level diversity -> strafification -> pick candidates from different authors
	- Media type diversity -> applicable for multimedia platforms -> intermix modality
	- Semantic diversity -> content understanding system -> classify user's affinity to topics -> sample across topics
	- Explore similar semantic nodes -> knowledge tree/graph

		- Explore parents, siblings, children of topics
		- Explore long tail for niche topics
		- Explore items that covers multiple topics
	- Maintain separate pool for short-term and long-term preferences
	- Utilize explore-exploit framework -> eps-greedy, ucb, thompson sampling
	- Prioritize behavioural metrics as much as accuracy metrics
	- Priotitize explicit negative feedbacks from users

- Strategic Solutions & Trade-Offs 

	- Diversity-Promoting Re-Ranking (DPP, Exploration Buffers) -> Reduces filter bubbles but may decrease engagement. 
	- Diversity-Constrained Search (Re-weighting ranking models) -> Promotes varied content but risks reducing precision. 
	- Hybrid User-Item Graphs (Graph Neural Networks for diversification) -> Balances exploration but requires expensive training. 

- Domain-Specific Notes 

	- Social Media (Facebook, Twitter, YouTube) -> Political echo chambers & misinformation bubbles. 
	- E-commerce (Amazon, Etsy, Zalando) -> Users seeing only one type of product repeatedly.

Short-Term Engagement vs. Long-Term User Retention 
------------------------------------------------------------------------------------
- Why It Matters 

	- Systems often optimize for immediate engagement (CTR, watch time, purchases), which can lead to addictive behaviors or content fatigue.
	- Over-exploitation of "sticky content" (clickbait, sensationalism, autoplay loops) may reduce long-term satisfaction.

- Strategic Solutions & Trade-Offs:

	- Multi-Objective Optimization (CTR + Long-Term Retention) -> Complex to balance but essential for sustainability.
	- Delayed Reward Models (Reinforcement Learning) -> Great for long-term user retention but slow learning process.
	- Personalization Decay (Balancing Freshness vs. Relevance) -> Introduces diverse content but can feel random to users.

- Domain-Specific Notes:

	- YouTube, TikTok, Instagram -> Prioritizing sensational viral content over educational material.
	- E-Commerce (Amazon, Alibaba) -> Short-term discounts vs. long-term brand loyalty.

Real-Time Personalization & Latency Trade-Offs 
------------------------------------------------------------------------------------
- Why It Matters 

	- Personalized recommendations require real-time feature updates and low-latency inference. 
	- Search relevance depends on immediate context (e.g., location, time of day, trending topics). 

- Strategic Solutions & Trade-Offs 

	- Precomputed User Embeddings (FAISS, HNSW, Vector DBs) → Speeds up search but sacrifices personalization flexibility. 
	- Edge AI for On-Device Personalization → Reduces latency but increases computational costs. 
	- Session-Based Recommendation Models (Transformers for Session-Based Context) → Great for short-term personalization but expensive for large user bases. 

- Domain-Specific Notes 

	- E-Commerce (Amazon, Walmart, Shopee) → Latency constraints for "similar item" recommendations. 
	- Search Engines (Google, Bing, Baidu) → Needing real-time personalization without slowing down results. 

Domain-Specific
====================================================================================
Search
------------------------------------------------------------------------------------ 
- Query Understanding & Intent Disambiguation

	- Users enter ambiguous or vague queries, requiring intent inference. 
	- Example Searching for “apple” – Is it a fruit, a company, or a music service? 
	- Solutions & Trade-Offs 

		- LLM-Powered Query Rewriting (T5, GPT) → Improves relevance but risks over-modifying queries. 
		- Session-Aware Query Expansion → Helps disambiguation but increases computational cost. 

E-Commerce
------------------------------------------------------------------------------------
- Balancing Revenue & User Satisfaction 

	- Revenue-driven recommendations (sponsored ads, promoted products) vs. organic recommendations. 
	- Example Amazon mixing sponsored and personalized search results. 
	- Solutions & Trade-Offs 

		- Hybrid Models (Re-ranking with Fairness Constraints) → Balances organic vs. paid but hard to tune for revenue goals. 
		- Trust-Based Ranking (Reducing deceptive sellers, fake reviews) → Improves satisfaction but may lower short-term sales. 

Video & Music Streaming
------------------------------------------------------------------------------------
- Content-Length Bias in Recommendations 

	- Recommendation models often favor shorter videos (TikTok, YouTube Shorts) over long-form content. 
	- Example YouTube’s watch-time optimization may prioritize clickbaity short videos over educational content. 
	- Solutions & Trade-Offs 

		- Normalized Engagement Metrics (Watch Percentage vs. Watch Time) → Improves long-form content exposure but may reduce video diversity. 
		- Hybrid-Length Recommendations (Mixing Shorts & Full Videos) → Enhances variety but harder to rank effectively.

************************************************************************************
Deep Dives
************************************************************************************
Personalisation
====================================================================================

Diversity
====================================================================================
.. important::
	- Music & video platforms (Spotify, YouTube, TikTok) use DPP and Bandits to introduce diverse content.
	- E-commerce (Amazon, Etsy) balances popularity-based downsampling with weighted re-ranking.
	- Newsfeeds (Google News, Facebook, Twitter) use category-sensitive filtering to prevent echo chambers.

- Goal

	- improving user engagement
	- avoiding filter bubbles
	- preventing over-reliance on popular content.
- Metric

	- TODO

- LLMs for Diversity in Recommendations

	.. note::	
		- YouTube - Uses LLMs for multi-modal retrieval (text, video, audio). 
		- Spotify - Uses LLMs for playlist diversification and exploration-based re-ranking. 
		- Netflix - Uses GPT-like models for diverse genre-based recommendations. 
		- Google Search & News - Uses BERT-based fairness filters for diverse search results. 

- Technique Summary

	.. csv-table::
		:header: "Technique", "Stage", "Pros", "Cons"
		:align: center

			Multi-Pool Retrieval, Retrieval, High diversity; multiple candidate sources, Computationally expensive
			Popularity-Based Downsampling, Retrieval, Prevents over-recommendation of trending items, May reduce engagement
			Minimum-Item Representation Heuristics, Filtering, Ensures fairness across categories, Might reduce personalization
			Category-Sensitive Filtering, Filtering, Adapts to user preferences dynamically, High computation cost
			Determinantal Point Processes (DPP), Re-Ranking, Mathematical diversity control, Computationally expensive
			Re-Ranking with Diversity Constraints, Re-Ranking, Tunable for personalization vs. diversity, Requires careful tuning
			Multi-Armed Bandits, Re-Ranking, Balances personalization and exploration, Hard to tune in real-world scenarios

- LLMs for Diversity at Each Stage 

	.. csv-table::
		:header: "Stage", "LLM Enhancements", "Pros", "Cons"
		:align: center

			Retrieval, Query expansion; Multi-modal retrieval, Increases recall & heterogeneity, Higher latency; Loss of precision
			Filtering & Merging, Semantic deduplication; Bias correction, Prevents redundancy; Fairer recommendations, Computationally expensive
			Re-Ranking, Diversity-aware reranking; Counterfactuals, Balances personalization & exploration, Risk of over-exploration; Expensive inference

Retrieval Stage
------------------------------------------------------------------------------------
.. note::
	Goal Ensuring Diversity in Candidate Selection

Multi-Pool Retrieval (Heterogeneous Candidate Selection)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	- Retrieves candidates from multiple independent sources (e.g., popularity-based pool, collaborative filtering pool, content-based retrieval).
	- Ensures that recommendations are not solely based on one dominant factor (e.g., trending items).

Pros:

	- Increases coverage by considering multiple types of items.
	- Helps balance long-term preferences vs. short-term interest.

Cons:

	- If not weighted properly, can introduce irrelevant or low-quality recommendations.
	- Computationally expensive when handling large numbers of pools.

Example:

	- YouTube retrieves candidates from watched videos, partially watched videos, new uploads, and popular in demographic to balance diversity.

Popularity-Based Downsampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	- Reduces the dominance of highly popular items in the candidate pool.
	- Ensures niche items have a fair chance of being retrieved.

Pros:

	- Prevents "rich-get-richer" feedback loops.
	- Encourages long-tail item discovery.

Cons:

	- Might hurt immediate engagement metrics (CTR, Watch Time).
	- New users may still prefer popular items over niche ones.

Example:

	- Spotifys Discover Weekly uses a mix of popular and long-tail recommendations to balance engagement and discovery.

LLMs for Diverse Candidate Selection 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	#. Query Expansion for Better Recall 

		- LLMs generate query variations to retrieve diverse candidates beyond exact keyword matching. 
		- Example Instead of just retrieving laptops, LLMs expand queries to include notebooks, MacBooks, ultrabooks. 
		- Technique Use T5/BERT-based semantic expansion to increase retrieval diversity. 
	
	#. Multi-Modal Understanding for Heterogeneous Retrieval 

		- LLMs bridge different modalities (text, image, video) to retrieve richer candidate pools. 
		- Example In YouTube Recommendations, an LLM can link a users watched TED Talk to blog articles on the same topic. 
		- Technique Use CLIP (for text-image-video embeddings) to retrieve across modalities. 

	#. User Preference Understanding for Contextual Retrieval 

		- Instead of static retrieval models, LLMs generate dynamic search queries based on user conversation history. 
		- Example A user searching for travel backpacks may also receive recommendations for hiking gear if LLMs infer the intent. 
		- Technique Use GPT-like models to rewrite user queries dynamically based on session context. 

Pros 

	- Improves Recall - LLMs retrieve more diverse content that traditional CF models miss. 
	- Better Cold-Start Handling - Generates synthetic preferences for new users. 

Cons 

	- High Latency - Generating queries dynamically can be slower than precomputed embeddings. 
	- Loss of Precision - More diverse candidates mean a higher risk of retrieving irrelevant results. 

Filtering & Merging Stage
------------------------------------------------------------------------------------
.. note::
	Goal Balancing Diversity Before Re-Ranking

Minimum-Item Representation Heuristics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	- Ensures that each category, genre, or provider has a minimum number of candidates before merging.
	- Helps prevent over-representation of any single category.

Pros:

	- Easy to implement with rule-based heuristics.
	- Ensures fairness in content exposure.

Cons:

	- Can sacrifice relevance by forcing underrepresented items.
	- Hard to scale for fine-grained personalization.

Example:

	- News Feeds (Facebook, Twitter, Google News) ensure a minimum number of international vs. local news, avoiding content silos.

Category-Sensitive Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	- Computes category entropy to measure diversity across different categories.
	- If a users recommendations lack category diversity, it enforces rebalancing by boosting underrepresented categories.

Pros:

	- Dynamically adapts to different users.
	- Can be optimized for long-term user retention.

Cons:

	- Requires real-time category tracking, which can be computationally expensive.
	- Poor tuning may result in irrelevant recommendations.

Example:

	- Netflix ensures that recommendations contain a mix of different genres rather than overloading one.

LLMs for Diversity-Aware Candidate Selection 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	#. Semantic Deduplication & Cluster Merging 

		- LLMs identify semantically similar items (even if they differ in wording) to prevent redundancy. 
		- Example In news recommendations, LLMs group articles covering the same event to avoid repetition. 
		- Technique Use sentence embeddings (SBERT) to cluster semantically duplicate items. 

	#. Bias & Fairness Control 

		- LLMs detect biased patterns (e.g., over-representing a certain demographic) and adjust recommendations accordingly. 
		- Example A job recommendation system might over-recommend tech jobs to menLLMs can balance exposure. 
		- Technique Use LLM-based fairness models (e.g., DebiasBERT) to adjust recommendations. 

	#. Context-Aware Filtering 

		- LLMs generate filtering rules on-the-fly based on user profile, session history, or external trends. 
		- Example If a user browses vegetarian recipes, LLMs downrank meat-based recipes dynamically. 
		- Technique Use GPT-powered filtering prompts to dynamically adjust content selection. 

Pros 

	- Prevents Repetitive Recommendations - Ensures users dont see redundant items. 
	- Improves Fairness & Representation - Adjusts for bias in candidate selection. 

Cons 

	- Computationally Expensive - Filtering millions of candidates using LLMs can increase inference costs. 
	- Difficult to Fine-Tune - Over-filtering may hide relevant recommendations. 

Re-Ranking Stage
------------------------------------------------------------------------------------
.. note::
	Goal Final Diversity Adjustments

Determinantal Point Processes (DPP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	- Uses probabilistic modeling to diversify ranked lists.
	- Given a candidate set, DPP selects a subset that maximizes diversity while maintaining relevance.
	- Works by modeling similarity between items and ensuring that similar items are not ranked too closely together.

Pros:

	- Mathematically principled and ensures diversity without arbitrary rules.
	- Used successfully in Spotify and Amazon for playlist & product recommendations.

Cons:

	- Computationally expensive, especially in large-scale deployments.
	- Needs proper similarity functions to be effective.

Example:

	- Spotify Playlist Generation - Ensures a playlist has a variety of artists and genres instead of only one type of song.

Re-Ranking with Diversity Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	- Uses weighted re-ranking algorithms that explicitly penalize redundant recommendations.
	- Can be tuned to balance diversity vs. personalization dynamically.

Pros:

	- Adjustable trade-off between diversity and user preferences.
	- Works well for personalized recommendations.

Cons:

	- Needs constant tuning to find the right balance.
	- If misconfigured, can make recommendations feel random or irrelevant.

Example:

	- YouTubes Ranking Model applies re-ranking constraints to prevent over-recommendation of a single creator in a session.

Multi-Armed Bandits for Explore-Exploit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	- Balances exploitation (showing relevant, known content) with exploration (introducing new, diverse content).
	- Upper Confidence Bound (UCB), Thompson Sampling are commonly used bandit techniques.

Pros:

	- Encourages personalized discovery while ensuring exploration.
	- Automatically adapts over time.

Cons:

	- Hard to tune exploration parameters in production settings.
	- May result in temporary engagement drops during exploration phases.

Example:

	- TikToks For You Page mixes known preferences with new content using bandit-based ranking.

LLMs for Diversity-Aware Ranking 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	#. Diversity-Aware Ranking Models 

		- LLMs act as personalization-aware rerankers, balancing relevance with diversity dynamically. 
		- Example Instead of showing only Marvel movies to a fan, LLMs inject DC movies or indie superhero films. 
		- Technique Use LLM-powered diversity re-ranking prompts in post-processing. 

	#. Personalized Exploration vs. Exploitation 

		- LLMs simulate user preferences in real-time and adjust ranking to include more exploration. 
		- Example In TikTok, if a user likes cooking videos, LLMs inject some fitness or travel videos to encourage exploration. 
		- Technique Use GPT-powered bandit re-ranking for adaptive diversity balancing. 

	#. Diversity-Aware Re-Ranking via Counterfactual Predictions 

		- LLMs generate counterfactual recommendations to test how users might respond to different recommendation lists. 
		- Example Instead of showing only trending news, LLMs inject underrepresented topics and measure user responses. 
		- Technique Use LLMs for offline counterfactual testing before deployment. 

Pros 

	- Balances Personalization & Diversity - Prevents filter bubbles. 
	- Improves Long-Term Engagement - Users are less likely to get bored. 

Cons 

	- Higher Inference Cost - Re-ranking every session in real-time increases server load. 
	- Risk of Over-Exploration - If diversity is forced, users may feel the system is less relevant.

Distribution Shift
====================================================================================
Identification
------------------------------------------------------------------------------------
Refer to Observability page

Addressal
------------------------------------------------------------------------------------
(A) Continuous Model Updating & Online Learning 

	- Solution Train fresh models on recent data to ensure up-to-date recommendations. 
	- Trade-Offs 

		- Frequent retraining is computationally expensive. 
		- Requires robust online learning pipelines (feature stores, incremental updates). 

Example 

	- Google Search updates its ranking models regularly to adapt to evolving search trends. 
	- Spotify retrains user embeddings frequently to reflect shifting music preferences. 

(B) Adaptive Sampling & Reweighting Older Data 

	- Solution Weight recent data more heavily while retaining historical knowledge for long-term trends. 
	- Trade-Offs 

		- Overweighting recent data may cause catastrophic forgetting of long-term preferences. 
		- Requires tuning of decay rates (e.g., exponential decay). 

Example 

	- E-Commerce platforms (Amazon, Walmart) use time-decayed embeddings to keep recommendations fresh. 

(C) Real-Time Personalization Using Session-Based Models 

	- Solution Use short-term session-based models (Transformers, RNNs) that adapt to recent interactions. 
	- Trade-Offs 

		- Session models work well short-term but lack long-term personalization. 
		- Requires fast inference pipelines (low latency). 

Example 

	- TikToks recommender adapts within a session, adjusting based on user behavior in real-time. 

(D) Reinforcement Learning for Adaptive Ranking 

	- Solution Use reinforcement learning (RL) models to dynamically adapt rankings based on user feedback. 
	- Trade-Offs 

		- RL models require a lot of data to converge. 
		- Training RL models online is computationally expensive. 

Example 

	- YouTubes ranking system adapts via reinforcement learning to balance freshness & engagement. 

(E) Hybrid Ensembles (Mixing Old & New Models) 

	- Solution Use an ensemble of multiple models trained on different time periods, allowing a blend of fresh & historical preferences. 
	- Trade-Offs 

		- Combining models increases complexity. 
		- Requires ensemble weighting tuning to balance long-term vs. short-term data. 

Example 

		- Netflix blends long-term preference models with session-based recommendations. 

************************************************************************************
Domain Knowledge
************************************************************************************
Search Advertising
====================================================================================
Relevance vs. Revenue Trade-Off
------------------------------------------------------------------------------------
Why It Matters 

	- Advertisers bid for visibility, but their ads may not always be relevant to the user's query. 
	- If high-bid but low-relevance ads are shown, users may lose trust in the search engine. 

Strategic Solutions & Trade-Offs 

	- Quality Score (Google Ads' Approach)  Ranks ads based on a combination of CTR, relevance, and landing page experience, not just bid amount. 
	- Hybrid Ranking Model (Revenue + User Engagement)  Balances ad revenue vs. user satisfaction. 

Trade-Offs 

	- Prioritizing high-relevance, low-bid ads reduces short-term revenue. 
	- Prioritizing high-bid, low-relevance ads hurts user trust & long-term retention. 

Click Spam & Ad Fraud
------------------------------------------------------------------------------------
Why It Matters 

	- Bots & malicious actors inflate clicks to waste competitor ad budgets (click fraud). 
	- Some advertisers run low-quality, misleading ads to generate fake engagement. 

Strategic Solutions & Trade-Offs 

	- Click Fraud Detection (Googles Invalid Click Detection)  Uses IP tracking, anomaly detection, and ML models to filter fraudulent clicks. 
	- Post-Click Analysis (User Behavior Analysis)  Detects bots based on engagement (bounce rate, session length, interactions). 

Trade-Offs 

	- False Positives  May block legitimate traffic, harming advertisers. 
	- False Negatives  Fraudulent clicks still get monetized, increasing costs for real advertisers. 

Ad Auction Manipulation & Bid Shading
------------------------------------------------------------------------------------
Why It Matters 

	- Sophisticated advertisers use AI-driven bidding strategies to game real-time auctions. 
	- Bid shading techniques lower ad costs while maintaining high visibility. 

Strategic Solutions & Trade-Offs 

	- Second-Price Auctions (Vickrey Auctions)  Advertisers only pay the second-highest bid price, reducing manipulation. 
	- Multi-Objective Bidding Models  Balances advertiser cost efficiency and search engine revenue. 

Trade-Offs 

	- Too much bid control reduces revenue  Search engines may earn less per click. 
	- Aggressive bid adjustments can reduce advertiser trust  If advertisers feel theyre losing transparency, they may pull budgets. 

Ad Fatigue & Banner Blindness
------------------------------------------------------------------------------------
Why It Matters 

	- Users ignore repetitive ads after multiple exposures, reducing CTR over time. 
	- If ads look too much like organic results, users may feel deceived. 

Strategic Solutions & Trade-Offs 

	- Adaptive Ad Rotation (Google Ads Optimize for Best Performing Mode)  Dynamically swaps low-performing ads with higher-engagement creatives. 
	- Ad Labeling Transparency  Clearer Sponsored tags improve user trust but reduce click rates. 

Trade-Offs 

	- Refreshing ads too frequently raises advertiser costs. 
	- Too much ad transparency leads to lower revenue per impression. 

Query Intent Mismatch
------------------------------------------------------------------------------------
Why It Matters 

	- Search queries are often ambiguous, and poor ad matching leads to bad user experience. 
	- Example Searching for Apple  Should the search engine show Apple iPhones (commercial intent) or apple fruit (informational intent)? 

Strategic Solutions & Trade-Offs 

	- Intent Classification Models (BERT, T5-based Models)  Classify queries into commercial vs. informational intent. 
	- Negative Keyword Targeting (Google Ads' Negative Keywords)  Advertisers block unrelated queries from triggering their ads. 

Trade-Offs 

	- Restricting ads based on intent can lower revenue. 
	- Allowing broad ad targeting risks user dissatisfaction. 

Landing Page Experience & Conversion Rate Optimization
------------------------------------------------------------------------------------
Why It Matters 

	- Even if an ad gets high CTR, if the landing page is misleading or slow, users bounce without converting. 
	- Google penalizes low-quality landing pages via Quality Score reductions. 

Strategic Solutions & Trade-Offs 

	- Landing Page Quality Audits (Googles Ad Quality Guidelines)  Checks for page speed, relevance, mobile-friendliness. 
	- Post-Click Engagement Monitoring  Uses bounce rate, time-on-site, conversion tracking to refine ranking. 

Trade-Offs 

	- Strict landing page rules limit advertiser flexibility. 
	- Relaxed rules allow low-quality ads, reducing long-term trust. 

Multi-Touch Attribution & Ad Budget Allocation
------------------------------------------------------------------------------------
Why It Matters 

	- Users may see an ad but not convert immediately  Traditional last-click attribution ignores earlier touchpoints. 
	- Advertisers struggle to allocate budgets across search, display, social, and video ads. 

Strategic Solutions & Trade-Offs 

	- Multi-Touch Attribution Models (Shapley Value, Markov Chains)  Assigns fair credit to different ad exposures. 
	- Cross-Channel Conversion Tracking  Tracks user journeys across search & display ads. 

Trade-Offs 

	- More complex attribution models require longer training times. 
	- Over-attributing upper-funnel ads can inflate costs without clear ROI. 

Fairness & Ad Bias Issues
------------------------------------------------------------------------------------
Why It Matters 

	- Some ad auctions are biased against small advertisers, favoring large ad budgets. 
	- Discriminatory ad targeting (e.g., gender/race bias in job/housing ads) can lead to regulatory penalties. 

Strategic Solutions & Trade-Offs 

	- Fairness-Constrained Bidding (Googles Fairness-Aware Ad Auctions)  Adjusts auction weights to prevent dominance by large advertisers. 
	- Bias Detection in Ad Targeting (Auditing Models for Discriminatory Targeting)  Ensures fair exposure of diverse ads. 

Trade-Offs 

	- Too much fairness correction may reduce revenue from high-bidding advertisers. 
	- Too little correction risks regulatory lawsuits (e.g., Facebooks 2019 lawsuit for discriminatory ad targeting). 

Music
====================================================================================
Playlist Generation & Curation in Music Recommendation Systems
------------------------------------------------------------------------------------
Types of Playlists & Their Challenges
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. csv-table::
	:header: "Playlist Type", "Example", "Key Challenges"
	:align: center

		Personalized Playlists, Spotifys Discover Weekly; YouTube Musics Your Mix, Ensuring balance between familiar & new tracks.
		Mood/Activity-Based Playlists, Workout Mix; Chill Vibes; Focus Music, Detecting mood & intent dynamically.
		Trending & Algorithmic Playlists, Spotifys Top 50; Apple Musics Charts, Avoiding popularity bias while staying relevant.
		Collaborative & Social Playlists, Spotify Blend; Apple Musics Shared Playlists, Handling conflicting preferences in shared lists.
		Genre/Artist-Centric Playlists, Best of 90s Rock; Jazz Classics, Ensuring diversity within a theme.

Solutions to Key Playlist Challenges
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. csv-table::
	:header: "Challenge", "Solution", "Trade-Off"
	:align: center

		Over-Personalization (Echo Chamber), Inject 5-20% exploration (Multi-Armed Bandits), Too much exploration may decrease CTR
		Repetition & Content Fatigue, Anti-repetition rules (e.g.; same song cannot appear in back-to-back sessions), May prevent users from hearing favorite tracks
		Cold-Start for New Songs, Boost underexposed songs using metadata (tempo; genre), Over-promoting new songs may harm engagement
		Context-Aware Playlists, Use real-time signals (e.g.; running mode detects movement; adjusts tempo), Misinterpreted context may cause poor recommendations
		Playlist Completion Rate, Optimize for average session length (shorter playlists for casual users; longer for engaged users), Shorter playlists may reduce playtime per session

Common Problems
------------------------------------------------------------------------------------
Cold-Start Problem for New Artists & Songs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Why It Matters:

	- New artists and newly released tracks struggle to get exposure since they have no engagement history.

- Strategic Solutions & Trade-Offs:

	- Metadata-Based Recommendations (Genre, BPM, lyrics embeddings)  Useful for early exposure but lacks engagement feedback.
	- Collaborative Boosting (Linking new artists to known artists)  Improves visibility but risks inaccurate pairing.
	- User-Driven Exploration (Playlists like Fresh Finds)  Promotes new songs but may not reach mainstream listeners.

- Example:

	- Spotifys Fresh Finds is a human-curated playlist designed for emerging artists.

Popularity Bias & Lack of Exposure for Niche Artists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Why It Matters:

	- Big-label artists dominate recommendations, making it hard for new/independent musicians to gain visibility.
	- Overemphasis on top charts and algorithmic repetition reinforces the same mainstream music.

- Strategic Solutions & Trade-Offs:

	- Fairness-Aware Re-Ranking (Exposing lesser-known artists)  Promotes diversity but may reduce engagement.
	- User Preference-Based Exploration (Blending familiar & new artists)  Increases discovery but harder to balance.
	- Contextual Boosting (Surfacing niche content in certain playlists)  Encourages exploration but risks user dissatisfaction.

- Spotifys Fix:

	- Discover Weekly and Release Radar to highlight emerging artists.

Balancing Exploration vs. Personalization in Playlists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Why It Matters:

	- Users want to hear familiar songs but also expect discovery of new tracks.
	- Too much exploration reduces engagement, too little keeps users stuck in their existing preferences.

- Strategic Solutions & Trade-Offs:

	- Reinforcement Learning-Based Ranking (Balancing Novelty & Familiarity)  Dynamically adjusts exploration but requires more data.
	- Hybrid Personalized Playlists (50% known, 50% new)  Encourages discovery but still risks disengagement.
	- Diversity Re-Ranking Models (Ensuring mix of different artist popularity levels)  Enhances engagement but increases complexity.

- Spotifys Fix:

	- Discover Weekly mixes familiar artists with newly recommended artists.

Repetition & Content Fatigue (Avoiding Overplayed Songs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Why It Matters:

	- Users dislike hearing the same songs too frequently in personalized playlists.
	- Music recommendation systems tend to reinforce top tracks due to high past engagement.

- Strategic Solutions & Trade-Offs:

	- Play-Session Awareness (Avoiding recently played tracks)  Prevents fatigue but risks reducing personalization strength.
	- Diversified Playlist Generation (Embedding Clustering)  Encourages discovery but may introduce unrelated tracks.
	- Temporal Diversity Constraints (Recommender-aware time gaps)  Reduces overexposure but adds complexity to ranking models.

- Spotify & Apple Musics Fix:

	- Autogenerated playlists (e.g., Daily Mix, Radio) have anti-repetition constraints.

Context-Aware Recommendations (Music for Different Situations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Why It Matters:

	- Music preferences vary by context (workout, driving, studying, relaxing), but most recommenders treat all listening the same.

- Strategic Solutions & Trade-Offs:

	- User-Controlled Context Tags (Spotifys Mood Playlists, YouTube Musics Activity Mode)  More control but adds friction.
	- Implicit Context Detection (Using location, time, device, previous context switches)  Improves automation but risks privacy concerns.
	- Adaptive Playlist Generation (Real-time context-aware re-ranking)  Better real-world usability but increases computational costs.

- Industry Example:

	- Spotifys Made for You mixes genres based on past listening sessions.

Short-Term vs. Long-Term Personalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Why It Matters:

	- Users music preferences change over time, but most recommendation models overly rely on recent activity.
	- Recommending only recently played songs can overfit short-term moods and ignore long-term preferences.

- Strategic Solutions & Trade-Offs:

	- Session-Based Personalization (Short-Term Context Models)  Captures mood-based preferences but can overfit recent choices.
	- Hybrid Long-Term + Short-Term Embeddings (Contrastive Learning on Listening History)  Balances nostalgia & discovery but computationally expensive.
	- Decay-Based Weighting on Past Behavior  Helps phase out stale preferences but requires careful tuning.

- Spotifys Approach:

	- Balances On Repeat (long-term) and Discover Weekly (exploration).

Multi-Modal Recommendation (Lyrics, Podcasts, Audio Similarity)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Why It Matters:

	- Music discovery can be driven by lyrics, themes, artist backstories, and spoken content (podcasts).
	- Traditional recommendation models focus only on collaborative filtering (listening history).

- Strategic Solutions & Trade-Offs:

	- Lyrics-Based Embeddings (Thematic music recommendations)  Enhances meaning-based recommendations but requires NLP processing.
	- Cross-Domain Music-Podcast Recommendation (Shared interests)  Improves discovery but harder to rank relevance.
	- Audio Similarity-Based Retrieval (Matching based on timbre, rhythm)  Better for organic discovery but requires deep learning models.

- Industry Example:

	- YouTube Music cross-recommends music & podcasts based on topics.

Social Media
====================================================================================

Video
====================================================================================

E-Commerce
====================================================================================
