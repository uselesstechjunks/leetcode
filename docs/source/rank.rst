####################################################################################
Search & Recommendation
####################################################################################
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

************************************************************************************
Resources
************************************************************************************
Metrics
====================================================================================
.. important::

	* [evidentlyai.com] `10 metrics to evaluate recommender and ranking systems <https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems>`_
	* [docs.evidentlyai.com] `Ranking metrics <https://docs.evidentlyai.com/reference/all-metrics/ranking-metrics>`_

Relevance
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
		Gini, Gini Index, :math:`1-\frac{1}{|I|-1}\sum_{k}^{|I|}(2k-|I|-1)p(i_k|L)`, :math:`p(i_k|L)`: how many times :math:`i_k` occured in `L`, Ignores user preference
		UDP, User Popularity Deviation, , ,
	
Diversity
------------------------------------------------------------------------------------
Personalsation
------------------------------------------------------------------------------------

Resources
====================================================================================
Overview: Stages
------------------------------------------------------------------------------------
.. csv-table:: 
	:header: "Stage", "Goals", "Key Metrics", "Common Techniques"
	:align: center
	
		Retrieval, Fetch diverse candidates from multiple sources, Recall@K; Coverage; Latency, Multi-tower models; ANN; User embeddings
		Combining & Filtering, Merge candidates; remove duplicates; apply business rules, Diversity; Precision@K; Fairness, Weighted merging; Min-hashing; Rule-based filtering
		Re-Ranking, Optimize order of recommendations for engagement, CTR; NDCG; Exploration Ratio, Neural Rankers; Bandits; DPP for diversity

Overview: Patterns
------------------------------------------------------------------------------------
.. csv-table:: 
	:header: "Pattern", "Traditional Approach", "LLM Augmentations"
	:align: center

		Query-Item, BM25; TF-IDF; Neural Ranking, LLM-based reranking; Query expansion
		Item-Item, Co-occurrence; Similarity Matching, Semantic matching; Multimodal embeddings
		User-Item, CF; Content-Based; Deep Learning, Personalized generation; Zero-shot preferences
		Session-Based, Sequential Models; Transformers, Few-shot reasoning; Context-aware personalization
		User-User, Graph-Based; Link Prediction, Profile-text analysis; Social graph augmentation

Overview: Common Issues
------------------------------------------------------------------------------------
- General Issues in Search & Recommendation Systems

	#. Cold-Start Problem (Users, items)
	#. Popularity Bias & Feedback Loops
	#. Short-Term Engagement vs. Long-Term User Retention
	#. Diversity vs. Personalization Trade-Off
	#. Real-Time Personalization & Latency Trade-Offs
	#. Balancing multiple business objectives (CTR vs. fairness vs. revenue)
	#. Cross-device and cross-session personalization
	#. Privacy concerns & compliance (GDPR, CCPA)
	#. Multi-modality & cross-domain recommendation challenges

- Domain-Specific Issues & Their Unique Challenges

	#. Search-Specific Issues: Query Understanding & Intent Disambiguation
	#. E-Commerce Issues: Balancing Revenue & User Satisfaction
	#. Video & Music Streaming: Content-Length Bias in Recommendations

Overview: Domains
------------------------------------------------------------------------------------
Sponsored Search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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

Music
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. csv-table:: 
	:header: "Challenge", "Why Its Important", "Trade-Offs"
	:align: center

		Personalization vs. Serendipity, Users want relevant music but also expect some new discoveries., Too much personalization  Feels repetitive. Too much exploration  Feels random.
		Repetition & Content Fatigue, Users get frustrated if the same songs appear too often., Strict anti-repetition  May exclude user favorites. Loose constraints  Risk of overplaying certain songs.
		Context & Mood Adaptation, Users listen to music differently based on mood; time; activity (workout; relaxation)., Explicit mood tagging is effective but requires manual input. Implicit context detection risks wrong assumptions.
		Balancing Popular & Niche Tracks, Highly popular songs dominate engagement; making it hard for lesser-known songs to gain exposure., Boosting niche tracks improves diversity; but may lower engagement metrics.
		Cold-Start for New Songs & Artists, Newly released songs struggle to get exposure due to lack of engagement signals., Over-boosting new music can lead to reduced user satisfaction.
		Playlist Length & Engagement Optimization, Users may not finish long playlists; leading to low engagement metrics., Shorter playlists increase completion rate; but longer ones improve session duration.

Videos
------------------------------------------------------------------------------------
- [youtube.com] `Stanford CS224W: Machine Learning w/ Graphs I 2023 I GNNs for Recommender Systems <https://www.youtube.com/watch?v=OV2VUApLUio>`_
.. note::
	- Mapped as an edge prediction problem in a bipartite graph
	- Ranking

		- Metric Recall@k (non differentiable)
		- Other metrics: HR@k, nDCG
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
		- Application: similar image recommendation in Pinterest 
		- Issue: doesn't have contextual awareness or session/temporal awareness

Key Papers
------------------------------------------------------------------------------------
	- BOF = Bag of features 
	- NG = N-Gram
	- CM = Causal Models (autoregressive)

.. csv-table:: 
	:header: "Tag", "Title"
	:align: center

		Two Tower; MLP, Neural Collaborative Filtering
		Two Tower; BOF, StarSpace: Embed All The Things!
		Two Tower; NG+BOF, Embedding-based Retrieval in Facebook Search
		GCN, LightGCN - Simplifying and Powering Graph Convolution Network for Recommendation
		CM; Session, Transformers4Rec: Bridging the Gap between NLP and Sequential / Session-Based Recommendation
		LLM, Collaborative Large Language Model for Recommender Systems
		LLM, Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach

More Papers
------------------------------------------------------------------------------------
.. csv-table:: 
	:header: "Year", "Title"
	:align: center

		2001,Item-Based Collaborative Filtering Recommendation Algorithms – Sarwar et al.
		2003,Amazon.com Recommendations: Item-to-Item Collaborative Filtering – Linden et al.
		2007,Link Prediction Approaches and Applications – Liben-Nowell et al.
		2008,An Introduction to Information Retrieval – Manning et al.
		2009,BM25 and Beyond – Robertson et al.
		2009,Matrix Factorization Techniques for Recommender Systems – Koren et al.
		2010,Who to Follow: Recommending People in Social Networks – Twitter Research
		2014,DeepWalk: Online Learning of Social Representations – Perozzi et al.
		2015,Learning Deep Representations for Content-Based Recommendation – Wang et al.
		2015,Netflix Recommendations: Beyond the 5 Stars – Gomez-Uribe et al.
		2016,Deep Neural Networks for YouTube Recommendations – Covington et al.
		2016,Wide & Deep Learning for Recommender Systems – Cheng et al.
		2016,Session-Based Recommendations with Recurrent Neural Networks – Hidasi et al.
		2017,DeepRank: A New Deep Architecture for Relevance Ranking in Information Retrieval – Pang et al.
		2017,Neural Collaborative Filtering – He et al.
		2017,A Guide to Neural Collaborative Filtering – He et al.
		2018,BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding – Devlin et al.
		2018,PinSage: Graph Convolutional Neural Networks for Web-Scale Recommender Systems – Ying et al.
		2018,Neural Architecture for Session-Based Recommendations – Tang & Wang
		2018,SASRec: Self-Attentive Sequential Recommendation – Kang & McAuley
		2018,Graph Convolutional Neural Networks for Web-Scale Recommender Systems – Ying et al.
		2019,Deep Learning Based Recommender System: A Survey and New Perspectives – Zhang et al.
		2019,Session-Based Recommendation with Graph Neural Networks – Wu et al.
		2019,Next Item Recommendation with Self-Attention – Sun et al.
		2019,BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations – Sun et al.
		2020,Dense Passage Retrieval for Open-Domain Question Answering – Karpukhin et al.
		2020,ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction Over BERT – Khattab et al.
		2020,T5 for Information Retrieval – Nogueira et al.
		2021,CLIP: Learning Transferable Visual Models from Natural Language Supervision – Radford et al.
		2021,Transformers4Rec: Bridging the Gap Between NLP and Sequential Recommendation – De Souza et al.
		2021,Graph Neural Networks: A Review of Methods and Applications – Wu et al.
		2021,Next-Item Prediction Using Pretrained Language Models – Sun et al.
		2022,Unified Vision-Language Pretraining for E-Commerce Recommendations – Wang et al.
		2022,Contextual Item Recommendation with Pretrained LLMs – Li et al.
		2023,InstructGPT for Information Retrieval – Ouyang et al.
		2023,GPT-4 for Web Search Augmentation – Bender et al.
		2023,CLIP-Recommend: Multimodal Learning for E-Commerce Recommendations – Xu et al.
		2023,Semantic-Aware Item Matching with Large Language Models – Chen et al.
		2023,GPT4Rec: A Generative Framework for Personalized Recommendation – Wang et al.
		2023,LLM-based Collaborative Filtering: Enhancing Recommendations with Large Language Models – Liu et al.
		2023,LLM-Powered Dynamic Personalized Recommendations – Guo et al.
		2023,Real-Time Recommendation with Large Language Models – Zhang et al.
		2023,Graph Neural Networks Meet Large Language Models: A Survey – Wu et al.
		2023,LLM-powered Social Graph Completion for Friend Recommendations – Huang et al.
		2023,LLM-Augmented Node Classification in Social Networks – Zhang et al.

************************************************************************************
Stages
************************************************************************************
A large-scale recommendation system consists of multiple stages designed to efficiently retrieve, filter, and rank items to maximize user engagement and satisfaction. The three primary stages are Retrieval, Combining & Filtering, and Re-Ranking.  

Retrieval  
====================================================================================
(Fetching an initial candidate pool from multiple sources)  

Goals:  
------------------------------------------------------------------------------------
	- Reduce a large item pool (millions of candidates) to a manageable number (thousands).  
	- Retrieve diverse candidates from multiple sources that might be relevant to the user.  
	- Balance long-term preferences vs. short-term intent.  

Metrics to Optimize For:  
------------------------------------------------------------------------------------
	- Recall@K – How many relevant items are in the top-K retrieved items?  
	- Coverage – Ensuring diversity by retrieving from multiple pools.  
	- Latency – Efficient retrieval in milliseconds at large scales.  

Common Techniques for Different Goals:  
------------------------------------------------------------------------------------
.. csv-table:: 
	:header: "Goal", "Techniques"
	:align: center

		Heterogeneous Candidate Retrieval, Multi-tower models; Hybrid retrieval (Collaborative Filtering + Content-Based)
		Personalization, User embeddings (e.g.; Two-Tower models; Matrix Factorization)
		Exploration & Freshness, Real-time embeddings; Bandit-based exploration
		Scalability & Efficiency, Approximate Nearest Neighbors (ANN); FAISS; HNSW
		Cold-Start Handling, Content-based retrieval (TF-IDF; BERT); Popularity-based heuristics

Example - YouTube Recommendation:  
------------------------------------------------------------------------------------
	- Candidate pools: Watched videos, partially watched videos, topic-based videos, demographically popular videos, newly uploaded videos, videos from followed channels.  
	- Techniques used: Two-Tower model for retrieval, Approximate Nearest Neighbors (ANN) for fast lookup.  

Combining & Filtering  
====================================================================================
(Merging retrieved candidates from different sources and removing low-quality items)  

Goals:  
------------------------------------------------------------------------------------
	- Merge multiple retrieved pools and assign confidence scores to each source.  
	- Filter out irrelevant, duplicate, or low-quality candidates.  
	- Apply business rules (e.g., compliance filtering, removing expired content).  

Metrics to Optimize For:  
------------------------------------------------------------------------------------
	- Diversity – Ensuring different content types are represented.  
	- Precision@K – How many retrieved items are actually relevant?  
	- Fairness & Representation – Avoiding over-exposure of popular items.  
	- Latency – Keeping the filtering process efficient.  

Common Techniques for Different Goals:  
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

Example - Newsfeed Recommendation:  
------------------------------------------------------------------------------------
	- Candidate sources: Text posts, image posts, video posts.  
	- Filtering techniques: Removing duplicate posts, blocking low-quality content, filtering based on engagement thresholds.  

Re-Ranking  
====================================================================================
(Final ranking of candidates based on personalization, diversity, and explore-exploit trade-offs)  

Goals:  
------------------------------------------------------------------------------------
	- Optimize the order of candidates to maximize engagement.  
	- Balance personalization with exploration (ensuring new content gets surfaced).  
	- Ensure fairness and representation (avoid showing only highly popular items).  

Metrics to Optimize For:  
------------------------------------------------------------------------------------
	- CTR (Click-Through Rate) – Measures immediate engagement.  
	- NDCG (Normalized Discounted Cumulative Gain) – Measures ranking quality.  
	- Exploration Ratio – Tracks new content shown to users.  
	- Long-Term Engagement – Measures retention and repeat interactions.  

Common Techniques for Different Goals:  
------------------------------------------------------------------------------------
.. csv-table:: 
	:header: "Goal", "Techniques"
	:align: center

		Personalized Ranking, Neural Ranking Models (e.g.; DeepFM; Wide & Deep; Transformer-based rankers)
		Diversity Promotion, Determinantal Point Processes (DPP); Re-ranking by category
		Explore-Exploit Balance, Multi-Armed Bandits (Thompson Sampling; UCB); Randomized Ranking
		Handling Highly Popular Items, Popularity dampening; Re-ranking with popularity decay
		Fairness & Representation, Re-weighting models; Exposure-aware ranking
		Fast Re-Ranking, Tree-based models (GBDT); LightGBM; XGBoost

Example - TikTok Recommendation:  
------------------------------------------------------------------------------------
	- Challenges: Need to mix trending videos, personalized content, and fresh videos.  
	- Techniques used: Transformer-based ranking, popularity dampening, diversity-based re-ranking.  

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
- Query-item recommendation is the foundation of search systems, where a user provides a query (text, image, voice, etc.), and the system retrieves the most relevant items. Unlike standard recommendations, search is explicit—users express intent directly.  

- Common approaches include:  

	- Lexical Matching (TF-IDF, BM25, keyword-based retrieval)  
	- Semantic Matching (Word embeddings, Transformer models like BERT, CLIP for vision-text matching)  
	- Hybrid Search (Combining lexical and semantic search, e.g., BM25 + embeddings)  
	- Learning-to-Rank (LTR) models optimizing ranking performance based on user interactions)  
	- Multimodal Search (Image-to-text retrieval, video search, voice search, etc.)  

Key Papers to Read  
------------------------------------------------------------------------------------
#. Traditional Information Retrieval  

	- "An Introduction to Information Retrieval" – Manning et al. (2008)  
	- "BM25 and Beyond" – Robertson et al. (2009)  

#. Neural Ranking Models  

	- "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" – Devlin et al. (2018)  
	- "Dense Passage Retrieval for Open-Domain Question Answering" – Karpukhin et al. (2020)  

#. Multimodal & Deep Learning-Based Search  

	- "CLIP: Learning Transferable Visual Models from Natural Language Supervision" – Radford et al. (2021)  
	- "DeepRank: A New Deep Architecture for Relevance Ranking in Information Retrieval" – Pang et al. (2017)  

Gathering Training Data & Labels  
------------------------------------------------------------------------------------
#. Supervised Learning:  

	- Label: Binary (clicked vs. not clicked) or relevance score (explicit ratings, dwell time).  
	- Data sources: Search logs, query-click data, user feedback (thumbs up/down).  
	- Challenges: Noisy labels (e.g., clicks may not always indicate relevance).  

#. Semi-Supervised Learning:  

	- Use query expansion techniques (e.g., weak supervision from similar queries).  
	- Leverage pseudo-labeling (e.g., use a weaker ranker to generate labels).  

#. Self-Supervised Learning:  

	- Contrastive learning (e.g., train embeddings by pulling query and relevant items closer).  
	- Masked query prediction (e.g., predicting missing words in search queries).  

Feature Engineering  
------------------------------------------------------------------------------------
- Query Features: Term frequency, query length, part-of-speech tagging.  
- Item Features: Title, description, category, metadata, embeddings.  
- Interaction Features: Click history, query-to-item dwell time, CTR.  
- Contextual Features: Time of query, device type, user history.  
- Embedding-Based Features: Pretrained word embeddings (Word2Vec, FastText, BERT embeddings).  

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

- Common approaches include:  

	- Item-Based Collaborative Filtering (Similarity between item interaction histories)  
	- Content-Based Filtering (Similarity using item attributes like text, image, category)  
	- Graph-Based Approaches (Item-item similarity using co-purchase graphs)  
	- Deep Learning Methods (Representation learning, embeddings)  
	- Hybrid Methods (Combining multiple approaches)  

Key Papers to Read  
------------------------------------------------------------------------------------
#. Collaborative Filtering-Based Approaches  

	- "Item-Based Collaborative Filtering Recommendation Algorithms" – Sarwar et al. (2001)  
	- "Matrix Factorization Techniques for Recommender Systems" – Koren et al. (2009)  

#. Content-Based Approaches  

	- "Learning Deep Representations for Content-Based Recommendation" – Wang et al. (2015)  
	- "Deep Learning Based Recommender System: A Survey and New Perspectives" – Zhang et al. (2019)  

#. Graph-Based & Hybrid Approaches  

	- "Amazon.com Recommendations: Item-to-Item Collaborative Filtering" – Linden et al. (2003)  
	- "PinSage: Graph Convolutional Neural Networks for Web-Scale Recommender Systems" – Ying et al. (2018)  

Gathering Training Data & Labels  
------------------------------------------------------------------------------------
#. Supervised Learning:  

	- Label: Binary (1 = two items are similar, 0 = not similar).  
	- Data sources: Co-purchase data, co-click data, content similarity.  
	- Challenges: Defining meaningful similarity when explicit labels don’t exist.  

#. Semi-Supervised Learning:  

	- Clustering similar items based on embeddings or co-occurrence.  
	- Weak supervision from user-generated tags, reviews.  

#. Self-Supervised Learning:  

	- Contrastive learning (e.g., learning embeddings by pushing dissimilar items apart).  
	- Masked item prediction (e.g., predicting missing related items in a session).  

Feature Engineering  
------------------------------------------------------------------------------------
- Item Features: Category, brand, price, textual description, images.  
- Interaction Features: Co-purchase counts, view sequences, co-engagement.  
- Graph Features: Item co-occurrence in user sessions, citation networks.  
- Embedding-Based Features: Learned latent item representations.  
- Contextual Features: Time decay (trending vs. evergreen items).   

User-Item Recommendation  
====================================================================================
- Homepage recommendations
- product recommendations
- videos you might like, etc

Key Concept  
------------------------------------------------------------------------------------
- User-item recommendation focuses on predicting a user's preference for an item based on historical interactions. This can be framed as:  

	#. Explicit feedback (e.g., ratings, thumbs up/down)  
	#. Implicit feedback (e.g., clicks, watch time, purchases)  

- Common approaches include:  

	#. Collaborative Filtering (CF) (Matrix Factorization, Neural CF)  
	#. Content-Based Filtering (Feature-based models)  
	#. Hybrid Models (Combining CF and content-based methods)  
	#. Deep Learning Approaches (Neural networks, Transformers)  

Key Papers to Read  
------------------------------------------------------------------------------------
#. Collaborative Filtering  

	- "Matrix Factorization Techniques for Recommender Systems" – Koren et al. (2009)  
	- "Neural Collaborative Filtering" – He et al. (2017)  

#. Deep Learning Approaches  

	- "Deep Neural Networks for YouTube Recommendations" – Covington et al. (2016)  
	- "Wide & Deep Learning for Recommender Systems" – Cheng et al. (2016)  

#. Hybrid and Production Systems  

	- "Netflix Recommendations: Beyond the 5 Stars" – Gomez-Uribe et al. (2015)  

Gathering Training Data & Labels  
------------------------------------------------------------------------------------
#. Supervised Learning:  

	- Label: binary (clicked/not clicked, purchased/not purchased) or continuous (watch time, rating).  
	- Data sources: user interactions, purchase logs, watch history.  
	- Challenges: Class imbalance (many more non-clicked items than clicked ones).  

#. Semi-Supervised Learning:  

	- Use self-training (pseudo-labeling) to expand labeled data.  
	- Graph-based methods to propagate labels across similar users/items.  

#. Self-Supervised Learning:  

	- Contrastive learning (e.g., SimCLR, BERT-style masked item prediction).  
	- Learning representations via session-based modeling (e.g., predicting the next item a user interacts with).  

Feature Engineering  
------------------------------------------------------------------------------------
- User Features: Past interactions, demographics, engagement signals.  
- Item Features: Category, text/image embeddings, historical engagement.  
- Cross Features: User-item interactions (e.g., user’s affinity to a category).  
- Contextual Features: Time of day, device, location.  
- Embedding-based Features: Learned latent factors from models like Word2Vec for items/users.  

Session-Based Recommendation  
====================================================================================
- Personalized recommendations based on recent user actions
- short-term intent modeling
- sequential recommendations

Key Concept  
------------------------------------------------------------------------------------
Session-based recommendation focuses on predicting the next relevant item for a user based on their recent interactions, rather than long-term historical data. This is useful when:  

	- Users don’t have extensive histories (e.g., guest users).  
	- Preferences shift dynamically (e.g., browsing sessions in e-commerce).  
	- Recent behavior is more indicative of intent than long-term history.  

Common approaches include:  

	- Rule-Based Methods (Most popular, trending, or recently viewed items)  
	- Markov Chains & Sequential Models (Predicting next item based on state transitions)  
	- Recurrent Neural Networks (RNNs, GRUs, LSTMs) (Capturing sequential dependencies)  
	- Graph-Based Approaches (Session-based Graph Neural Networks)  
	- Transformer-Based Models (Attention-based architectures for session modeling)  

Key Papers to Read  
------------------------------------------------------------------------------------
#. Traditional Approaches & Sequential Models  

	- "Session-Based Recommendations with Recurrent Neural Networks" – Hidasi et al. (2016)  
	- "Neural Architecture for Session-Based Recommendations" – Tang & Wang (2018)  

#. Graph-Based Methods  

	- "Session-Based Recommendation with Graph Neural Networks" – Wu et al. (2019)  
	- "Next Item Recommendation with Self-Attention" – Sun et al. (2019)  

#. Transformer-Based Methods  

	- "SASRec: Self-Attentive Sequential Recommendation" – Kang & McAuley (2018)  
	- "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations" – Sun et al. (2019)  

Gathering Training Data & Labels  
------------------------------------------------------------------------------------
#. Supervised Learning:  

	- Label: Next item in sequence (e.g., clicked/purchased item).  
	- Data sources: User sessions, browsing logs, cart abandonment data.  
	- Challenges: Short sessions make training harder; sparse interaction data.  

#. Semi-Supervised Learning:  

	- Use self-supervised tasks like predicting masked interactions.  
	- Graph-based node propagation to learn session similarities.  

#. Self-Supervised Learning:  

	- Contrastive learning (e.g., predict next item from different user sessions).  
	- Next-click prediction using masked sequence modeling (BERT-style).  

Feature Engineering  
------------------------------------------------------------------------------------
- Session Features: Time spent, number of items viewed, recency of last interaction.  
- Item Features: Product category, textual embeddings, popularity trends.  
- Sequence Features: Click sequences, time gaps between interactions.  
- Contextual Features: Device type, time of day, geographical location.  
- Embedding-Based Features: Pretrained session embeddings (e.g., Word2Vec-like for items).  

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

- Common approaches include:  

	#. Collaborative Filtering (User-Based CF)  
	#. Graph-Based Approaches (Graph Neural Networks, PageRank, Node2Vec, etc.)  
	#. Feature-Based Matching (Demographic and behavior similarity)  
	#. Hybrid Approaches (Graph + CF + Deep Learning)  

Key Papers to Read  
------------------------------------------------------------------------------------
#. Collaborative Filtering-Based Approaches  

	- "Item-Based Collaborative Filtering Recommendation Algorithms" – Sarwar et al. (2001)  
	- "A Guide to Neural Collaborative Filtering" – He et al. (2017)  

#. Graph-Based Approaches  

	- "DeepWalk: Online Learning of Social Representations" – Perozzi et al. (2014)  
	- "Graph Convolutional Neural Networks for Web-Scale Recommender Systems" – Ying et al. (2018)  
	- "Graph Neural Networks: A Review of Methods and Applications" – Wu et al. (2021)  

#. Hybrid and Large-Scale User-User Recommendation  

	- "Link Prediction Approaches and Applications" – Liben-Nowell et al. (2007)  
	- "Who to Follow: Recommending People in Social Networks" – Twitter Research (2010)  

Gathering Training Data & Labels  
------------------------------------------------------------------------------------
#. Supervised Learning:  

	- Label: Binary (1 = connection exists, 0 = no connection).  
	- Data sources: Friendship graphs, follow/unfollow actions, mutual interests.  
	- Challenges: Highly imbalanced data (most user pairs are not connected).  

#. Semi-Supervised Learning:  

	- Graph-based label propagation (e.g., predicting missing edges in a user graph).  
	- Use unlabeled users with weak supervision from social structures.  

#. Self-Supervised Learning:  

	- Contrastive learning (e.g., learning embeddings where connected users are closer in vector space).  
	- Masked edge prediction (e.g., hide some connections and train the model to reconstruct them).  

Feature Engineering  
------------------------------------------------------------------------------------
- User Features: Profile attributes (age, location, industry, interests).  
- Graph Features: Common neighbors, Jaccard similarity, Adamic-Adar score.  
- Interaction Features: Message frequency, engagement level.  
- Embedding-Based Features: Node2Vec or GNN-based embeddings.  
- Contextual Features: Activity time, shared communities.  

************************************************************************************
LLM Applications
************************************************************************************
Leveraging Large Language Models (LLMs) like GPT, BERT, and T5 for various recommendation patterns

Query-Item Recommendation
====================================================================================
Key Concept  
------------------------------------------------------------------------------------
- Traditional search relies on lexical matching (BM25, TF-IDF) or vector search.  
- LLMs enhance ranking via reranking models (ColBERT, T5-based retrieval).  
- Can be used for query expansion, understanding user intent, and handling ambiguous queries.  
- Example use case: Google Search, AI-driven Q&A search (Perplexity AI).  

Key Papers to Read  
------------------------------------------------------------------------------------
#. LLM-Based Search Ranking  

	- "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction Over BERT" – Khattab et al. (2020)  
	- "T5 for Information Retrieval" – Nogueira et al. (2020)  
#. LLM-Augmented Search  

	- "InstructGPT for Information Retrieval" – Ouyang et al. (2023)  
	- "GPT-4 for Web Search Augmentation" – Bender et al. (2023)  

Item-Item Recommendation  
====================================================================================
Key Concept  
------------------------------------------------------------------------------------
- Traditional methods use co-occurrence matrices or content similarity (TF-IDF, embeddings).  
- LLMs improve semantic similarity scoring, identifying nuanced item relationships.  
- Multimodal LLMs (e.g., CLIP) combine text, images, and metadata to enhance recommendations.  
- Example use case: E-commerce (Amazon's “similar items”), content platforms (Netflix’s related videos).  

Key Papers to Read  
------------------------------------------------------------------------------------
#. Multimodal LLMs for Recommendation  

	- "CLIP-Recommend: Multimodal Learning for E-Commerce Recommendations" – Xu et al. (2023)  
	- "Unified Vision-Language Pretraining for E-Commerce Recommendations" – Wang et al. (2022)  
#. Semantic Similarity Using LLMs  

	- "Semantic-Aware Item Matching with Large Language Models" – Chen et al. (2023)  
	- "Contextual Item Recommendation with Pretrained LLMs" – Li et al. (2022)  

User-Item Recommendation  
====================================================================================
Key Concept  
------------------------------------------------------------------------------------
- Traditional approaches rely on collaborative filtering (CF) or content-based filtering to predict user preferences.  
- LLMs enhance this by learning richer user and item embeddings, capturing nuanced interactions.  
- LLMs can generate user preferences dynamically via zero-shot/few-shot learning, improving personalization.  
- Example use case: Personalized product descriptions, interactive recommendation assistants.  

Key Papers to Read  
------------------------------------------------------------------------------------
#. LLM-powered Recommendation  

	- "GPT4Rec: A Generative Framework for Personalized Recommendation" – Wang et al. (2023)  
	- "LLM-based Collaborative Filtering: Enhancing Recommendations with Large Language Models" – Liu et al. (2023)  
#. Transformer-Based RecSys  

	- "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations" – Sun et al. (2019)  
	- "SASRec: Self-Attentive Sequential Recommendation" – Kang & McAuley (2018)  

Session-Based Recommendation  
====================================================================================
Key Concept  
------------------------------------------------------------------------------------
- Traditional methods use sequential models (RNNs, GRUs, Transformers) to predict next-item interactions.  
- LLMs enhance session modeling by leveraging sequential reasoning and contextual awareness.  
- Few-shot prompting allows LLMs to infer session preferences without explicit training.  
- Example use case: Dynamic content feeds (TikTok), real-time recommendations (Spotify session playlists).  

Key Papers to Read  
------------------------------------------------------------------------------------
#. Transformer-Based Session Recommendations  

	- "SASRec: Self-Attentive Sequential Recommendation" – Kang & McAuley (2018)  
	- "Next-Item Prediction Using Pretrained Language Models" – Sun et al. (2021)  
#. LLM-Driven Dynamic Recommendation  

	- "LLM-Powered Dynamic Personalized Recommendations" – Guo et al. (2023)  
	- "Real-Time Recommendation with Large Language Models" – Zhang et al. (2023)  

User-User Recommendation  
====================================================================================
Key Concept  
------------------------------------------------------------------------------------
- Typically modeled as a graph-based link prediction problem, where users are nodes.  
- LLMs can enhance user similarity computations by processing richer profile texts (e.g., bios, chat history).  
- Social connections can be inferred by analyzing natural language data, rather than relying solely on structural graph features.  
- Example use case: Professional networking (LinkedIn), AI-assisted friend suggestions.  

Key Papers to Read  
------------------------------------------------------------------------------------
#. Graph-Based LLMs  

	- "Graph Neural Networks Meet Large Language Models: A Survey" – Wu et al. (2023)  
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
- Why It Matters:  

	- New users: No interaction history makes personalization difficult.  
	- New items: Struggle to get exposure due to lack of engagement signals.  

- Strategic Solutions & Trade-Offs:  

	- Content-Based Methods (Text embeddings, Image/Video features) → Good for new items, but lacks user personalization.  
	- Demographic-Based Recommendations (Cluster similar users) → Generalizes well but risks oversimplification.  
	- Randomized Exploration (Show new items randomly) → Increases fairness but can reduce CTR.  

- Domain-Specific Notes:  

	- E-commerce (Amazon, Etsy) → Cold-start for new sellers & niche products.  
	- Video Streaming (Netflix, YouTube) → Cold-start for newly released content.  

Popularity Bias & Feedback Loops
------------------------------------------------------------------------------------
- Why It Matters:  

	- Over-recommending already popular items creates a "rich-get-richer" effect.  
	- Items with low initial exposure struggle to gain traction.  
	- Reinforces biases in user engagement, making it harder to surface niche or novel content.  

- Strategic Solutions & Trade-Offs:  

	- Re-Ranking with Popularity Dampening (Decay-based adjustments) → Improves exposure but can hurt user satisfaction.  
	- Counterfactual Learning (Causal ML for fairness) → Breaks bias loops but hard to implement at scale.  
	- Multi-Armed Bandits (UCB, Thompson Sampling) → Helps exploration but can reduce short-term revenue.  

- Domain-Specific Notes:  

	- Social Media (TikTok, Twitter, Facebook) → Celebrity overexposure (e.g., verified users dominating feeds).  
	- News Aggregators (Google News, Apple News) → Same sources getting recommended (e.g., mainstream news over independent journalism).  

Short-Term Engagement vs. Long-Term User Retention  
------------------------------------------------------------------------------------
- Why It Matters:  

	- Systems often optimize for immediate engagement (CTR, watch time, purchases), which can lead to addictive behaviors or content fatigue.  
	- Over-exploitation of "sticky content" (clickbait, sensationalism, autoplay loops) may reduce long-term satisfaction.  

- Strategic Solutions & Trade-Offs:  

	- Multi-Objective Optimization (CTR + Long-Term Retention) → Complex to balance but essential for sustainability.  
	- Delayed Reward Models (Reinforcement Learning) → Great for long-term user retention but slow learning process.  
	- Personalization Decay (Balancing Freshness vs. Relevance) → Introduces diverse content but can feel random to users.  

- Domain-Specific Notes:  

	- YouTube, TikTok, Instagram → Prioritizing sensational viral content over educational material.  
	- E-Commerce (Amazon, Alibaba) → Short-term discounts vs. long-term brand loyalty.  

Diversity vs. Personalization Trade-Off  
------------------------------------------------------------------------------------
- Why It Matters:  

	- Highly personalized feeds often reinforce user preferences too strongly, limiting exposure to new content.  
	- Users may get stuck in content silos (e.g., political polarization, filter bubbles).  

- Strategic Solutions & Trade-Offs:  

	- Diversity-Promoting Re-Ranking (DPP, Exploration Buffers) → Reduces filter bubbles but may decrease engagement.  
	- Diversity-Constrained Search (Re-weighting ranking models) → Promotes varied content but risks reducing precision.  
	- Hybrid User-Item Graphs (Graph Neural Networks for diversification) → Balances exploration but requires expensive training.  

- Domain-Specific Notes:  

	- Social Media (Facebook, Twitter, YouTube) → Political echo chambers & misinformation bubbles.  
	- E-commerce (Amazon, Etsy, Zalando) → Users seeing only one type of product repeatedly.  

Real-Time Personalization & Latency Trade-Offs  
------------------------------------------------------------------------------------
- Why It Matters:  

	- Personalized recommendations require real-time feature updates and low-latency inference.  
	- Search relevance depends on immediate context (e.g., location, time of day, trending topics).  

- Strategic Solutions & Trade-Offs:  

	- Precomputed User Embeddings (FAISS, HNSW, Vector DBs) → Speeds up search but sacrifices personalization flexibility.  
	- Edge AI for On-Device Personalization → Reduces latency but increases computational costs.  
	- Session-Based Recommendation Models (Transformers for Session-Based Context) → Great for short-term personalization but expensive for large user bases.  

- Domain-Specific Notes:  

	- E-Commerce (Amazon, Walmart, Shopee) → Latency constraints for "similar item" recommendations.  
	- Search Engines (Google, Bing, Baidu) → Needing real-time personalization without slowing down results.  

Domain-Specific
====================================================================================
Search
------------------------------------------------------------------------------------  
- Query Understanding & Intent Disambiguation

	- Users enter ambiguous or vague queries, requiring intent inference.  
	- Example: Searching for “apple” – Is it a fruit, a company, or a music service?  
	- Solutions & Trade-Offs:  
	
		- LLM-Powered Query Rewriting (T5, GPT) → Improves relevance but risks over-modifying queries.  
		- Session-Aware Query Expansion → Helps disambiguation but increases computational cost.  

E-Commerce
------------------------------------------------------------------------------------
- Balancing Revenue & User Satisfaction  

	- Revenue-driven recommendations (sponsored ads, promoted products) vs. organic recommendations.  
	- Example: Amazon mixing sponsored and personalized search results.  
	- Solutions & Trade-Offs:  
	
		- Hybrid Models (Re-ranking with Fairness Constraints) → Balances organic vs. paid but hard to tune for revenue goals.  
		- Trust-Based Ranking (Reducing deceptive sellers, fake reviews) → Improves satisfaction but may lower short-term sales.  

Video & Music Streaming
------------------------------------------------------------------------------------
- Content-Length Bias in Recommendations  

	- Recommendation models often favor shorter videos (TikTok, YouTube Shorts) over long-form content.  
	- Example: YouTube’s watch-time optimization may prioritize clickbaity short videos over educational content.  
	- Solutions & Trade-Offs:  
	
		- Normalized Engagement Metrics (Watch Percentage vs. Watch Time) → Improves long-form content exposure but may reduce video diversity.  
		- Hybrid-Length Recommendations (Mixing Shorts & Full Videos) → Enhances variety but harder to rank effectively. 
************************************************************************************
Personalisation
************************************************************************************

************************************************************************************
Diversity
************************************************************************************
- Goal

	- improving user engagement
	- avoiding filter bubbles
	- preventing over-reliance on popular content.
- Metric

	- TODO
- Ensuring diversity in recommendation systems requires a multi-stage approach, balancing user engagement, fairness, and exploration. The best strategies depend on the product type:

	.. important::
		- Music & video platforms (Spotify, YouTube, TikTok) use DPP and Bandits to introduce diverse content.
		- E-commerce (Amazon, Etsy) balances popularity-based downsampling with weighted re-ranking.
		- Newsfeeds (Google News, Facebook, Twitter) use category-sensitive filtering to prevent echo chambers.

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
====================================================================================
.. note::
	Goal: Ensuring Diversity in Candidate Selection

Multi-Pool Retrieval (Heterogeneous Candidate Selection)
------------------------------------------------------------------------------------
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
------------------------------------------------------------------------------------
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
------------------------------------------------------------------------------------
	#. Query Expansion for Better Recall  
	
		- LLMs generate query variations to retrieve diverse candidates beyond exact keyword matching.  
		- Example: Instead of just retrieving laptops, LLMs expand queries to include notebooks, MacBooks, ultrabooks.  
		- Technique: Use T5/BERT-based semantic expansion to increase retrieval diversity.  
		
	#. Multi-Modal Understanding for Heterogeneous Retrieval  
	
		- LLMs bridge different modalities (text, image, video) to retrieve richer candidate pools.  
		- Example: In YouTube Recommendations, an LLM can link a users watched TED Talk to blog articles on the same topic.  
		- Technique: Use CLIP (for text-image-video embeddings) to retrieve across modalities.  
	
	#. User Preference Understanding for Contextual Retrieval  
	
		- Instead of static retrieval models, LLMs generate dynamic search queries based on user conversation history.  
		- Example: A user searching for travel backpacks may also receive recommendations for hiking gear if LLMs infer the intent.  
		- Technique: Use GPT-like models to rewrite user queries dynamically based on session context.  
	
Pros:  

	- Improves Recall - LLMs retrieve more diverse content that traditional CF models miss.  
	- Better Cold-Start Handling - Generates synthetic preferences for new users.  

Cons:  

	- High Latency - Generating queries dynamically can be slower than precomputed embeddings.  
	- Loss of Precision - More diverse candidates mean a higher risk of retrieving irrelevant results.  

Filtering & Merging Stage
====================================================================================
.. note::
	Goal: Balancing Diversity Before Re-Ranking

Minimum-Item Representation Heuristics
------------------------------------------------------------------------------------
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
------------------------------------------------------------------------------------
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
------------------------------------------------------------------------------------
	#. Semantic Deduplication & Cluster Merging  
	
		- LLMs identify semantically similar items (even if they differ in wording) to prevent redundancy.  
		- Example: In news recommendations, LLMs group articles covering the same event to avoid repetition.  
		- Technique: Use sentence embeddings (SBERT) to cluster semantically duplicate items.  
	
	#. Bias & Fairness Control  
	
		- LLMs detect biased patterns (e.g., over-representing a certain demographic) and adjust recommendations accordingly.  
		- Example: A job recommendation system might over-recommend tech jobs to menLLMs can balance exposure.  
		- Technique: Use LLM-based fairness models (e.g., DebiasBERT) to adjust recommendations.  
	
	#. Context-Aware Filtering  
	
		- LLMs generate filtering rules on-the-fly based on user profile, session history, or external trends.  
		- Example: If a user browses vegetarian recipes, LLMs downrank meat-based recipes dynamically.  
		- Technique: Use GPT-powered filtering prompts to dynamically adjust content selection.  

Pros:  

	- Prevents Repetitive Recommendations - Ensures users dont see redundant items.  
	- Improves Fairness & Representation - Adjusts for bias in candidate selection.  

Cons:  

	- Computationally Expensive - Filtering millions of candidates using LLMs can increase inference costs.  
	- Difficult to Fine-Tune - Over-filtering may hide relevant recommendations.  

Re-Ranking Stage
====================================================================================
.. note::
	Goal: Final Diversity Adjustments

Determinantal Point Processes (DPP)
------------------------------------------------------------------------------------
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
------------------------------------------------------------------------------------
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
------------------------------------------------------------------------------------
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
------------------------------------------------------------------------------------
	#. Diversity-Aware Ranking Models  

		- LLMs act as personalization-aware rerankers, balancing relevance with diversity dynamically.  
		- Example: Instead of showing only Marvel movies to a fan, LLMs inject DC movies or indie superhero films.  
		- Technique: Use LLM-powered diversity re-ranking prompts in post-processing.  
	
	#. Personalized Exploration vs. Exploitation  

		- LLMs simulate user preferences in real-time and adjust ranking to include more exploration.  
		- Example: In TikTok, if a user likes cooking videos, LLMs inject some fitness or travel videos to encourage exploration.  
		- Technique: Use GPT-powered bandit re-ranking for adaptive diversity balancing.  
	
	#. Diversity-Aware Re-Ranking via Counterfactual Predictions  

		- LLMs generate counterfactual recommendations to test how users might respond to different recommendation lists.  
		- Example: Instead of showing only trending news, LLMs inject underrepresented topics and measure user responses.  
		- Technique: Use LLMs for offline counterfactual testing before deployment.  

Pros:  

	- Balances Personalization & Diversity - Prevents filter bubbles.  
	- Improves Long-Term Engagement - Users are less likely to get bored.  

Cons:  

	- Higher Inference Cost - Re-ranking every session in real-time increases server load.  
	- Risk of Over-Exploration - If diversity is forced, users may feel the system is less relevant.  

************************************************************************************
Domain Knowledge
************************************************************************************
Sponsored Search
====================================================================================
Relevance vs. Revenue Trade-Off
------------------------------------------------------------------------------------
Why It Matters:  

	- Advertisers bid for visibility, but their ads may not always be relevant to the user's query.  
	- If high-bid but low-relevance ads are shown, users may lose trust in the search engine.  

Strategic Solutions & Trade-Offs:  

	- Quality Score (Google Ads' Approach)  Ranks ads based on a combination of CTR, relevance, and landing page experience, not just bid amount.  
	- Hybrid Ranking Model (Revenue + User Engagement)  Balances ad revenue vs. user satisfaction.  

Trade-Offs:  

	- Prioritizing high-relevance, low-bid ads reduces short-term revenue.  
	- Prioritizing high-bid, low-relevance ads hurts user trust & long-term retention.  

Click Spam & Ad Fraud
------------------------------------------------------------------------------------
Why It Matters:  

	- Bots & malicious actors inflate clicks to waste competitor ad budgets (click fraud).  
	- Some advertisers run low-quality, misleading ads to generate fake engagement.  

Strategic Solutions & Trade-Offs:  

	- Click Fraud Detection (Googles Invalid Click Detection)  Uses IP tracking, anomaly detection, and ML models to filter fraudulent clicks.  
	- Post-Click Analysis (User Behavior Analysis)  Detects bots based on engagement (bounce rate, session length, interactions).  

Trade-Offs:  

	- False Positives  May block legitimate traffic, harming advertisers.  
	- False Negatives  Fraudulent clicks still get monetized, increasing costs for real advertisers.  

Ad Auction Manipulation & Bid Shading
------------------------------------------------------------------------------------
Why It Matters:  

	- Sophisticated advertisers use AI-driven bidding strategies to game real-time auctions.  
	- Bid shading techniques lower ad costs while maintaining high visibility.  

Strategic Solutions & Trade-Offs:  

	- Second-Price Auctions (Vickrey Auctions)  Advertisers only pay the second-highest bid price, reducing manipulation.  
	- Multi-Objective Bidding Models  Balances advertiser cost efficiency and search engine revenue.  

Trade-Offs:  

	- Too much bid control reduces revenue  Search engines may earn less per click.  
	- Aggressive bid adjustments can reduce advertiser trust  If advertisers feel theyre losing transparency, they may pull budgets.  

Ad Fatigue & Banner Blindness
------------------------------------------------------------------------------------
Why It Matters:  

	- Users ignore repetitive ads after multiple exposures, reducing CTR over time.  
	- If ads look too much like organic results, users may feel deceived.  

Strategic Solutions & Trade-Offs:  

	- Adaptive Ad Rotation (Google Ads Optimize for Best Performing Mode)  Dynamically swaps low-performing ads with higher-engagement creatives.  
	- Ad Labeling Transparency  Clearer Sponsored tags improve user trust but reduce click rates.  

Trade-Offs:  

	- Refreshing ads too frequently raises advertiser costs.  
	- Too much ad transparency leads to lower revenue per impression.  

Query Intent Mismatch
------------------------------------------------------------------------------------
Why It Matters:  

	- Search queries are often ambiguous, and poor ad matching leads to bad user experience.  
	- Example: Searching for Apple  Should the search engine show Apple iPhones (commercial intent) or apple fruit (informational intent)?  

Strategic Solutions & Trade-Offs:  

	- Intent Classification Models (BERT, T5-based Models)  Classify queries into commercial vs. informational intent.  
	- Negative Keyword Targeting (Google Ads' Negative Keywords)  Advertisers block unrelated queries from triggering their ads.  

Trade-Offs:  

	- Restricting ads based on intent can lower revenue.  
	- Allowing broad ad targeting risks user dissatisfaction.  

Landing Page Experience & Conversion Rate Optimization
------------------------------------------------------------------------------------
Why It Matters:  

	- Even if an ad gets high CTR, if the landing page is misleading or slow, users bounce without converting.  
	- Google penalizes low-quality landing pages via Quality Score reductions.  

Strategic Solutions & Trade-Offs:  

	- Landing Page Quality Audits (Googles Ad Quality Guidelines)  Checks for page speed, relevance, mobile-friendliness.  
	- Post-Click Engagement Monitoring  Uses bounce rate, time-on-site, conversion tracking to refine ranking.  

Trade-Offs:  

	- Strict landing page rules limit advertiser flexibility.  
	- Relaxed rules allow low-quality ads, reducing long-term trust.  

Multi-Touch Attribution & Ad Budget Allocation
------------------------------------------------------------------------------------
Why It Matters:  

	- Users may see an ad but not convert immediately  Traditional last-click attribution ignores earlier touchpoints.  
	- Advertisers struggle to allocate budgets across search, display, social, and video ads.  

Strategic Solutions & Trade-Offs:  

	- Multi-Touch Attribution Models (Shapley Value, Markov Chains)  Assigns fair credit to different ad exposures.  
	- Cross-Channel Conversion Tracking  Tracks user journeys across search & display ads.  

Trade-Offs:  

	- More complex attribution models require longer training times.  
	- Over-attributing upper-funnel ads can inflate costs without clear ROI.  

Fairness & Ad Bias Issues
------------------------------------------------------------------------------------
Why It Matters:  

	- Some ad auctions are biased against small advertisers, favoring large ad budgets.  
	- Discriminatory ad targeting (e.g., gender/race bias in job/housing ads) can lead to regulatory penalties.  

Strategic Solutions & Trade-Offs:  

	- Fairness-Constrained Bidding (Googles Fairness-Aware Ad Auctions)  Adjusts auction weights to prevent dominance by large advertisers.  
	- Bias Detection in Ad Targeting (Auditing Models for Discriminatory Targeting)  Ensures fair exposure of diverse ads.  

Trade-Offs:  

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
