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

Handling Nuisances & Trade-offs  
------------------------------------------------------------------------------------
#. Cold-Start Problem  

	- New queries: Use query expansion via NLP techniques (e.g., synonym mining).  
	- New items: Use embeddings to map items to semantically similar ones.  
	- Trade-off: Over-expansion can retrieve noisy or irrelevant results.  

#. Novelty Effects  

	- Promote fresh items by boosting recency in ranking models.  
	- Trade-off: Too much emphasis on new content may degrade long-term user satisfaction.  

#. Popularity Bias  

	- Penalize over-recommended items in search results.  
	- Introduce personalized ranking adjustments per user profile.  
	- Trade-off: Too much de-biasing may harm engagement rates.  

#. Explore-Exploit Balance  

	- Contextual bandits for dynamic ranking adjustments.  
	- Randomized exploration in search results (e.g., diverse result sets).  
	- Trade-off: Excessive exploration may reduce precision and user satisfaction.  

#. Avoiding Feedback Loops  

	- Regularly update embeddings and ranking models.  
	- Use counterfactual learning to estimate impact of unseen queries.  
	- Trade-off: More frequent retraining requires higher computational cost.  

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

Handling Nuisances & Trade-offs  
------------------------------------------------------------------------------------
#. Cold-Start Problem  

	- New items: Use content-based methods (text, image embeddings).  
	- Few interactions: Boost exploration via diversity in recommendations.  
	- Trade-off: Content-based methods may miss collaborative signals.  

#. Novelty Effects 
 
	- Boost engagement by temporarily surfacing new items.  
	- Trade-off: Over-promoting new items can degrade relevance.  

#. Popularity Bias  

	- Downweight extremely popular items to avoid over-recommendation.  
	- Diversify item recommendations per user segment.  
	- Trade-off: Reducing popular items too much may lower engagement.  

#. Explore-Exploit Balance  

	- Use Thompson Sampling or contextual bandits.  
	- Introduce diversity via re-ranking strategies.  
	- Trade-off: Too much exploration may hurt conversion rates.  

#. Avoiding Feedback Loops  

	- Periodically refresh item similarities to prevent stale recommendations.  
	- Use counterfactual evaluations to assess if users are stuck in recommendation bubbles.  
	- Trade-off: More frequent updates increase computational complexity.  

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

Handling Nuisances & Trade-offs  
------------------------------------------------------------------------------------
#. Cold-Start Problem  

	- New users: Use demographic-based recommendations or onboarding surveys.  
	- New items: Use content-based embeddings, metadata-based recommendations.  
	- Trade-off: Over-relying on metadata may lead to popularity bias.  
#. Novelty Effects  

	- Boosting exploration by temporarily ranking new items higher.  
	- Trade-off: Can over-promote untested items, leading to poor user experience.  
#. Popularity Bias  

	- Penalizing highly popular items in ranking.  
	- Personalized diversity re-ranking (e.g., promoting niche items based on user profile).  
	- Trade-off: Too much de-biasing can hurt engagement.  
#. Explore-Exploit Balance  

	- Bandit-based approaches (e.g., Thompson Sampling, UCB).  
	- Randomized ranking perturbation to introduce diversity.  
	- Trade-off: Excess exploration can hurt short-term metrics.  
#. Avoiding Feedback Loops  

	- Regular model updates to prevent reinforcing the same recommendations.  
	- Counterfactual evaluation (simulate what would happen if different recommendations were shown).  
	- Trade-off: More frequent model updates increase computational costs.  

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

Handling Nuisances & Trade-offs 
------------------------------------------------------------------------------------ 
#. Cold-Start Problem  

	- New sessions: Default to trending/popular items.  
	- New items: Leverage content-based recommendations within sessions.  
	- Trade-off: May fail to capture truly personalized intent.  

#. Novelty Effects  

	- Boost recent items in ranking to reflect dynamic preferences.  
	- Trade-off: Overweighting recent activity may hurt recommendation diversity.  

#. Popularity Bias  

	- Adjust session-based rankings to include niche or long-tail items.  
	- Trade-off: Too much diversity can reduce perceived relevance.  

#. Explore-Exploit Balance  

	- Reinforcement learning to adaptively explore new items in a session.  
	- Trade-off: Over-exploration can degrade short-term session relevance.  

#. Avoiding Feedback Loops  

	- Periodically reset session-based learning to avoid stale recommendations.  
	- Trade-off: Frequent resets may cause loss of valuable session insights.  

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

Handling Nuisances & Trade-offs  
------------------------------------------------------------------------------------
#. Cold-Start Problem  

	- New users: Recommend based on shared attributes (location, interests).  
	- New connections: Use heuristic-based similarity scores.  
	- Trade-off: Can lead to homophily (over-recommending similar users).  

#. Novelty Effects  

	- Boost recommendations of new users to help them integrate faster.  
	- Trade-off: High visibility to new users might hurt engagement from established users.  

#. Popularity Bias  

	- Penalize highly connected users in ranking.  
	- Use diversity-enhancing ranking strategies.  
	- Trade-off: Reducing bias may lower connection acceptance rates.  

#. Explore-Exploit Balance  

	- Multi-Armed Bandit strategies (exploring less-connected users).  
	- Randomized ranking variations to encourage serendipity.  
	- Trade-off: Excessive exploration can degrade user experience.  

#. Avoiding Feedback Loops  

	- Use randomized A/B testing to prevent algorithmic bias reinforcement.  
	- Regularly refresh graph-based embeddings.  
	- Trade-off: Frequent updates increase computational costs.  

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
Diversity in Recommendation Systems
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
