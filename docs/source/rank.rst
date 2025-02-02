####################################################################################
Ranking & Recommendation
####################################################################################
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
Papers
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

Videos
------------------------------------------------------------------------------------
- [youtube.com] `Stanford CS224W: Machine Learning w/ Graphs I 2023 I GNNs for Recommender Systems <https://www.youtube.com/watch?v=OV2VUApLUio>`_

	- Mapped as an edge prediction problem in a bipartite graph

Ranking
====================================================================================
- Metric Recall@k (non differentiable)
- Other metrics: HR@k, nDCG
- Differentiable Discriminative loss - binary loss (similar to cross entropy), Bayesian prediction loss (BPR)
- Issue with binary, BPR solves the ranking problem better
- Trick to choose neg samples
- Not suitable for ANN

Collaborative filtering
====================================================================================
- DNN to capture user item similarity with cosine or InfoNCE loss
- ANN friendly 
- Doesn't consider longer than 1 hop in the bipartite graph 

GCN
====================================================================================
- Smoothens the embeddings by GCN layer interactions using undirected edges to enforce similar user and similar item signals
- Neural GCN or LightGCN
- Application: similar image recommendation in Pinterest 
- Issue: doesn't have contextual awareness or session/temporal awareness

Session/sequential RS
====================================================================================
- Attention based
- Transformer4rec

LLM for Recommendation
====================================================================================
- gotta read

************************************************************************************
Patterns
************************************************************************************
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
	- "Transformers4Rec: Bridging the Gap Between NLP and Sequential Recommendation" – De Souza et al. (2021)  

#. Hybrid and Production Systems  

	- "Amazon.com Recommendations: Item-to-Item Collaborative Filtering" – Linden et al. (2003)  
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
