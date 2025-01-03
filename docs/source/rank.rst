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

************************************************************************************
Personalised homepage recommendations 
************************************************************************************
RecSys learning summary (stanford graph NN lecture):

- mapped as an edge prediction problem in a bipartite graph

Ranking
====================================================================================
- Metric Recall@k (non differentiable)
- Other metrics: HR@k, nDCG
- Differentiable Discriminative loss (binary loss (similar to cross entropy), Bayesian prediction loss (BPR)
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

************************************************************************************
Session/sequential RS
************************************************************************************
- Attention based
- Transformer4rec

************************************************************************************
LLM for Recommendation
************************************************************************************
- gotta read
