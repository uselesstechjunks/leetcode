########################################################################################
Search Engine
########################################################################################
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none
****************************************************************************************
Minimal Event Logs
****************************************************************************************
Problem: `This question from Leetcode discussion forum <https://leetcode.com/discuss/interview-experience/512591/google-machine-learning-engineer-bangalore-dec-2019-reject/911775>`_

.. note::
	- You are given user_id, timestamp, search_query and list of links clicked by user while doing google search.
	
		.. csv-table::
			:header: user_id, timestamp, search_query, list_of_links_clicked
			:align: center
			
				5844364346, 1355563265.81, 'hotels in london', [linkurl1; linkurl2; linkurl2]
				2657673352, 1355565575.36, 'flowers', [linkurl2]
				3686586523, 1355547455.81, 'insurance', []
	- Design a machine learning system which predicts which links to show based on search query.
========================================================================================
What To Predict?
========================================================================================
#. Pointwise Ranking using estimated CTR

	.. csv-table::
				:header: user_id, timestamp, search_query, link, ctr
				:align: center

					58443665466, 1355563675.81, 'hotels in paris', linkurl1, 0.35
#. Ranked List of K Links

	.. csv-table::
				:header: user_id, timestamp, search_query, ranked_links
				:align: center

					58443665466, 1355563675.81, 'hotels in paris', [linkurl1; linkurl4; linkurl6]
========================================================================================
What Matters?
========================================================================================
#. User

	- Historical click propensity of users (new user/old user)	
	- User's latest query propensity (activity)
	- User's latest click propensity (activity)
	- User's query propensity at a given time
	- User's click propensity at a given time
#. Query

	- Historical popularity of query
	- Historical click propensity for a query
	- Recent popularity of query (trending)
	- Recent click propensity for a query
	- ??
#. Link

	- Historical popularity of links
	- Recent popularity of links (trending)
	- ??
#. Cross

	- User's latest queries
	- User's affinity towards certain queries
	- User's affinity towards certain links
	- User's propensity towards certain queries at a given time
	- User's propensity towards certain links at a given time
	- ??
========================================================================================
What Can I Count?
========================================================================================
========================================================================================
What Would I Have During Inference Time?
========================================================================================
========================================================================================
How Do I Design Features?
========================================================================================
========================================================================================
How Do I Train Model?
========================================================================================
========================================================================================
How Do I Evaluate Model?
========================================================================================
========================================================================================
How Do I Debug Model?
========================================================================================
========================================================================================
How Do I Deploy Model?
========================================================================================
========================================================================================
How Do I Monitor Model?
========================================================================================
****************************************************************************************
Rich Event Logs
****************************************************************************************
****************************************************************************************
Content
****************************************************************************************
****************************************************************************************
Context
****************************************************************************************
