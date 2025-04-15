#################################################################################
Commerce
#################################################################################
.. image:: ../../img/marketplace.png
	:width: 600
	:alt: Framework

.. contents:: Table of Contents
	:depth: 2
	:local:
	:backlinks: none

*********************************************************************************
Domain Understanding
*********************************************************************************
Listings
=================================================================================
.. csv-table::
	:header: "Attribute", "Sub-attribute", "Examples", "Characteristics"
	:widths: 16 12 12 32
	:align: center
	
		1. Title/Desc, , , uninformative; misleading; spelling/grammar errors
		2. Images, , , low quality
		3. Location, , ,
		, Postal, , user provided -> low coverage; incorrect
		, Lat-Long, , gps inferred -> high coverage; incorrect; upload location might be different than product availability
		4. Price, , , incorrect; misleading/scam
		5. Category, , , mostly missing; possibly incorrect
		6. Tags, , , category dependent; mostly missing; possibly incorrect
		, 1. Attributes, colour; size,
		, 2. Condition, new; refurbished, 
		, 3. Style, minimalistic; vintage; casual,
		, 4. Use-case, gift-ideas; travel friendly,
		, 5. Occasion, wedding; office; gym,
		, 6. Catchphrases, huge discount, open-ended; clickbaity
*********************************************************************************
Product Understanding
*********************************************************************************
Taxonomy classification
=================================================================================
Attribute extraction
=================================================================================
Entity linking
=================================================================================
*********************************************************************************
Product Quality & Integrity
*********************************************************************************
Duplicate detection
=================================================================================
Moderation
=================================================================================
*********************************************************************************
Product Search
*********************************************************************************
1. use-case
	1. system: 
		- text queries
		- system returns a list of listings
		- sorted in decreasing order of relevance
		- filtered by geolocation
		- [x] personalisation
		- [x] contextualisation
		- available across different surfaces
	2. actions (users)
		- click -> product details page -> contact seller
		- scroll past
	2. actors:
		- users, sellers, platform
	3. interests:
		- users: find most relevant results
		- sellers: increase coverage of their listings
		- platform:
			- [x] quality: results should not contain listings that violate policies
			- user engagement: 
2. business kpis
	- CTR, coverage, QBR, DwellTime
3. scale
	- 1M sellers, 50M listings, 1M/day new listings
	- 1B users, 95% on mobile device, low latency req
4. signals
	- search logs (noisy)
	- baseline - kw search
5. misc
	- surrogate systems: 
		- kw extraction
		- taxonomy inference
		- attribute extraction
		- dynamic tag suggestion
		- moderation
		- query understanding

Keyword search
=================================================================================
Semantic search
=================================================================================
*********************************************************************************
Product Recommendation
*********************************************************************************
Similar listings recommendation
=================================================================================
Homepage recommendation
=================================================================================
