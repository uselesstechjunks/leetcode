##########################################################################
Semantic Document Understanding
##########################################################################
.. contents:: Table of Contents
	:depth: 2
	:local:
	:backlinks: none

**************************************************************************
Application
**************************************************************************
- Content enhancement using pretrained transformers + metadata
- Multimodal fusion of user-uploaded image + free-text
- Categorization using LLMs or BERT-classifiers
- Cross-listing product linking across Facebook, Instagram shops, and Marketplace
- Title rewrites or suggestion systems
- Policy enforcement models using understanding of item type + suspicious language

**************************************************************************
Task
**************************************************************************
It involves extracting structured meaning from unstructured or semi-structured data like:

	- Titles
	- Descriptions
	- Tags
	- Images
	- Videos
	- Metadata (price, category, brand)
	- Other multimodal cues (e.g., seller info, location, posting time)

**************************************************************************
Resources
**************************************************************************
1. Product Classification & Attribute Extraction
==========================================================================
- Goal: Categorize products, extract attributes like brand, color, material, condition, etc.
- Papers:

	- [aaai.org] `Is a Picture Worth a Thousand Words? A Deep Multi-Modal Architecture for Product Classification in E-Commerce <https://ojs.aaai.org/index.php/AAAI/article/download/11419/11278>`_
- Techniques:

	- BERT/XLM-R based encoders over title + description
	- Sequence tagging (CRF, LSTM-CRF) or span extraction for attributes
	- Denoising Autoencoder or MLM pretraining on titles + descriptions

2. Multimodal Product Representation
==========================================================================
- Goal: Fuse visual and textual signals to get high-quality item embeddings.
- Papers:

	- [ieee.org] `Deep Multimodal Representation Learning: A Survey <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8715409>`_
	- [openaccess.thecvf.com] `Learning Instance-Level Representation for Large-Scale Multi-Modal Pretraining in E-commerce <https://openaccess.thecvf.com/content/CVPR2023/papers/Jin_Learning_Instance-Level_Representation_for_Large-Scale_Multi-Modal_Pretraining_in_E-Commerce_CVPR_2023_paper.pdf>`_
	- [amazon.science] `Unsupervised Multi-Modal Representation Learning for High Quality Retrieval of Similar Products at E-commerce Scale <https://assets.amazon.science/54/5e/df0e19f94b26afb451dd2c156612/unsupervised-multi-modal-representation-learning-for-high-quality-retrieval-of-similar-products-at-e-commerce-scale.pdf>`_
- Techniques:

	- Image encoder (e.g., ResNet, CLIP) + text encoder (BERT)
	- Multimodal Fusion: concatenation, attention-based fusion, co-attention networks
	- Training objective: classification, contrastive learning (CLIP-style)

3. Product Title Normalization & Rewriting
==========================================================================
- Goal: Rewrite cluttered or inconsistent product titles for better standardization and retrieval.
- Papers:

	- https://paperswithcode.com/task/attribute-value-extraction
- Methods:

	- Encoder-decoder (BART, T5)
	- Post-processing with rule-based constraints

4. Product Deduplication and Matching
==========================================================================
- Goal: Identify duplicate listings across users or platforms (e.g., same product uploaded multiple times).
- Papers:

	- [arxiv.org] `Deep Product Matching for E-commerce Search <https://arxiv.org/abs/1806.06159>`_
	- [arxiv.org] `Multi-modal Product Retrieval in Large-scale E-commerce <https://arxiv.org/abs/2011.09566>`_
- Methods:

	- Siamese Networks, contrastive learning
	- Title+image fusion
	- Use of embedding similarity or learned matching functions

5. Taxonomy Mapping & Enhancement
==========================================================================
- Goal: Map user-uploaded listings to structured product taxonomy or enhance weak labels.
- Resources:

	- [arxiv.org] `Semantic Enrichment of E-commerce Taxonomies <https://arxiv.org/abs/2102.05806>`_
	- [arxiv.org] `TaxoEmbed: Product Categorization with Taxonomy-Aware Label Embedding <https://arxiv.org/abs/2010.12862>`_
- Methods:

	- Label embedding
	- Graph neural networks (if taxonomy structure is hierarchical)
