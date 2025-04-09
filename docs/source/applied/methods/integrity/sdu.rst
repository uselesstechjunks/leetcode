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

TODO - fix later

	- | Tool / Model | Use Case | Notes |
	- | mT5 / BART / LLaMA | Text generation & rewriting | Use for product title/desc enhancement |
	- | CLIP / BLIP / ResNet | Visual embeddings | Combine with title/desc in multimodal fusion |
	- | XLM-R / DistilBERT | Multilingual title/desc encoding | Works well in low-resource or multilingual markets |
	- | FAISS | Deduplication, similarity search | For embedding-based matching |
	- | Spacy + Rule-based | Attribute extraction | Hybrid approach in noisy settings |

**************************************************************************
Resources
**************************************************************************
- Multi Modal models

	- [encord.com] `Top 10 Multimodal Models <https://encord.com/blog/top-multimodal-models/>`_
- Vision-text encoder:

	- [medium.com] `Understanding OpenAI’s CLIP model <https://medium.com/@paluchasz/understanding-openais-clip-model-6b52bade3fa3>`_
	- [amazon.science] `KG-FLIP: Knowledge-guided Fashion-domain Language-Image Pre-training for E-commerce <https://assets.amazon.science/fb/63/9b81471c4b46bad6bd1cbcb591bc/kg-flip-knowledge-guided-fashion-domain-language-image-pre-training-for-e-commerce.pdf>`_
	- [amazon.science] `Unsupervised multi-modal representation learning for high quality retrieval of similar products at e-commerce scale <https://www.amazon.science/publications/unsupervised-multi-modal-representation-learning-for-high-quality-retrieval-of-similar-products-at-e-commerce-scale>`_
- Vision-encoder text-decoder:

	- [amazon.science] `MMT4: Multi modality to text transfer transformer <https://www.amazon.science/publications/mmt4-multi-modality-to-text-transfer-transformer>`_
	- [research.google] `MaMMUT: A simple vision-encoder text-decoder architecture for multimodal tasks <https://research.google/blog/mammut-a-simple-vision-encoder-text-decoder-architecture-for-multimodal-tasks/>`_
	- [medium.com] `Understanding DeepMind’s Flamingo Visual Language Models <https://medium.com/@paluchasz/understanding-flamingo-visual-language-models-bea5eeb05268>`_
- E-commerce publications

	- [amazon.science] `Amazon Science e-Commerce <https://www.amazon.science/publications?q=&f1=0000017b-cb9b-d0be-affb-cbbf08e40000&s=0>`_

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
Product Categorisation
==========================================================================
- Goal: Assign each product to one of a fixed set of predefined categories (e.g., "sofa", "headphones", "jacket").
- Formulation:

	- Single-label classification
	- Classes are mutually exclusive
	- Typically ~10s to 1,000s of classes
	- Output: a single class label
	- Must map to structured taxonomy (e.g., for catalog consistency)
- Use-case:

	- Structured Browsing & Filtering
	- Search Relevance & Ranking
	- Inventory Organization & Duplication Handling
	- Recommendation Relevance
	- Content Moderation Routing
- Resources:

	- [arxiv.org] `Semantic Enrichment of E-commerce Taxonomies <https://arxiv.org/abs/2102.05806>`_
	- [arxiv.org] `TaxoEmbed: Product Categorization with Taxonomy-Aware Label Embedding <https://arxiv.org/abs/2010.12862>`_
- Methods:

	- Label embedding
	- Graph neural networks (if taxonomy structure is hierarchical)

Dynamic Product Tag Suggestion
==========================================================================
- Goal: Suggest a set of relevant tags (e.g., "leather", "portable", "Bluetooth", "red", "minimalist") to describe a product.
- Formulation:

	- Multi-label classification or tag ranking
	- Tags are not mutually exclusive
	- Tags can be from a dynamic or evolving vocabulary
	- Output: list of top-k tags, optionally with confidence scores
- Use-case:

	- Search Recall Expansion
	- Visual Attribute Search
	- Recommendation Diversification
	- Ad Targeting / Sponsored Listings
	- Content Moderation & Policy Enforcement
	- Seller Assistance / Listing Enhancement

Multimodal Product Representation
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

Product Title Normalization & Rewriting
==========================================================================
- Goal: Rewrite cluttered or inconsistent product titles for better standardization and retrieval.
- Papers:

	- https://paperswithcode.com/task/attribute-value-extraction
- Methods:

	- Encoder-decoder (BART, T5)
	- Post-processing with rule-based constraints

Product Deduplication and Matching
==========================================================================
- Goal: Identify duplicate listings across users or platforms (e.g., same product uploaded multiple times).
- Papers:

	- [arxiv.org] `Deep Product Matching for E-commerce Search <https://arxiv.org/abs/1806.06159>`_
	- [arxiv.org] `Multi-modal Product Retrieval in Large-scale E-commerce <https://arxiv.org/abs/2011.09566>`_
- Methods:

	- Siamese Networks, contrastive learning
	- Title+image fusion
	- Use of embedding similarity or learned matching functions

**************************************************************************
Examples
**************************************************************************
1. Product Categorisation - Image only
==========================================================================
Case A: 100k labeled examples + 1M unlabeled
--------------------------------------------------------------------------
1. Pretraining:
	- Use pretrained ResNet or ViT (ImageNet) as base.
	- Optionally run domain-adaptive pretraining on 1M unlabeled images using SimCLR/DINO.

2. Finetuning:
	- Replace classification head with new head (1,000 classes).
	- Finetune full model on 100k labeled samples with label smoothing, strong augmentation, and class balancing.
	- Use early unfreezing strategy if pretrained on different domain.

3. Regularization:
	- Mixup, CutMix, RandAugment.
	- Confidence-based pseudo-labeling on 1M unlabeled to expand training data.

4. Evaluation:
	- Accuracy@1, Accuracy@5.
	- Confusion matrix to analyze inter-class errors.

Case B: Only 10k labeled examples
--------------------------------------------------------------------------
1. Pretraining:
	- Use stronger pretrained backbone (e.g., ViT MAE pretrained on ImageNet-21k or OpenImages).
	- Optionally pretrain on 1M unlabeled data (SimCLR, SwAV, DINO).

2. Finetuning:
	- Use **linear probing** first (freeze encoder, train classifier only).
	- Then **gradually unfreeze** layers (e.g., using discriminative learning rates).
	- Regularize with dropout, weight decay, and Mixup.

3. Semi-supervised:
	- Train pseudo-labeling pipeline on 1M unlabeled images using high-confidence predictions.

4. Evaluation:
	- Macro/micro F1-score (especially if classes are imbalanced).

1. Image Search - Mobile Image Query - Product Image
==========================================================================
Case A: 10k (query, matched product) labeled pairs
--------------------------------------------------------------------------
1. Pretraining:
	- Pretrain ResNet/ViT using SimCLR or DINO on product images with augmentations.
	- Learn product-invariant and view-invariant embeddings.

2. Finetuning:
	- Use InfoNCE contrastive loss on 10k query-product pairs.
	- Use in-batch negatives and/or hard mined negatives.
	- Augment with product image pairs to regularize.

3. Embedding Aggregation:
	- Per product: average of embeddings of its 5 images.
	- Optional: trainable attention-based image pooling per product.

4. Retrieval:
	- Use Faiss or ScaNN for approximate nearest neighbor search.
	- Index product embeddings offline; query embeddings at runtime.

5. Evaluation:
	- Recall@k, mean Average Precision (mAP), Precision@k.

Case 2: 200M unlabeled mobile images (no labels)
--------------------------------------------------------------------------
1. Pretraining:
	- Use DINO, MAE, or SimCLR on 200M mobile photos to learn domain-aligned embeddings.
	- Incorporate augmentations reflecting phone capture artifacts (blur, shadow, exposure).

2. Semi-Supervised Learning:
	- Cluster mobile images; use nearest neighbors as pseudo-positives.
	- Use self-training with high-confidence retrieval matches as additional positives.

3. Hard Negatives:
	- Select visually similar images that do *not* match (via clustering or retrieval) as hard negatives.

4. Finetuning (optional):
	- Finetune on the small labeled query-product dataset (10k), possibly using LoRA or head-only tuning.

5. Retrieval + Evaluation:
	- Same as Case 1.
	- Test generalization on held-out queries and unseen product classes.
