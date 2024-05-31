#########################################################################################
Natural Language Processing
#########################################################################################
.. warning::
	Goal: Write summary of key ideas and summary of papers

*****************************************************************************************
Tokenizers
*****************************************************************************************
WordPiece
=========================================================================================
SentencePiece
=========================================================================================
Byte-Pair Encoding (BPE)
=========================================================================================

*****************************************************************************************
Word Embeddings
*****************************************************************************************
.. note::
	* Word2Vec: Efficient Estimation of Word Representations in Vector Space
	* GloVe: Global Vectors forWord Representation
	* Evaluation methods for unsupervised word embeddings

*****************************************************************************************
Attention
*****************************************************************************************
.. note::
	* `[jalammar.github.io] The Illustrated Transformer <https://jalammar.github.io/illustrated-transformer/>`_
	* `[lilianweng.github.io] Attention? Attention! <https://lilianweng.github.io/posts/2018-06-24-attention/>`_
	* Attention Is All You Need

*****************************************************************************************
Architecture
*****************************************************************************************
RNN
=========================================================================================
.. note::
	* On the diculty of training Recurrent Neural Networks
	* Sequence to Sequence Learning with Neural Networks
	* NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE

Encoder [BERT]
=========================================================================================
.. note::
	* BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
	* A Primer in BERTology: What we know about how BERT works
	* RoBERTa: A Robustly Optimized BERT Pretraining Approach
	* XLM: Cross-lingual Language Model Pretraining
	* XLM-R: Unsupervised Cross-lingual Representation Learning at Scale
	* TwinBERT: Distilling Knowledge to Twin-Structured BERT Models for Eicient Retrieval

Decoder [GPT]
=========================================================================================
.. note::
	* `[jalammar.github.io] The Illustrated GPT-2 <https://jalammar.github.io/illustrated-gpt2/>`_
	* GPT-2: Language Models are Unsupervised Multitask Learners
	* GPT-3: Language Models are Few-Shot Learners

Encoder-Decoder [T5]
=========================================================================================
.. note::
	* T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
	* mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer

*****************************************************************************************
Training
*****************************************************************************************
Pretraining
=========================================================================================
Domain-Adaptation
=========================================================================================
Fine-Tuning
=========================================================================================
Choice of Loss Function
-----------------------------------------------------------------------------------------
Cross-Entropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Contrastive Loss
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*****************************************************************************************
Special Techniques
*****************************************************************************************
Low-Rank Approximations (LoRA)
=========================================================================================
Reinforcement Learning with Human Feedback (RLHF)
=========================================================================================

*****************************************************************************************
Task Specific Setup
*****************************************************************************************
Text Generation
=========================================================================================
.. note::
	* `[mlabonne.github.io] Decoding Strategies in Large Language Models <https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html>`_

Text Classification
=========================================================================================
Finding Similar Items
=========================================================================================
Approximate Nearest Neighbour Search [DiskANN]
-----------------------------------------------------------------------------------------

*****************************************************************************************
Conversational Models
*****************************************************************************************
Prompt Engineering
=========================================================================================
Prompt Tuning
=========================================================================================

Resources
=========================================================================================
.. note::
	* `[HN] You probably don’t need to fine-tune an LLM <https://news.ycombinator.com/item?id=37174850>`_
	* `[Ask HN] Most efficient way to fine-tune an LLM in 2024? <https://news.ycombinator.com/item?id=39934480>`_
	* `[HN] Finetuning Large Language Models <https://news.ycombinator.com/item?id=35666201>`_

		* `[magazine.sebastianraschka.com] Finetuning Large Language Models <https://magazine.sebastianraschka.com/p/finetuning-large-language-models>`_
	* `[Github] LLM Course <https://github.com/mlabonne/llm-course>`_
