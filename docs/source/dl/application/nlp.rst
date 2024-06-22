#########################################################################################
Natural Language Processing
#########################################################################################
.. warning::
	Goal: Write summary of key ideas and summary of papers

*****************************************************************************************
Sequence Modeling
*****************************************************************************************
RNN
=========================================================================================
.. collapse:: RNN implementation in PyTorch

   .. literalinclude:: ../../code/rnn.py
      :language: python
      :linenos:

.. note::
	* On the diffculty of training Recurrent Neural Networks
	* Sequence to Sequence Learning with Neural Networks
	* NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE

LSTM
=========================================================================================
.. collapse:: LSTM implementation in PyTorch

   .. literalinclude:: ../../code/lstm.py
      :language: python
      :linenos:

.. note::
	* `StatQuest on LSTM <https://www.youtube.com/watch?v=YCzL96nL7j0>`_

*****************************************************************************************
Tokenizers
*****************************************************************************************
.. warning::
	* TODO

WordPiece
=========================================================================================
SentencePiece
-----------------------------------------------------------------------------------------
Byte-Pair Encoding (BPE)
=========================================================================================

Extending Vocab for Domain-Adaptation or Fine-Tuning
=========================================================================================
Problem Statement:
-----------------------------------------------------------------------------------------
I develop ranking and recommendation systems for my customers. I want to leverage an LLM to improve the performance of the ranking and recommendation systems. In particular, I am planning to use the embeddings from the LLM for my downstream tasks.

I am planning to take a pre-trained, publicly available LLM which is an autoregressive model, as in, it is pre-trained to predict the next token in a sequence given previous tokens in that sequence. I plan to adapt it for my specific domain by performing continuous training with the same pre-training objective as the original LLM. 

Here is the issue. The data that I work with contains a lot of domain-specific terms which might have no been seen by the original LLM's tokenizer (which uses byte-pair encoding tokenizer and is trained on publicly available datasets). Therefore, many of these domain-specific terms from my data would get assigned to a common UNKNOWN token and therefore, the embeddings for those terms would be useless for my downstream task.

Question (a) How would I incorporate my domain specific terms into the LLM's tokenizer vocabulary? How should I rescale the original LLM's input Embedding matrix to accomodate for these new tokens? 
Question (b) I want to keep the original token embeddings intact. For the new tokens that I'll add in this process, the model would learn embeddings from the end-to-end pretraining objective.

Solution:
-----------------------------------------------------------------------------------------
To incorporate domain-specific terms into the tokenizer vocabulary of a pre-trained autoregressive Language Model (LLM) and subsequently adjust the embedding matrix while preserving the original embeddings, you can follow these steps. Let's break it down:

1. Extend the Tokenizer Vocabulary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
First, you need to extend the tokenizer's vocabulary to include your domain-specific terms. Since you mentioned using a pre-trained LLM with a byte-pair encoding (BPE) tokenizer (e.g., GPT-3), you'll need to add your terms to this tokenizer.

.. code-block:: python

	from transformers import GPT2Tokenizer, GPT2Model
	
	# Load the pre-trained tokenizer and model
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
	
	# Example of extending vocabulary with domain-specific terms
	domain_specific_terms = ["term1", "term2", "term3"]
	tokenizer.add_tokens(domain_specific_terms)
	
	# If you are also fine-tuning the model, adjust the model to handle new tokens
	model = GPT2Model.from_pretrained('gpt2')
	model.resize_token_embeddings(len(tokenizer))

.. note::
	* tokenizer.add_tokens(domain_specific_terms): This adds your domain-specific terms to the tokenizer vocabulary.
	* model.resize_token_embeddings(len(tokenizer)): This adjusts the model's embedding layer to accommodate the new tokens. This step is crucial if you plan to fine-tune the model with these new tokens.

2. Tinkering with the Embedding Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Since you want to preserve the original token embeddings for continuous training and only allow the model to learn embeddings for the new tokens from scratch, you need to handle the embedding matrix carefully:

.. code-block:: python

	import torch
	
	# Load the original model again for clarity
	model = GPT2Model.from_pretrained('gpt2')
	
	# Assuming you have already added new tokens to the tokenizer
	new_token_ids = tokenizer.encode(domain_specific_terms, add_special_tokens=False)
	
	# Initialize the new token embeddings randomly
	new_token_embeddings = torch.randn(len(new_token_ids), model.config.hidden_size)
	
	# Concatenate original embeddings with new token embeddings
	original_embeddings = model.transformer.wte.weight[:tokenizer.vocab_size]
	combined_embeddings = torch.cat([original_embeddings, new_token_embeddings], dim=0)
	
	# Overwrite the original embedding matrix in the model
	model.transformer.wte.weight.data = combined_embeddings

.. note::
	* tokenizer.encode(domain_specific_terms, add_special_tokens=False): This encodes the domain-specific terms to get their token IDs in the tokenizer's vocabulary.
	* torch.randn(len(new_token_ids), model.config.hidden_size): This initializes random embeddings for new tokens. Alternatively, you can initialize them differently based on your specific needs.
	* model.transformer.wte.weight[:tokenizer.vocab_size]: Extracts the original embeddings up to the size of the original vocabulary.
	* torch.cat([original_embeddings, new_token_embeddings], dim=0): Concatenates the original embeddings with the new token embeddings.

Notes:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Tokenizer Vocabulary: Ensure that after extending the tokenizer vocabulary, you save it or use it consistently across your tasks.
* Embedding Adjustment: The approach here adds new tokens and initializes their embeddings separately from the pre-trained embeddings. This keeps the original embeddings intact while allowing new tokens to have their embeddings learned during fine-tuning.
* Fine-Tuning: If you plan to fine-tune the model on your specific tasks, you would then proceed with training using your domain-specific data, where the model will adapt not only to the new tokens but also to the specific patterns in your data.

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
Position Encoding
*****************************************************************************************
.. note::
	* Position Information in Transformers: An Overview
	* Rethinking Positional Encoding in Language Pre-training

*****************************************************************************************
Architecture
*****************************************************************************************
Encoder [BERT]
=========================================================================================
.. note::
	* BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
	* A Primer in BERTology: What we know about how BERT works
	* RoBERTa: A Robustly Optimized BERT Pretraining Approach
	* XLM: Cross-lingual Language Model Pretraining
	* TwinBERT: Distilling Knowledge to Twin-Structured BERT Models for Eicient Retrieval

Decoder [GPT]
=========================================================================================
.. note::
	* `[jalammar.github.io] The Illustrated GPT-2 <https://jalammar.github.io/illustrated-gpt2/>`_
	* `[cameronrwolfe.substack.com] Decoder-Only Transformers: The Workhorse of Generative LLMs <https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse>`_
	* GPT-2: Language Models are Unsupervised Multitask Learners
	* GPT-3: Language Models are Few-Shot Learners

Encoder-Decoder [T5]
=========================================================================================
.. note::
	* T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

Autoencoder
=========================================================================================
.. note::
	* BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension

Cross-Lingual
=========================================================================================
.. note::
	* `[ruder.io] The State of Multilingual AI <https://www.ruder.io/state-of-multilingual-ai/>`_
	* [Encoder] XLM-R [Roberta]: Unsupervised Cross-lingual Representation Learning at Scale
	* [Decoder] XGLM [GPT-3]: Few-shot Learning with Multilingual Generative Language Models
	* [Encoder-Decoder] mT5 [T5]: A Massively Multilingual Pre-trained Text-to-Text Transformer
	* [Autoencoder] mBART [BART]: Multilingual Denoising Pre-training for Neural Machine Translation

*****************************************************************************************
Training
*****************************************************************************************
Pretraining
=========================================================================================
.. note::
	* Improving Language Understanding by Generative Pre-Training
	* Universal Language Model Fine-tuning for Text Classification

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
.. note::
	* Text Generation

		* `[mlabonne.github.io] Decoding Strategies in Large Language Models <https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html>`_

	* Text Classification

		* Token Classification
		* Sentence Classification

			* Sentiment Analysis

	* Language Understanding

		* Finding Similar Items

			* Approximate Nearest Neighbour Search [DiskANN]

		* Document Summarization
		* Question Answering

	* Machine Translation

*****************************************************************************************
LLM Technology Stack
*****************************************************************************************
Embeddings for Search and Retrieval
=========================================================================================
Embedding Generation and Eval
-----------------------------------------------------------------------------------------
.. note::
	* [TechTarget] `Embedding models for semantic search: A guide <https://www.techtarget.com/searchenterpriseai/tip/Embedding-models-for-semantic-search-A-guide>`_	
	* Evaluation Metrics:

		* `BEIR <https://openreview.net/pdf?id=wCu6T5xFjeJ>`_
		* `MTEB <https://arxiv.org/pdf/2210.07316>`_
		* For speech and vision, refer to the guide above from TechTarget.

Model Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
	* [Huggingface] `SBERT <https://sbert.net/docs/sentence_transformer/pretrained_models.html>`_
	* [Google GTR - T5 Based] `Large Dual Encoders Are Generalizable Retrievers <https://arxiv.org/pdf/2112.07899>`_
	* [`Microsoft E5 <https://github.com/microsoft/unilm/tree/master/e5>`_] `Improving Text Embeddings with Large Language Models <https://arxiv.org/pdf/2401.00368>`_
	* [Cohere - Better Perf on RAG] `Embed v3 <https://cohere.com/blog/introducing-embed-v3>`_

Resources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
	* `Matryoshka (Russian Doll) Embeddings <https://huggingface.co/blog/matryoshka>`_ - learning embeddings of different dimensions

Embedding Retrieval: Vector DB
-----------------------------------------------------------------------------------------
.. note::
	* Pinecone `YouTube Playlist <https://youtube.com/playlist?list=PLRLVhGQeJDTLiw-ZJpgUtZW-bseS2gq9-&si=UBRFgChTmNnddLAt>`_
	* `LlamaIndex <https://www.llamaindex.ai/>`_: `YouTube Channel <https://www.youtube.com/@LlamaIndex>`_

Retrieval Augmented Generation (RAG)
-----------------------------------------------------------------------------------------
.. note::
	* [Stanford Lecture] `Stanford CS25: V3 I Retrieval Augmented Language Models <https://www.youtube.com/watch?v=mE7IDf2SmJg>`_
	* [Huggingface] `Rag Doc <https://huggingface.co/docs/transformers/main/en/model_doc/rag#rag>`_
	* [Nvidia] `RAG 101: Demystifying Retrieval-Augmented Generation Pipelines <https://resources.nvidia.com/en-us-ai-large-language-models/demystifying-rag-blog>`_
	* [Nvidia] `RAG 101: Retrieval-Augmented Generation Questions Answered <https://developer.nvidia.com/blog/rag-101-retrieval-augmented-generation-questions-answered/>`_

Resources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
	* [FAIR] `Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks <https://arxiv.org/pdf/2005.11401>`_
	* [FAIR] `REPLUG: Retrieval-Augmented Black-Box Language Models <https://arxiv.org/pdf/2301.12652>`_
	* RALM - `In-Context Retrieval-Augmented Language Models <https://arxiv.org/pdf/2302.00083>`_
	* `Improving the Domain Adaptation of Retrieval Augmented Generation (RAG) Models for Open Domain Question Answering <https://arxiv.org/pdf/2210.02627v1>`_
	* `FINE-TUNE THE ENTIRE RAG ARCHITECTURE (INCLUDING DPR RETRIEVER) FOR QUESTION-ANSWERING <https://arxiv.org/pdf/2106.11517v1>`_

[TODO: Classify Later] Other Topics
=========================================================================================
	* Prompt Engineering
	* Prompt Tuning
	* RLHF/DPO: `Huggingface TRL <https://huggingface.co/docs/trl/index>`_
	* `[PEFT] <https://huggingface.co/docs/peft/index>`_ - Performance Efficient Fine-Tuning
	* `[BitsAndBytes] <https://huggingface.co/docs/bitsandbytes/index>`_ - Quantization
	* RAG

Resources
=========================================================================================
.. note::
	* `OpenAI Docs <https://platform.openai.com/docs/overview>`_
	* `[HN] You probably don’t need to fine-tune an LLM <https://news.ycombinator.com/item?id=37174850>`_
	* `[Ask HN] Most efficient way to fine-tune an LLM in 2024? <https://news.ycombinator.com/item?id=39934480>`_
	* `[HN] Finetuning Large Language Models <https://news.ycombinator.com/item?id=35666201>`_

		* `[magazine.sebastianraschka.com] Finetuning Large Language Models <https://magazine.sebastianraschka.com/p/finetuning-large-language-models>`_
	* `[Github] LLM Course <https://github.com/mlabonne/llm-course>`_
