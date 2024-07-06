#########################################################################################
Natural Language Processing
#########################################################################################
.. warning::
	Goal: Write summary of key ideas and summary of papers

*****************************************************************************************
Practical
*****************************************************************************************
.. note::
	* `What Every User Should Know About Mixed Precision Training in PyTorch <https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/>`_

*****************************************************************************************
Tokenizers
*****************************************************************************************
WordPiece
=========================================================================================
.. seealso::
	`Google's Neural Machine Translation System <https://arxiv.org/abs/1609.08144v2>`_

SentencePiece
=========================================================================================
.. seealso::
	`SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing <https://arxiv.org/abs/1808.06226>`_

Byte-Pair Encoding (BPE)
=========================================================================================
.. seealso::
	`Neural Machine Translation of Rare Words with Subword Units <https://arxiv.org/abs/1508.07909v5>`_

*****************************************************************************************
Word Embeddings
*****************************************************************************************
.. note::
	* Word2Vec: Efficient Estimation of Word Representations in Vector Space
	* GloVe: Global Vectors forWord Representation
	* Evaluation methods for unsupervised word embeddings

*****************************************************************************************
Sequence Modeling
*****************************************************************************************
RNN
=========================================================================================
.. seealso::
	.. collapse:: Expand Code

	   .. literalinclude:: ../../code/rnn.py
	      :language: python
	      :linenos:

.. note::
	* `On the diffculty of training Recurrent Neural Networks <https://arxiv.org/abs/1211.5063>`_
	* `Sequence to Sequence Learning with Neural Networks <https://arxiv.org/abs/1409.3215>`_
	* `Neural Machine Translation by Jointly Learning to Align and Translate <https://arxiv.org/abs/1409.0473>`_

LSTM
=========================================================================================
.. seealso::
	.. collapse:: Expand Code

	   .. literalinclude:: ../../code/lstm.py
	      :language: python
	      :linenos:

.. note::
	* `StatQuest on LSTM <https://www.youtube.com/watch?v=YCzL96nL7j0>`_

*****************************************************************************************
Transformer
*****************************************************************************************
Resources
=========================================================================================
.. warning::
	* [Karpathy] `LLM101n: Let's build a Storyteller <https://github.com/karpathy/LLM101n>`_
	* [MoE] `Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity <https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf>`_

.. note::
	* [Harvard] `The Annotated Transformer <https://nlp.seas.harvard.edu/annotated-transformer/>`_
	* [jalammar.github.io] `The Illustrated Transformer <https://jalammar.github.io/illustrated-transformer/>`_
	* [lilianweng.github.io] `Attention? Attention! <https://lilianweng.github.io/posts/2018-06-24-attention/>`_
	* [d2l] `The Transformer Architecture <https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html>`_

Attention
=========================================================================================
Dot product Attention (single query)
-----------------------------------------------------------------------------------------
.. note::
	.. collapse:: Expand Code

	   .. literalinclude:: ../../code/attn.py
	      :language: python
	      :linenos:

Multi-head Attention (single query)
-----------------------------------------------------------------------------------------
.. note::
	.. collapse:: Expand Code

	   .. literalinclude:: ../../code/mha.py
	      :language: python
	      :linenos:

Multi-head Attention (sequential query)
-----------------------------------------------------------------------------------------
.. note::
	.. collapse:: Expand Code

	   .. literalinclude:: ../../code/mha_seq.py
	      :language: python
	      :linenos:

Masked Multi-head Attention (parallel query)
-----------------------------------------------------------------------------------------
.. note::
	.. collapse:: Expand Code

	   .. literalinclude:: ../../code/mha_par.py
	      :language: python
	      :linenos:

Masked Multi-head Attention Batched (parallel query)
-----------------------------------------------------------------------------------------
.. note::
	.. collapse:: Expand Code
	
	   .. literalinclude:: ../../code/mha_par_batched.py
	      :language: python
	      :linenos:

Multi-head Attention Batched (sequential query)
-----------------------------------------------------------------------------------------
.. note::
	.. collapse:: Expand Code

	   .. literalinclude:: ../../code/mha_seq_batched.py
	      :language: python
	      :linenos:

Masked Multi-query Attention Batched (parallel query)
-----------------------------------------------------------------------------------------
.. note::
	.. collapse:: Expand Code
	
	   .. literalinclude:: ../../code/mqa_par_batched.py
	      :language: python
	      :linenos:

Multi-query Attention Batched (sequential query)
-----------------------------------------------------------------------------------------
.. note::
	.. collapse:: Expand Code

	   .. literalinclude:: ../../code/mqa_seq_batched.py
	      :language: python
	      :linenos:

UnitTest
-----------------------------------------------------------------------------------------
.. seealso::
	.. collapse:: UnitTest of implementation

	   .. literalinclude:: ../../code/attn_test.py
	      :language: python
	      :linenos:

Resources
-----------------------------------------------------------------------------------------
* [MHA] `Attention Is All You Need <https://arxiv.org/abs/1706.03762v7>`_
* [MQA] `Fast Transformer Decoding: One Write-Head is All You Need <https://arxiv.org/abs/1911.02150>`_
* [GQA] `GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints <https://arxiv.org/abs/2305.13245v3>`_

Activations
=========================================================================================
.. note::
	* [Noam] `GLU Variants Improve Transformer <https://arxiv.org/pdf/2002.05202v1>`_

Normalisation
=========================================================================================
* `Layer Normalization <https://arxiv.org/abs/1607.06450>`_
* [RMSNorm] `Root Mean Square Layer Normalization <https://arxiv.org/abs/1910.07467>`_
* [PreNorm] `On Layer Normalization in the Transformer Architecture <https://arxiv.org/abs/2002.04745>`_

Position Encoding
=========================================================================================
.. note::
	* `Position Information in Transformers: An Overview <https://arxiv.org/abs/2102.11090>`_
	* `Rethinking Positional Encoding in Language Pre-training <https://arxiv.org/abs/2006.15595>`_
	* `RoPE <https://blog.eleuther.ai/rotary-embeddings/>`_

*****************************************************************************************
Transformer Architecture
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
LLM Technology Stack
*****************************************************************************************
Embeddings for Search and Retrieval
=========================================================================================
.. note::
	* SPLADE: `SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval <https://arxiv.org/pdf/2109.10086>`_
	* [Meta] DRAGON: `How to Train Your DRAGON: Diverse Augmentation Towards Generalizable Dense Retrieval <https://arxiv.org/pdf/2302.07452>`_

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

Embedding Retrieval
-----------------------------------------------------------------------------------------
Vector DB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
	* Pinecone `YouTube Playlist <https://youtube.com/playlist?list=PLRLVhGQeJDTLiw-ZJpgUtZW-bseS2gq9-&si=UBRFgChTmNnddLAt>`_
	* Chroma, Weaviate

RAG Focused
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
	* `LlamaIndex <https://www.llamaindex.ai/>`_: `YouTube Channel <https://www.youtube.com/@LlamaIndex>`_
	* `[LlamaIndex] Structured Hierarchical Retrieval <https://docs.llamaindex.ai/en/stable/examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval/#structured-hierarchical-retrieval>`_
	* `Child-Parent Recursive Retriever <https://docs.llamaindex.ai/en/stable/examples/retrievers/recursive_retriever_nodes/>`_

Retrieval Augmented Generation (RAG)
=========================================================================================
.. note::
	* [SUPER IMPORTANT][Stanford Lecture] `Stanford CS25: V3 I Retrieval Augmented Language Models <https://www.youtube.com/watch?v=mE7IDf2SmJg>`_
	* [Huggingface] `RAG paper - RAG Doc <https://huggingface.co/docs/transformers/main/en/model_doc/rag#rag>`_
	* [Nvidia] `RAG 101: Demystifying Retrieval-Augmented Generation Pipelines <https://resources.nvidia.com/en-us-ai-large-language-models/demystifying-rag-blog>`_
	* [Nvidia] `RAG 101: Retrieval-Augmented Generation Questions Answered <https://developer.nvidia.com/blog/rag-101-retrieval-augmented-generation-questions-answered/>`_

Resources
-----------------------------------------------------------------------------------------
Frozen RAG
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
	* [FAIR] `REPLUG: Retrieval-Augmented Black-Box Language Models <https://arxiv.org/pdf/2301.12652>`_
	* RALM: `In-Context Retrieval-Augmented Language Models <https://arxiv.org/pdf/2302.00083>`_

Trained RAG
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
	* [FAIR] RAG: `Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks <https://arxiv.org/pdf/2005.11401>`_
	* [FAIR] FiD: `Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering <https://arxiv.org/pdf/2007.01282>`_
	* [FAIR] Atlas: `Few-shot Learning with Retrieval Augmented Language Models <https://arxiv.org/pdf/2208.03299>`_	
	* [FAIR] kNN-LM: `Generalization through Memorization: Nearest Neighbor Language Models <https://arxiv.org/pdf/1911.00172>`_
	* [Goog] REALM: `Retrieval-Augmented Language Model Pre-Training <https://arxiv.org/pdf/2002.08909>`_
	* [FAIR] FLARE: `Active Retrieval Augmented Generation <https://arxiv.org/pdf/2305.06983>`_
	* [FAIR] Toolformer: `Language Models Can Teach Themselves to Use Tools <https://arxiv.org/pdf/2302.04761>`_
	* `SILO Language Models: Isolating Legal Risk In a Nonparametric Datastore <https://arxiv.org/pdf/2308.04430>`_
	* `Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection <https://arxiv.org/pdf/2310.11511>`_
	* [FAIR] RA-DIT: `Retrieval-Augmented Dual Instruction Tuning <https://arxiv.org/pdf/2310.01352>`_
	* Might not work well in practice:

		* [DeepMind] Retro: `Improving language models by retrieving from trillions of tokens <https://arxiv.org/pdf/2112.04426>`_
		* [Nvidia] Retro++: `InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining <https://arxiv.org/pdf/2310.07713v2>`_
	* Other stuff:

		* Issue with Frozen RAG: `Lost in the Middle: How Language Models Use Long Contexts <https://arxiv.org/pdf/2307.03172>`_
		* `Improving the Domain Adaptation of Retrieval Augmented Generation (RAG) Models for Open Domain Question Answering <https://arxiv.org/pdf/2210.02627v1>`_
		* `FINE-TUNE THE ENTIRE RAG ARCHITECTURE (INCLUDING DPR RETRIEVER) FOR QUESTION-ANSWERING <https://arxiv.org/pdf/2106.11517v1>`_

LM Eval
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
	* MMLU - `Measuring Massive Multitask Language Understanding <https://arxiv.org/pdf/2009.03300>`_
	* OpenQA - `Retrieving and Reading: A Comprehensive Survey on Open-domain Question Answering <https://arxiv.org/pdf/2101.00774>`_

.. seealso::
	* `Toolformer: Language Models Can Teach Themselves to Use Tools <https://arxiv.org/pdf/2302.04761>`_

Tech Stack
-----------------------------------------------------------------------------------------
.. note::
	* [LlamaIndex] `RAG pipeline with Llama3 <https://docs.llamaindex.ai/en/stable/examples/cookbooks/llama3_cookbook/#lets-build-rag-pipeline-with-llama3>`_
	* [Huggingface] `Simple RAG for GitHub issues using Hugging Face Zephyr and LangChain <https://huggingface.co/learn/cookbook/en/rag_zephyr_langchain>`_
	* [Huggingface] `Advanced RAG on Hugging Face documentation using LangChain <https://huggingface.co/learn/cookbook/en/advanced_rag>`_
	* [Huggingface] `RAG Evaluation <https://huggingface.co/learn/cookbook/en/rag_evaluation>`_
	* [Huggingface] `Building A RAG Ebook “Librarian” Using LlamaIndex <https://huggingface.co/learn/cookbook/en/rag_llamaindex_librarian>`_

Summary
-----------------------------------------------------------------------------------------
.. note::
	* x = query
	* z = doc
	* y = output

* Frozen RAG:

	- In-context:

		(a) In context RALM:

			- Retrieve k documents Z_k.
			- Rerank the docs using (1) zero-shot LM or (2) dedicated trained ranker.
			- Select top doc Z_top.
			- Prepend top doc in textual format as-is to the query as a part of the prompt for the LM to generate.
			- What we pass to the decoder: prompt with Z_top in it.
			- Issues: problematic for multiple docs (!)
	- In-context or in Seq2Seq or in decoder:

		(b) RePLUG:

			- Retrieve k documents.
			- Use cosine similarity score to compute p(Z_k | X).
			- What we pass to the decoder: concat{Z_k, X} or prompt with Z_k in it.
			- Make k forward passes in the decoder for each token to compute the likelihood over vocab using softmax p(Y_i | concat{Z_k, X}, Y_1..{i-1}).
			- Rescale the softmax with p(Z_k | X) and marginalize.
			- Pass the marginalized softmax to the decoder.
			- Issues: k forward passes at each token.
	- Just decoder:

		(c) kNN-LN:

			- For the current token consider X = encode(Y_1...Y_{i-1}).
			- Retrieve k documents Z_k matching X.
			- Make k forward passes in the decoder with the matching doc p_k(Y_i | Z_1..{i-1}).
			- Rescale p_k(Y_i | Z_1..{i-1}) over k and marginalize over the next token Y_i.
			- Do the same in the original sequence p_decode(Y_i | Z_1..{i-1}).
			- Interpolate between these using a hyperparameter.
			- Issues: k forward passes + retrieval at each token.
* Retriever trainable RAG:

	- Seq2Seq:

		(a) RePLUG-LSR:

			- Uses the parametric LM's output to update the retriever.
			- Loss: KL div between p(Z_k | X) and the posterior p(Z_k | X, Y_1..Y_N) works well.
* E2E trainable RAG:

	- Seq2Seq:

		(a) RAG:

			- Per token: same as RePLUG - output probability is marginalised at the time of generation of each token, pass it to beam decoder.
			- Per sequence: output probability is marginalised for the entire sequence.

				- Results in #Y generated sequences.
				- Might require additional passes.

			- Training - NLL loss across predicted tokens.
			- Issues: E2E training makes doc index update problematic, solution: just update the query encoder.
		(b) Atlas:

			- Multiple choice for updating the retriever - simple RePLUG-LSR type formulation based on the KL div between p(Z_k | X) and the posterior p(Z_k | X, Y_1..Y_N) works well.
			- Pre-training: same objective as the Seq2Seq (prefixLM or MLM) or decoder-only objective works well.
			- Training:
			- Issues:

[TODO: Classify Later] Other Topics
=========================================================================================
* Prompt Engineering

	* https://www.prompthub.us/blog
	* Nice video from OpenAi - https://youtu.be/ahnGLM-RC1Y?si=irFR4SoEfrEzyPh9
* Prompt Tuning
* Dataset search tool by google: https://datasetsearch.research.google.com
* Instruction Finetuning datasets

	* NaturalInstructions: https://github.com/allenai/natural-instructions/
* Supervised Finetuning datasets

	* UltraChat: https://github.com/thunlp/UltraChat
* RLHF/DPO datasets

	* Ultrafeedback: https://huggingface.co/datasets/argilla/ultrafeedback-curated
* Evaluation of instruction tuned/pre-trained models

	* MMLU

		* Paper: `Measuring Massive Multitask Language Understanding <https://arxiv.org/pdf/2009.03300>`_
		* Dataset: https://huggingface.co/datasets/cais/mmlu
	* Big-Bench

		* Paper: `Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models <https://arxiv.org/pdf/2206.04615>`_
		* Dataset: https://github.com/google/BIG-bench
* RLHF/DPO: `Huggingface TRL <https://huggingface.co/docs/trl/index>`_
* `[PEFT] <https://huggingface.co/docs/peft/index>`_ - Performance Efficient Fine-Tuning
* `[BitsAndBytes] <https://huggingface.co/docs/bitsandbytes/index>`_ - Quantization

Resources
=========================================================================================
.. note::
	* `OpenAI Docs <https://platform.openai.com/docs/overview>`_
	* `[HN] You probably don’t need to fine-tune an LLM <https://news.ycombinator.com/item?id=37174850>`_
	* `[Ask HN] Most efficient way to fine-tune an LLM in 2024? <https://news.ycombinator.com/item?id=39934480>`_
	* `[HN] Finetuning Large Language Models <https://news.ycombinator.com/item?id=35666201>`_

		* `[magazine.sebastianraschka.com] Finetuning Large Language Models <https://magazine.sebastianraschka.com/p/finetuning-large-language-models>`_
	* `[Github] LLM Course <https://github.com/mlabonne/llm-course>`_
