#########################################################################################
Natural Language Processing
#########################################################################################
*****************************************************************************************
Practical
*****************************************************************************************
.. warning::
	* [horace.io] `Making Deep Learning Go Brrrr From First Principles <https://horace.io/brrr_intro.html>`_
	* [Paper] `Data Movement Is All You Need: A Case Study on Optimizing Transformers <https://arxiv.org/abs/2007.00072>`_
	* [Video] `Profiling PyTorch Models for NVIDIA GPUs <https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31644/>`_	
	* [pytorch.org] `What Every User Should Know About Mixed Precision Training in PyTorch <https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/>`_
	* [pytorch.org] `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_
	* [pytorch.org] `PyTorch Profiler <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`_
	* [tinkerd.net] `Distributed Training and DeepSpeed <https://tinkerd.net/blog/machine-learning/distributed-training/>`_
	* [tinkerd.net] `Writing CUDA Kernels for PyTorch <https://tinkerd.net/blog/machine-learning/cuda-basics/>`_
	* [spaCy] `Library for NLU/IE Tasks <https://spacy.io/usage/spacy-101>`_, `LLM-variants <https://spacy.io/usage/large-language-models>`_
	* [confident.ai] `LLM Evaluation Metrics: The Ultimate LLM Evaluation Guide <https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation>`_

.. note::
	* [Paper] `The Responsible Foundation Model Development Cheatsheet: A Review of Tools & Resources <https://arxiv.org/abs/2406.16746>`_
	* [PapersWithCode] `Natural Language Processing <https://paperswithcode.com/area/natural-language-processing>`_ (check all relevant subcategories)

*****************************************************************************************
Activations
*****************************************************************************************
.. note::
	* [SiL] `Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_
	* [GELU] `Gaussian Error Linear Units <https://arxiv.org/abs/1606.08415>`_
	* [Swish] `Searching for Activation Functions <https://arxiv.org/pdf/1710.05941v2>`_	
	* [Medium] `On the Disparity Between Swish and GELU <https://towardsdatascience.com/on-the-disparity-between-swish-and-gelu-1ddde902d64b>`_
	* [Noam] `GLU Variants Improve Transformer <https://arxiv.org/pdf/2002.05202v1>`_

*****************************************************************************************
Normalisation
*****************************************************************************************
* [Internal Covariate Shift][BN] `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_
* [LN] `Layer Normalization <https://arxiv.org/abs/1607.06450>`_
* [RMSNorm] `Root Mean Square Layer Normalization <https://arxiv.org/abs/1910.07467>`_
* [PreLN][Detailed Study with Mean-Field Theory] `On Layer Normalization in the Transformer Architecture <https://arxiv.org/abs/2002.04745>`_

.. warning::
	For theoretical understanding of MFT and NTK, start from this MLSS video `here <https://youtu.be/rzPHnBGmr_E?si=JifFfB9r0Ax373VR>`_.

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

Position Encoding
=========================================================================================
.. note::
	* `Position Information in Transformers: An Overview <https://arxiv.org/abs/2102.11090>`_
	* `Rethinking Positional Encoding in Language Pre-training <https://arxiv.org/abs/2006.15595>`_
	* [Blog] `RoPE <https://blog.eleuther.ai/rotary-embeddings/>`_
	* RoFormer: `Enhanced Transformer with Rotary Position Embedding <https://arxiv.org/abs/2104.09864>`_

Attention
=========================================================================================
Understanding Einsum
-----------------------------------------------------------------------------------------
.. warning::
	.. collapse:: Expand Code
	
	   .. literalinclude:: ../../code/einsum.py
	      :language: python
	      :linenos:

.. note::
	* Dot product Attention (single query)

		.. collapse:: Expand Code

		   .. literalinclude:: ../../code/attn.py
		      :language: python
		      :linenos:

	* Multi-head Attention (single query)

		.. collapse:: Expand Code

		   .. literalinclude:: ../../code/mha.py
		      :language: python
		      :linenos:

	* Multi-head Attention (sequential query)

		.. collapse:: Expand Code

		   .. literalinclude:: ../../code/mha_seq.py
		      :language: python
		      :linenos:

	* Masked Multi-head Attention (parallel query)

		.. collapse:: Expand Code

		   .. literalinclude:: ../../code/mha_par.py
		      :language: python
		      :linenos:

	* Masked Multi-head Attention Batched (parallel query)

		.. collapse:: Expand Code

		   .. literalinclude:: ../../code/mha_par_batched.py
		      :language: python
		      :linenos:

	* Multi-head Attention Batched (sequential query)

		.. collapse:: Expand Code

		   .. literalinclude:: ../../code/mha_seq_batched.py
		      :language: python
		      :linenos:

	* Masked Multi-query Attention Batched (parallel query)

		.. collapse:: Expand Code

		   .. literalinclude:: ../../code/mqa_par_batched.py
		      :language: python
		      :linenos:

	* Multi-query Attention Batched (sequential query)

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
* [tinkerd.net] `Multi-Query & Grouped-Query Attention <https://tinkerd.net/blog/machine-learning/multi-query-attention/>`_

Decoding
=========================================================================================
* Beam Search, Top-K, Top-p/Nuclear, Temperature
* `[mlabonne.github.io] Decoding Strategies in Large Language Models <https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html>`_
* Speculative Deocding

*****************************************************************************************
Transformer Architecture
*****************************************************************************************
Encoder [BERT]
=========================================================================================
.. note::
	* BERT: `Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`_

		* [tinkerd.net] Additional Resources: `BERT Tokenization <https://tinkerd.net/blog/machine-learning/bert-tokenization/>`_, `BERT Embeddings <https://tinkerd.net/blog/machine-learning/bert-embeddings/>`_, `BERT Encoder Layer <https://tinkerd.net/blog/machine-learning/bert-encoder/>`_
	* `A Primer in BERTology: What we know about how BERT works <https://arxiv.org/abs/2002.12327>`_
	* RoBERTa: `A Robustly Optimized BERT Pretraining Approach <https://arxiv.org/abs/1907.11692>`_
	* XLM: `Cross-lingual Language Model Pretraining <https://arxiv.org/abs/1901.07291>`_
	* TwinBERT: `Distilling Knowledge to Twin-Structured BERT Models for Efficient Retrieval <https://arxiv.org/abs/2002.06275>`_

Decoder [GPT]
=========================================================================================
.. note::
	* `[jalammar.github.io] The Illustrated GPT-2 <https://jalammar.github.io/illustrated-gpt2/>`_
	* `[cameronrwolfe.substack.com] Decoder-Only Transformers: The Workhorse of Generative LLMs <https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse>`_
	* GPT-2: `Language Models are Unsupervised Multitask Learners <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>`_
	* GPT-3: `Language Models are Few-Shot Learners <https://arxiv.org/abs/2005.14165>`_

Encoder-Decoder [T5]
=========================================================================================
.. note::
	* T5: `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer <https://arxiv.org/abs/1910.10683>`_

Autoencoder [BART]
=========================================================================================
.. note::
	* BART: `Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension <https://arxiv.org/abs/1910.13461>`_

Cross-Lingual
=========================================================================================
.. note::
	* [Encoder] XLM-R [Roberta]: `Unsupervised Cross-lingual Representation Learning at Scale <https://arxiv.org/abs/1911.02116>`_
	* [Decoder] XGLM [GPT-3]: `Few-shot Learning with Multilingual Generative Language Models <https://arxiv.org/abs/2112.10668>`_
	* [Encoder-Decoder] mT5 [T5]: `A Massively Multilingual Pre-trained Text-to-Text Transformer <https://arxiv.org/abs/2010.11934>`_
	* [Autoencoder] mBART [BART]: `Multilingual Denoising Pre-training for Neural Machine Translation <https://arxiv.org/abs/2001.08210>`_

.. seealso::
	* `[ruder.io] The State of Multilingual AI <https://www.ruder.io/state-of-multilingual-ai/>`_

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
SoDA

Supervised Fine-Tuning
=========================================================================================
Reinforcement Learning with Human Feedback (RLHF)
=========================================================================================
Direct Preference Optimisation (DPO)
=========================================================================================
*****************************************************************************************
Special Techniques
*****************************************************************************************
Low-Rank Approximations (LoRA)
=========================================================================================
.. note::
	* [tinkerd.net]: `Language Model Fine-Tuning with LoRA <https://tinkerd.net/blog/machine-learning/lora/>`_

MOE
=========================================================================================
.. note::
	* `Mixture of Experts Pattern for Transformer Models <https://tinkerd.net/blog/machine-learning/mixture-of-experts/>`_
	* Mixtral

Long Context
=========================================================================================
.. csv-table:: 
	:header: "Category","Model","Max sequence length"
	:align: center

		Full Attention,Flash Attention,Not specified
		Augmented Attention,Transformer-XL,Up to 16k tokens (depends on the segment length)
		Augmented Attention,Longformer,Up to 4k tokens
		Recurrence,RMT,Not specified
		Recurrence,xLSTM,Not specified
		Recurrence,Feedback Attention,Not specified
		State Space,Mamba,Not specified
		State Space,Jamba,Not specified

Optimized Full Attention
-----------------------------------------------------------------------------------------
* Flash Attention

Augmented Attention
-----------------------------------------------------------------------------------------
* Receptive Field Modification: Transformer-xl
* Sparse Attention: Longformer

Recurrence
-----------------------------------------------------------------------------------------
* RMT: Recurrent Memory Transformer
* Feedback Attention

Non Transformer
-----------------------------------------------------------------------------------------
* State SpaceModels: Mamba, Jamba
	.. note::
		* [Mamba] `Linear-Time Sequence Modeling with Selective State Spaces <https://arxiv.org/abs/2312.00752>`_
		* `Understanding State Space Models <https://tinkerd.net/blog/machine-learning/state-space-models/>`_

* LSTM: xLSTM

Retrieval Augmented
-----------------------------------------------------------------------------------------
* Bidirectional Attention for encoder: BERT, T5, Electra, Matryoshka, Multimodal

	* Approximate Nearest Neighbour Search
* Causal attention for decoder: GPT, Multimodal generation

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
	* [MSR] `From Local to Global: A Graph RAG Approach to Query-Focused Summarization <https://arxiv.org/pdf/2404.16130>`_

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
	* RAGAS: `Automated Evaluation of Retrieval Augmented Generation <https://arxiv.org/abs/2309.15217>`_
	* [confident.ai] `DeepEval <https://docs.confident-ai.com/docs/getting-started>`_

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

*****************************************************************************************
Task Specific Setup
*****************************************************************************************
Classification Tasks
=========================================================================================
1. Sentiment Analysis
-----------------------------------------------------------------------------------------
Description:
Sentiment analysis involves determining the sentiment or emotional tone behind a piece of text, typically classified as positive, negative, or neutral.

Example:

- Input: "I love this product!"
- Output: "Positive"

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score

Benchmark Datasets:

- IMDb Movie Reviews
- Sentiment140
- SST (Stanford Sentiment Treebank)

Example Prompt:
"Determine the sentiment of the following text: 'I love this product!'"

Information Retrieval (IR) Tasks
=========================================================================================
1. Document Retrieval
-----------------------------------------------------------------------------------------
Description:
Document retrieval involves finding and ranking relevant documents from a large corpus in response to a user's query.

Example:

- Input: Query: "What are the symptoms of COVID-19?"
- Output: [List of relevant documents about COVID-19 symptoms]

Evaluation Metrics:

- Precision at k (P@k)
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG)

Benchmark Datasets:

- TREC (Text REtrieval Conference)
- CLEF (Conference and Labs of the Evaluation Forum)
- MSMARCO

Example Prompt:
"Retrieve the top 5 documents related to the query: 'What are the symptoms of COVID-19?'"

2. Passage Retrieval
-----------------------------------------------------------------------------------------
Description:
Passage retrieval involves finding and ranking relevant passages or sections within documents in response to a user's query.

Example:

- Input: Query: "What is the capital of France?"
- Output: [List of passages containing information about the capital of France]

Evaluation Metrics:

- Precision at k (P@k)
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG)

Benchmark Datasets:

- MSMARCO Passage Ranking
- TREC Deep Learning

Example Prompt:
"Retrieve the top 5 passages related to the query: 'What is the capital of France?'"

3. Query Expansion
-----------------------------------------------------------------------------------------
Description:
Query expansion involves modifying a user's query by adding additional terms to improve retrieval performance.

Example:

- Input: Query: "COVID-19"
- Output: Expanded Query: "COVID-19 coronavirus symptoms pandemic"

Evaluation Metrics:

- Precision
- Recall
- Mean Average Precision (MAP)

Benchmark Datasets:

- TREC
- CLEF

Example Prompt:
"Expand the following query to improve search results: 'COVID-19'"

4. Question Answering (QA)
-----------------------------------------------------------------------------------------
Description:
QA involves retrieving answers to questions posed in natural language, often using information from a large corpus.

Example:

- Input: Question: "What is the tallest mountain in the world?"
- Output: "Mount Everest"

Evaluation Metrics:

- Exact Match (EM)
- F1 Score

Benchmark Datasets:

- SQuAD (Stanford Question Answering Dataset)
- Natural Questions
- TriviaQA

Example Prompt:
"Answer the following question: 'What is the tallest mountain in the world?'"

Information Extraction (IE) Tasks
=========================================================================================
1. Named Entity Recognition (NER)
-----------------------------------------------------------------------------------------
Description:
NER involves identifying and classifying entities in text into predefined categories such as names of people, organizations, locations, dates, etc.

Example:

- Input: "Barack Obama was born in Hawaii."
- Output: [("Barack Obama", "PERSON"), ("Hawaii", "LOCATION")]

Evaluation Metrics:

- Precision
- Recall
- F1 Score

Benchmark Datasets:

- CoNLL-2003
- OntoNotes
- WNUT 2017

Example Prompt:
"Identify and classify named entities in the following sentence: 'Barack Obama was born in Hawaii.'"

2. Relation Extraction
-----------------------------------------------------------------------------------------
Description:
Relation extraction involves identifying and classifying the relationships between entities in text.

Example:

- Input: "Barack Obama was born in Hawaii."
- Output: ("Barack Obama", "born in", "Hawaii")

Evaluation Metrics:

- Precision
- Recall
- F1 Score

Benchmark Datasets:

- TACRED
- SemEval
- ACE 2005

Example Prompt:
"Identify the relationship between entities in the following sentence: 'Barack Obama was born in Hawaii.'"

3. Event Extraction
-----------------------------------------------------------------------------------------
Description:
Event extraction involves identifying events in text and their participants, attributes, and the context in which they occur.

Example:

- Input: "An earthquake of magnitude 6.5 struck California yesterday."
- Output: [("earthquake", "magnitude 6.5", "California", "yesterday")]

Evaluation Metrics:

- Precision
- Recall
- F1 Score

Benchmark Datasets:

- ACE 2005
- MUC-4
- TAC KBP

Example Prompt:
"Extract events and their details from the following text: 'An earthquake of magnitude 6.5 struck California yesterday.'"

4. Coreference Resolution
-----------------------------------------------------------------------------------------
Description:
Coreference resolution involves identifying when different expressions in a text refer to the same entity.

Example:

- Input: "Jane went to the market. She bought apples."
- Output: [("Jane", "She")]

Evaluation Metrics:

- Precision
- Recall
- F1 Score

Benchmark Datasets:

- CoNLL-2012 Shared Task
- OntoNotes

Example Prompt:
"Identify coreferences in the following text: 'Jane went to the market. She bought apples.'"

Sequence to Sequence Tasks
=========================================================================================
1. Machine Translation
-----------------------------------------------------------------------------------------
Description:
Machine translation involves translating text from one language to another.

Example:

- Input: "Hello, how are you?" (English)
- Output: "Hola, ¿cómo estás?" (Spanish)

Evaluation Metrics:

- BLEU Score
- METEOR
- TER

Benchmark Datasets:

- WMT (Workshop on Machine Translation)
- IWSLT (International Workshop on Spoken Language Translation)

Example Prompt:
"Translate the following text from English to Spanish: 'Hello, how are you?'"

2. Text Summarization
-----------------------------------------------------------------------------------------
Description:
Text summarization involves generating a concise summary of a longer document while preserving key information.

Example:

- Input: "Artificial intelligence is a branch of computer science that aims to create intelligent machines. It has become an essential part of the technology industry."
- Output: "AI is a branch of computer science aiming to create intelligent machines, essential in technology."

Evaluation Metrics:

- ROUGE Score
- BLEU Score

Benchmark Datasets:

- CNN/Daily Mail
- XSum
- Gigaword

Example Prompt:
"Summarize the following text: 'Artificial intelligence is a branch of computer science that aims to create intelligent machines. It has become an essential part of the technology industry.'"

3. Text Generation
-----------------------------------------------------------------------------------------
Description:
Text generation involves creating new text that is coherent and contextually relevant based on a given input prompt.

Example:

- Input: "Once upon a time"
- Output: "Once upon a time, in a small village, there lived a brave young girl named Ella."

Evaluation Metrics:

- Perplexity
- BLEU Score
- Human Evaluation

Benchmark Datasets:

- OpenAI GPT-3 Playground
- EleutherAI's Pile
- WikiText

Example Prompt:
"Generate a continuation for the following text: 'Once upon a time, in a small village, there lived a brave young girl named Ella.'"

Multimodal Tasks
=========================================================================================
1. Text-to-Speech (TTS)
-----------------------------------------------------------------------------------------
Description:
TTS involves converting written text into spoken words.

Example:

- Input: "Good morning, everyone."
- Output: [Audio clip saying "Good morning, everyone."]

Evaluation Metrics:

- Mean Opinion Score (MOS)
- Word Error Rate (WER)
- Naturalness

Benchmark Datasets:

- LJSpeech
- LibriSpeech
- VCTK

Example Prompt:
"Convert the following text to speech: 'Good morning, everyone.'"

2. Speech Recognition
-----------------------------------------------------------------------------------------
Description:
Speech recognition involves converting spoken language into written text.

Example:

- Input: [Audio clip saying "Hello, world!"]
- Output: "Hello, world!"

Evaluation Metrics:

- Word Error Rate (WER)
- Sentence Error Rate (SER)

Benchmark Datasets:

- LibriSpeech
- TED-LIUM
- Common Voice

Example Prompt:
"Transcribe the following audio clip: [Audio clip saying 'Hello, world!']"

Extending Vocab for Domain-Adaptation or Fine-Tuning
=========================================================================================
1. Extend the Tokenizer Vocabulary
-----------------------------------------------------------------------------------------
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
-----------------------------------------------------------------------------------------
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
-----------------------------------------------------------------------------------------
* Tokenizer Vocabulary: Ensure that after extending the tokenizer vocabulary, you save it or use it consistently across your tasks.
* Embedding Adjustment: The approach here adds new tokens and initializes their embeddings separately from the pre-trained embeddings. This keeps the original embeddings intact while allowing new tokens to have their embeddings learned during fine-tuning.
* Fine-Tuning: If you plan to fine-tune the model on your specific tasks, you would then proceed with training using your domain-specific data, where the model will adapt not only to the new tokens but also to the specific patterns in your data.

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

TODO
- constitutional ai
- guardrails

https://github.com/microsoft/unilm

- eval for ie tasks - open vs supervised
- llm evals: https://github.com/openai/evals
- multimodal ie
- multimodal: text + image
	- classification: 
		- clip: https://github.com/openai/CLIP
			Learning Transferable Visual Models From Natural Language Supervision
		- cnn
	- generation: 
		- dall-e: https://github.com/openai/dall-e
			Zero-Shot Text-to-Image Generation
		- latent-diffusion: https://github.com/CompVis/latent-diffusion
			- High-Resolution Image Synthesis with Latent Diffusion Models
			- Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models
		- stable diffusion: https://github.com/CompVis/stable-diffusion
			- Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
		- vision transformers and diffusion models 
	- eval
- cnn:
	- image classification
	- object detection (bounding box): 
		https://paperswithcode.com/task/object-detection
		YOLOv4: Optimal Speed and Accuracy of Object Detection
	- image segmentation:
		- GeminiFusion: Efficient Pixel-wise Multimodal Fusion for Vision Transformer
- recsys - context based (in session rec - llm), interaction based (collaborative filtering - mf, gcn)
- nlp downstream tasks
- hardware p40, v100, a100 - arch, cost

training: domain adaptation (mlm/rtd/ssl-kl/clm), finetuning (sft/it), alignment and preference optim (rhlf/dpo)

design e2e: integrate user feedback

- prompt best guide: Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine
	- Zero-shot
	- Random few-shot
	- Random few-shot, chain-of-thought
	- kNN, few-shot, chain-of-though
	- Ensemble w/ choice shuffle
- domain understanding

1. information retrieval (q/a, summarization, retrieval)
	- MLM based: BERT, T5
	- RTD based: Electra
	- Contrastive Learning based:
		- image: OG image and distorted image form pos-pairs
		- text: contriever
			- contrastive learning based embeddings
			- infonce loss: softmax over 1 positive and K negative
			- getting positive: 
				(a) Inverse Cloze Task (contiguous segment as query, rest as doc) - relates with closure of a query
				(b) Independent cropping - sample two independent contiguous pieces of text
			- getting negatives:
				(a) in-batch negatives
				(b) negs from previous batch docs - called keys. either not updated or updated slowly with different parameterization including momentum (moco)
		- text: e5

- monosemanticity
	- many neurons are polysemantic: they respond to mixtures of seemingly unrelated inputs.
	- neural network represents more independent "features" of the data than it has neurons by assigning each feature its own linear combination of neurons. If we view each feature as a vector over the neurons, then the set of features form an overcomplete linear basis for the activations of the network neurons.
	- towards monosemanticity:
		(1) creating models without superposition, perhaps by encouraging activation sparsity; 
		(2) using dictionary learning to find an overcomplete feature basis in a model exhibiting superposition; and 
		(3) hybrid approaches relying on a combination of the two.
	- developed counterexamples which persuaded us that the 
		- sparse architectural approach (approach 1) was insufficient to prevent polysemanticity, and that 
		- standard dictionary learning methods (approach 2) had significant issues with overfitting.
	- use a weak dictionary learning algorithm called a sparse autoencoder to generate learned features from a trained model that offer a more monosemantic unit of analysis than the model's neurons themselves.

- "lost in the middle" using longer context (primacy bias, recency bias) - U-shaped curve
	-> if using only a decoder model, due to masked attention, put the question at the end 
	-> instruction tuned is much better
	-> relevance order of the retriever matters a lot

- extending context length
	- needle in a haystack
	- l-eval, novelqa, infty-bench
	- nocha (fictional, unseen books with true/false q/a pairs 
		- performs better when fact is present in the book at sentence level
		- performs worse if requires global reasoning or if contains extensive world building
	- position embeddings 
		- change the angle hyperparameter in RoPE to deal with longer sequences
	- efficient attention 
		- full attention with hardware-aware algorithm design - flash attention
		- sparse attention techniques: sliding window attention, block attention
	- data engineering - replicate larger model perf using 7b/13b llama
		- continuous pretraining
			- 1-5B new tokens for 
			- upsampling longer sequences
			- same #tokens per batch (adjusted as per sequence length and batch size)
			- 2e-5 lr cosine schedule
			- 2x8 a100 gpu, 7 day training, flashattention (3x time for 80k vs 4k, majority time goes in cpu<->gpu, gpu<->gpu, and hbm<->sm)
		- instruction tuning: rlhf data + self instruct
			- (a) chunk long doc (b) from long doc formulate q/a (c) use OG doc and q/a pair as training
			- 1e-5 lr constant
			- lora/qlora
	- incorporating some form of recurrance relation - transformer-xl, longformer, rmt

- rag based solution
	- baseline rag struggles
		- answering a question requires traversing disparate pieces of information through their shared attributes
		- holistically understand summarized semantic concepts over large data collections or even singular large documents.
	- graph rag: https://microsoft.github.io/graphrag/
		- The LLM processes the entire private dataset, creating references to all entities and relationships within the source data, which are then used to create an LLM-generated knowledge graph. 
		- This graph is then used to create a bottom-up clustering that organizes the data hierarchically into semantic clusters This partitioning allows for pre-summarization of semantic concepts and themes, which aids in holistic understanding of the dataset. 
		- At query time, both of these structures are used to provide materials for the LLM context window when answering a question. 
		- eval:
			- comprehensiveness (completeness within the framing of the implied context of the question)
			- human enfranchisement (provision of supporting source material or other contextual information)
			- diversity (provision of differing viewpoints or angles on the question posed)
			- selfcheckgpt
- chain-of-agents

issues:
- hallucination detection and mitigation
	- supervised: translation, summarization, image captioning
		- n-gram (bleu/rouge, meteor)
			- issues:
				- reference dependent, usually only one reference
				- often coarse or granular
				- unable to capture semantics: fail to adapt to stylistic changes in the reference
		- ask gpt (selfcheckgpt, g-eval)
			- evaluate on (a) adherence (b) correctness
			- issues:
				- blackbox, unexplainable
				- expensive
	- unsupervised:
		- perplexity-based (gpt-score, entropy, token confidence) - good second order metric to check
			- issues:
				- too granular, represents confusion - not hallucination in particular, often red herring
				- not always available

- sycophany

- alignment and preference
	- rlhf
	- dpo
	- reflexion

2. information extraction: **UniversalNER, **GLiNER
	- (entity) NER: named entity recognition, entity-linking
		- predefined entity-classes: location (LOC), organizations (ORG), person (PER) and Miscellaneous (MISC). 
			- https://huggingface.co/dslim/bert-base-NER
			- https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll03-english			
		- open entity-classes: 
			- UniversalNER: https://universal-ner.github.io/, https://huggingface.co/Universal-NER
			- GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer https://huggingface.co/urchade/gliner_large-v2
			- GLiNER - Multitask: https://www.knowledgator.com/ -> https://huggingface.co/knowledgator/gliner-multitask-large-v0.5

		- Open IE eval: Preserving Knowledge Invariance: Rethinking Robustness Evaluation of Open Information Extraction (https://github.com/qijimrc/ROBUST/tree/master)		
		- LLMaAA: Making Large Language Models as Active Annotators https://github.com/ridiculouz/LLMaAA/tree/main
		- A Deep Learning Based Knowledge Extraction Toolkit for Knowledge Graph Construction (https://github.com/zjunlp/DeepKE)

	- (relationship) RE: relationship extraction
		- QA4RE: Aligning Instruction Tasks Unlocks Large Language Models as Zero-Shot Relation Extractors (ZS Pr) https://github.com/OSU-NLP-Group/QA4RE
		- DocGNRE: Semi-automatic Data Enhancement for Document-Level Relation Extraction with Distant Supervision from Large Language Models (https://github.com/bigai-nlco/DocGNRE)

	- (event) EE: event extraction

Resources
=========================================================================================
.. note::
	* `OpenAI Docs <https://platform.openai.com/docs/overview>`_
	* `[HN] You probably don’t need to fine-tune an LLM <https://news.ycombinator.com/item?id=37174850>`_
	* `[Ask HN] Most efficient way to fine-tune an LLM in 2024? <https://news.ycombinator.com/item?id=39934480>`_
	* `[HN] Finetuning Large Language Models <https://news.ycombinator.com/item?id=35666201>`_

		* `[magazine.sebastianraschka.com] Finetuning Large Language Models <https://magazine.sebastianraschka.com/p/finetuning-large-language-models>`_
	* `[Github] LLM Course <https://github.com/mlabonne/llm-course>`_
