*****************************************************************************************
LLM Technology Stack
*****************************************************************************************
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

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
	* [Neo4j] `The GraphRAG Manifesto: Adding Knowledge to GenAI <https://neo4j.com/blog/graphrag-manifesto/>`_

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

Tech Stack
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
	* [LlamaIndex] `RAG pipeline with Llama3 <https://docs.llamaindex.ai/en/stable/examples/cookbooks/llama3_cookbook/#lets-build-rag-pipeline-with-llama3>`_
	* [Huggingface] `Simple RAG for GitHub issues using Hugging Face Zephyr and LangChain <https://huggingface.co/learn/cookbook/en/rag_zephyr_langchain>`_
	* [Huggingface] `Advanced RAG on Hugging Face documentation using LangChain <https://huggingface.co/learn/cookbook/en/advanced_rag>`_
	* [Huggingface] `RAG Evaluation <https://huggingface.co/learn/cookbook/en/rag_evaluation>`_
	* [Huggingface] `Building A RAG Ebook “Librarian” Using LlamaIndex <https://huggingface.co/learn/cookbook/en/rag_llamaindex_librarian>`_

RAG Key Paper Summary
=========================================================================================
.. note::
	* x = query
	* z = doc
	* y = output

Frozen RAG
-----------------------------------------------------------------------------------------
In-context
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. important::
	RALM

		- Retrieve k documents Z_k.
		- Rerank the docs using (1) zero-shot LM or (2) dedicated trained ranker.
		- Select top doc Z_top.
		- Prepend top doc in textual format as-is to the query as a part of the prompt for the LM to generate.
		- What we pass to the decoder: prompt with Z_top in it.
		- Issues: problematic for multiple docs (!)

In-context/Seq2Seq/Decoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. important::
	RePLUG

		- Retrieve k documents.
		- Use cosine similarity score to compute p(Z_k | X).
		- What we pass to the decoder: concat{Z_k, X} or prompt with Z_k in it.
		- Make k forward passes in the decoder for each token to compute the likelihood over vocab using softmax p(Y_i | concat{Z_k, X}, Y_1..{i-1}).
		- Rescale the softmax with p(Z_k | X) and marginalize.
		- Pass the marginalized softmax to the decoder.
		- Issues: k forward passes at each token.

Decoder Only
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. important::
	kNN-LN
	
		- For the current token consider X = encode(Y_1...Y_{i-1}).
		- Retrieve k documents Z_k matching X.
		- Make k forward passes in the decoder with the matching doc p_k(Y_i | Z_1..{i-1}).
		- Rescale p_k(Y_i | Z_1..{i-1}) over k and marginalize over the next token Y_i.
		- Do the same in the original sequence p_decode(Y_i | Z_1..{i-1}).
		- Interpolate between these using a hyperparameter.
		- Issues: k forward passes + retrieval at each token.

Retriever trainable RAG
-----------------------------------------------------------------------------------------
Seq2Seq
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. important::
	RePLUG-LSR

		- Uses the parametric LM's output to update the retriever.
		- Loss: KL div between p(Z_k | X) and the posterior p(Z_k | X, Y_1..Y_N) works well.

E2E trainable RAG
-----------------------------------------------------------------------------------------
Seq2Seq
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. important::
	* RAG

		- Per token: same as RePLUG - output probability is marginalised at the time of generation of each token, pass it to beam decoder.
		- Per sequence: output probability is marginalised for the entire sequence.

			- Results in #Y generated sequences.
			- Might require additional passes.

		- Training - NLL loss across predicted tokens.
		- Issues: E2E training makes doc index update problematic, solution: just update the query encoder.
	* Atlas

		- Multiple choice for updating the retriever - simple RePLUG-LSR type formulation based on the KL div between p(Z_k | X) and the posterior p(Z_k | X, Y_1..Y_N) works well.
		- Pre-training: same objective as the Seq2Seq (prefixLM or MLM) or decoder-only objective works well.
		- Training:
		- Issues:

Graph RAG
-----------------------------------------------------------------------------------------
.. important::
	- Baseline rag struggles
	
		- answering a question requires traversing disparate pieces of information through their shared attributes
		- holistically understand summarized semantic concepts over large data collections or even singular large documents.
	
	- Graph RAG: https://microsoft.github.io/graphrag/
	
		.. note::
			- Source documents -> Text Chunks: Note: Tradeoff P/R in chunk-size with number of LLM calls vs quality of extraction (due to lost in the middle)
			- Text Chunks -> Element Instances: 
			
				- Multipart LLM prompt for (a) Entity and then (b) Relationship. Extract descriptions as well.
				- Tailor prompt for each domain with FS example. 
				- Additional extraction covariates (e.g. events). 
				- Multiple rounds of gleaning - detect additional entities with high logit bias for yes/no. Prepend "MANY entities were missed".
			- Element Instances -> Element Summaries
			- Element Summaries -> Graph Communities
			- Graph Communities -> Community Summaries
	
				- Leaf level communities
				- Higher level communities
			- Community Summaries -> Community Answers -> Global Answer
	
				- Prepare community summaries: Shuffle and split into chunks to avoid concentration of information and therefore lost in the middle.
				- Map-Reduce community summaries
	
			- Summarisation tasks
	
				- Abstractive vs extractive
				- Generic vs query-focused
				- Single document vs multi-document
	
		- The LLM processes the entire private dataset, creating references to all entities and relationships within the source data, which are then used to create an LLM-generated knowledge graph. 
		- This graph is then used to create a bottom-up clustering that organizes the data hierarchically into semantic clusters This partitioning allows for pre-summarization of semantic concepts and themes, which aids in holistic understanding of the dataset. 
		- At query time, both of these structures are used to provide materials for the LLM context window when answering a question.	
		- Eval:
	
			- Comprehensiveness (completeness within the framing of the implied context of the question)
			- Human enfranchisement (provision of supporting source material or other contextual information)
			- Diversity (provision of differing viewpoints or angles on the question posed)
			- Selfcheckgpt

LLM vs LC
-----------------------------------------------------------------------------------------
.. important::
	- RAG FTW: Xu et al (NVDA): RETRIEVAL MEETS LONG CONTEXT LARGE LANGUAGE MODELS (Jan 2024)

		- Compares between 4k+RAG and 16k/32k LC finetuned with rope trick with 40B+ models
		- Scroll and long bench
	- LC FTW: Li et al (DM): Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach (Jul 2024)

		- Systematized the eval framework using infty-bench EN.QA (~150k) and EN.MC (~142k) and 7 datasets from long-bench (<20k)
		- 60% of the cases RAG and LC agrees (even makes the same mistakes)
		- Cases where RAG fails 

			(a) multi-hop retrieval 
			(b) general query where semantic similarity doesn't make sense 
			(c) long and complex query 
			(d) implicit query requiring a holistic view of the context
		- Key contribution: Proposes self-reflectory approach with RAG first with an option to respond "unanswerable", then LC
	- RAG FTW: Wu et al (NVDA): In Defense of RAG in the Era of Long-Context Language Models (Sep 2024)

		- Same eval method as the above
		- Key contribution: keep the chunks in the same order as they appear in the original text instead of ordering them based on sim measure

LM Eval
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
	* MMLU - `Measuring Massive Multitask Language Understanding <https://arxiv.org/pdf/2009.03300>`_
	* OpenQA - `Retrieving and Reading: A Comprehensive Survey on Open-domain Question Answering <https://arxiv.org/pdf/2101.00774>`_
	* RAGAS: `Automated Evaluation of Retrieval Augmented Generation <https://arxiv.org/abs/2309.15217>`_
	* RAGChecker: `A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation <https://arxiv.org/abs/2408.08067>`_
	* [confident.ai] `DeepEval <https://docs.confident-ai.com/docs/getting-started>`_

.. seealso::
	* `Toolformer: Language Models Can Teach Themselves to Use Tools <https://arxiv.org/pdf/2302.04761>`_

LLM and KG
=========================================================================================
.. seealso::
	* Unifying Large Language Models and Knowledge Graphs: A Roadmap
	* QA-GNN: Reasoning with Language Models and Knowledge Graphs for Question Answering
	* SimKGC: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models

KG-enhanced LLMs
-----------------------------------------------------------------------------------------
- pre-training:

	- ERNIE: Enhanced language representation with informative entities
	- Knowledge-aware language model pretraining
- inference time:

	- Retrieval-augmented generation for knowledge intensive nlp tasks
- KG for facts LLM for reasoning:

	- Language models as knowledge bases?
	- KagNet: Knowledgeaware graph networks for commonsense reasoning

LLM enhanced KGs: KG completion and KG reasoning
-----------------------------------------------------------------------------------------
- LLMs for Knowledge Graph Construction and Reasoning
- Pretrain-KGE: Learning Knowledge Representation from Pretrained Language Models
- From Discrimination to Generation: Knowledge Graph Completion with Generative Transformer

Synergized KG LLM
-----------------------------------------------------------------------------------------
- KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation
- Search: LaMDA: Language Models for Dialog Applications
- RecSys: Is chatgpt a good recommender? a preliminary study
- AI Assistant: ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation

