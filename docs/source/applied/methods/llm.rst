#########################################################################################
Large Language Models
#########################################################################################
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

*****************************************************************************************
Training
*****************************************************************************************
Practical
=========================================================================================
Scaling Large Models
-----------------------------------------------------------------------------------------
.. important::
	* [github.io] `How To Scale Your Model <https://jax-ml.github.io/scaling-book/index>`_

Data Engineering
-----------------------------------------------------------------------------------------
.. important::
	* [github.com] `LLMDataHub: Awesome Datasets for LLM Training <https://github.com/Zjh-819/LLMDataHub>`_
	* [arxiv.org] `The Pile: An 800GB Dataset of Diverse Text for Language Modeling <https://arxiv.org/abs/2101.00027>`_	

Hardware Utilisation
-----------------------------------------------------------------------------------------
.. important::
	* [horace.io] `Making Deep Learning Go Brrrr From First Principles <https://horace.io/brrr_intro.html>`_
	* [newsletter.maartengrootendorst.com] `A Visual Guide to Quantization <https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization>`_
	* [nvidia.com] `Profiling PyTorch Models for NVIDIA GPUs <https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31644/>`_
	* [pytorch.org] `What Every User Should Know About Mixed Precision Training in PyTorch <https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/>`_
	* [pytorch.org] `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_
	* [arxiv.org] `Hardware Acceleration of LLMs: A comprehensive survey and comparison <https://arxiv.org/pdf/2409.03384>`_

Pipelines
-----------------------------------------------------------------------------------------
.. important::
	* [huggingface] `LLM Inference at scale with TGI <https://huggingface.co/blog/martinigoyanes/llm-inference-at-scale-with-tgi>`_
	* [vLLM] `Easy, Fast, and Cheap LLM Serving with PagedAttention <https://blog.vllm.ai/2023/06/20/vllm.html>`_
	* [HuggingFace Blog] `Fine-tuning LLMs to 1.58bit: extreme quantization made easy <https://huggingface.co/blog/1_58_llm_extreme_quantization>`_
	* [Paper] `Data Movement Is All You Need: A Case Study on Optimizing Transformers <https://arxiv.org/abs/2007.00072>`_

Tools
-----------------------------------------------------------------------------------------
.. important::
	* [pytorch.org] `PyTorch Profiler <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`_
	* [tinkerd.net] `Writing CUDA Kernels for PyTorch <https://tinkerd.net/blog/machine-learning/cuda-basics/>`_
	* [spaCy] `Library for NLU/IE Tasks <https://spacy.io/usage/spacy-101>`_, `LLM-variants <https://spacy.io/usage/large-language-models>`_
	* [tinkerd.net] `Distributed Training and DeepSpeed <https://tinkerd.net/blog/machine-learning/distributed-training/>`_

Evaluation
-----------------------------------------------------------------------------------------
.. important::
	* [confident.ai] `LLM Evaluation Metrics: The Ultimate LLM Evaluation Guide <https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation>`_
	* [guardrailsai.com] `Guardrails AI Docs <https://www.guardrailsai.com/docs>`_
	* [arxiv.org] `The Responsible Foundation Model Development Cheatsheet: A Review of Tools & Resources <https://arxiv.org/abs/2406.16746>`_

Pretraining
=========================================================================================
.. note::
	* Improving Language Understanding by Generative Pre-Training
	* Universal Language Model Fine-tuning for Text Classification

Domain-Adaptation
=========================================================================================
SoDA

Supervised Fine-Tuning (SFT)
=========================================================================================
Reinforcement Learning with Human Feedback (RLHF)
========================================================================================
Direct Preference Optimisation (DPO)
=========================================================================================
Reinforcement Fine-Tuning (RFT)
=========================================================================================
.. note::
	* [philschmid.de] `Bite: How Deepseek R1 was trained <https://www.philschmid.de/deepseek-r1>`_
	* [arxiv.org] `DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models <https://arxiv.org/abs/2402.03300>`_
	* [predibase.com] `How Reinforcement Learning Beats Supervised Fine-Tuning When Data is Scarce <https://predibase.com/blog/how-reinforcement-learning-beats-supervised-fine-tuning-when-data-is-scarce>`_

*****************************************************************************************
Special Techniques
*****************************************************************************************
Low-Rank Approximations (LoRA)
=========================================================================================
.. note::
	* [tinkerd.net] `Language Model Fine-Tuning with LoRA <https://tinkerd.net/blog/machine-learning/lora/>`_

MoE
=========================================================================================
.. note::
	* [tinkerd.net] `Mixture of Experts Pattern for Transformer Models <https://tinkerd.net/blog/machine-learning/mixture-of-experts/>`_
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

Pruning
-----------------------------------------------------------------------------------------
* LazyLLM: Dynamic Token Pruning for Efficient Long Context LLM Inference

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

Prompt best guide
-----------------------------------------------------------------------------------------
Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine

	- Zero-shot
	- Random few-shot
	- Random few-shot, chain-of-thought
	- kNN, few-shot, chain-of-though
	- Ensemble w/ choice shuffle

Logit Bias
-----------------------------------------------------------------------------------------
A logit bias can be used to influence the output probabilities of a language model (LLM) to steer it towards a desired output, such as a "yes" or "no" answer. Here's how it works:

What is Logit Bias?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the context of language models, logits are the raw, unnormalized scores that a model outputs before applying the softmax function to obtain probabilities. Logit bias refers to the adjustment of these logits to favor or disfavor certain tokens.

How Logit Bias Works
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Logit Adjustment:
   - Each token in the vocabulary has an associated logit value.
   - By adding a bias to the logits of specific tokens, you can increase or decrease the likelihood that those tokens will be selected when the model generates text.

2. Softmax Function:
   - After adjusting the logits, the softmax function is applied to convert these logits into probabilities.
   - Tokens with higher logits will have higher probabilities of being selected.

Forcing a Yes/No Answer with Logit Bias

To force an LLM into a yes/no answer, you can adjust the logits for the "yes" and "no" tokens.

Steps to Apply Logit Bias
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Identify Token IDs:

   - Determine the token IDs for "yes" and "no" in the model's vocabulary. For instance, suppose "yes" is token ID 345 and "no" is token ID 678.

2. Apply Bias:

   - Adjust the logits for these tokens. Typically, you would add a positive bias to both "yes" and "no" tokens to increase their probabilities and/or subtract a bias from all other tokens to decrease their probabilities.

3. Implementing the Bias:

   - If using an API or library that supports logit bias (e.g., OpenAI GPT-3), you can specify the bias directly in the request.

Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here's an example of how you might apply a logit bias in a request using a hypothetical API:

.. code-block:: json

	{
	  "prompt": "Is the sky blue?",
	  "logit_bias": {
		"345": 10,  // Bias for "yes"
		"678": 10   // Bias for "no"
	  }
	}

Practical Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Magnitude of Bias:

   - The magnitude of the bias determines how strongly the model will favor "yes" or "no." A larger bias will make the model more likely to choose these tokens.

2. Context Sensitivity:

   - The model may still consider the context of the prompt. If the context strongly indicates one answer over the other, the model may lean towards that answer even with a bias.

3. Balanced Bias:

   - If you want the model to have an equal chance of saying "yes" or "no," you can apply equal positive biases to both tokens. If you want to skew the response towards one answer, apply a larger bias to that token.

Example in Practice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Consider a scenario where you want the model to respond with "yes" or "no" to the question "Is the sky blue?"

.. code-block:: text

	- Prompt: "Is the sky blue?"
	- Logit Bias:
	  - Yes token (ID 345): +10
	  - No token (ID 678): +10

This setup ensures that the model will highly favor "yes" and "no" as possible outputs. The prompt and biases are designed so that "yes" or "no" are the most likely completions.

API Implementation Example (Pseudo-Code)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here's a pseudo-code example of how you might implement this with an API:

.. code-block:: python

	import openai

	response = openai.Completion.create(
	  engine="text-davinci-003",
	  prompt="Is the sky blue?",
	  max_tokens=1,
	  logit_bias={"345": 10, "678": 10}
	)

	print(response.choices[0].text.strip())

In this example:
- The `prompt` is set to "Is the sky blue?"
- The `logit_bias` dictionary adjusts the logits for the "yes" and "no" tokens to be higher.
- The `max_tokens` is set to 1 to ensure only one word is generated.
- By using logit bias in this way, you can guide the LLM to produce a "yes" or "no" answer more reliably.

Issues with LLMs
-----------------------------------------------------------------------------------------
	- hallucination 
		- detection and mitigation
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
	- alignment and preference
		- rlhf
		- dpo
		- reflexion

TODO
-----------------------------------------------------------------------------------------
- constitutional ai
- guardrails
- https://github.com/microsoft/unilm
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
- training: domain adaptation (mlm/rtd/ssl-kl/clm), finetuning (sft/it), alignment and preference optim (rhlf/dpo)
- domain understanding
- design e2e: integrate user feedback

Resources
=========================================================================================
.. note::
	* `OpenAI Docs <https://platform.openai.com/docs/overview>`_
	* `[HN] You probably don’t need to fine-tune an LLM <https://news.ycombinator.com/item?id=37174850>`_
	* `[Ask HN] Most efficient way to fine-tune an LLM in 2024? <https://news.ycombinator.com/item?id=39934480>`_
	* `[HN] Finetuning Large Language Models <https://news.ycombinator.com/item?id=35666201>`_

		* `[magazine.sebastianraschka.com] Finetuning Large Language Models <https://magazine.sebastianraschka.com/p/finetuning-large-language-models>`_
	* `[Github] LLM Course <https://github.com/mlabonne/llm-course>`_

*****************************************************************************************
Applied LLMs
*****************************************************************************************
In Context Learning (ICL)
=========================================================================================
.. note::
	* [aclanthology.org] `Diverse Demonstrations Improve In-context Compositional Generalization <https://aclanthology.org/2023.acl-long.78.pdf>`_

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
.. attention::
	* [Stanford Lecture] `Stanford CS25: V3 I Retrieval Augmented Language Models <https://www.youtube.com/watch?v=mE7IDf2SmJg>`_
	* [arxiv.org] `Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG <https://arxiv.org/abs/2501.09136>`_

.. note::
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
	* `Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning <https://arxiv.org/abs/2501.15228>`_
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

