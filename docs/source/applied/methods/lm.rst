#########################################################################################
Language Understanding and Language Models
#########################################################################################
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

*****************************************************************************************
Activations
*****************************************************************************************
.. note::
	* [SiLU] `Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_
	* [GELU] `Gaussian Error Linear Units <https://arxiv.org/abs/1606.08415>`_
	* [Swish] `Searching for Activation Functions <https://arxiv.org/pdf/1710.05941v2>`_	
	* [Swish v GELU] `On the Disparity Between Swish and GELU <https://towardsdatascience.com/on-the-disparity-between-swish-and-gelu-1ddde902d64b>`_
	* [GLU] `GLU Variants Improve Transformer <https://arxiv.org/pdf/2002.05202v1>`_

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
General Resources
=========================================================================================
.. warning::
	* [github.com] `LLM101n: Let's build a Storyteller <https://github.com/karpathy/LLM101n>`_
	* [jmlr.org] `Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity <https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf>`_
	* [epoch.ai] `How has DeepSeek improved the Transformer architecture? <https://epoch.ai/gradient-updates/how-has-deepseek-improved-the-transformer-architecture>`_

.. note::
	* [harvard.edu] `The Annotated Transformer <https://nlp.seas.harvard.edu/annotated-transformer/>`_
	* [jalammar.github.io] `The Illustrated Transformer <https://jalammar.github.io/illustrated-transformer/>`_
	* [lilianweng.github.io] `Attention? Attention! <https://lilianweng.github.io/posts/2018-06-24-attention/>`_
	* [d2l.ai] `The Transformer Architecture <https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html>`_
	* [newsletter.languagemodels.co] `The Illustrated DeepSeek-R1: A recipe for reasoning LLMs <https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1>`_

Position Encoding
=========================================================================================
.. note::
	* [arxiv.org] `Position Information in Transformers: An Overview <https://arxiv.org/abs/2102.11090>`_
	* [arxiv.org] `Rethinking Positional Encoding in Language Pre-training <https://arxiv.org/abs/2006.15595>`_
	* [eleuther.ai] `RoPE <https://blog.eleuther.ai/rotary-embeddings/>`_
	* [arxiv.org] `LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens <https://arxiv.org/abs/2402.13753>`_
	* [arxiv.org] `RoFormer: Enhanced Transformer with Rotary Position Embedding <https://arxiv.org/abs/2104.09864>`_

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
* [mlabonne.github.io] `Decoding Strategies in Large Language Models <https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html>`_
* Speculative Deocding

*****************************************************************************************
Transformer Architecture
*****************************************************************************************
Encoder [BERT]
=========================================================================================
.. note::
	* BERT: `Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`_
	* Additional Resources

		* [tinkerd.net] `BERT Tokenization <https://tinkerd.net/blog/machine-learning/bert-tokenization/>`_
		* [tinkerd.net] `BERT Embeddings <https://tinkerd.net/blog/machine-learning/bert-embeddings/>`_, 
		* [tinkerd.net] `BERT Encoder Layer <https://tinkerd.net/blog/machine-learning/bert-encoder/>`_
	* `A Primer in BERTology: What we know about how BERT works <https://arxiv.org/abs/2002.12327>`_
	* RoBERTa: `A Robustly Optimized BERT Pretraining Approach <https://arxiv.org/abs/1907.11692>`_
	* XLM: `Cross-lingual Language Model Pretraining <https://arxiv.org/abs/1901.07291>`_
	* TwinBERT: `Distilling Knowledge to Twin-Structured BERT Models for Efficient Retrieval <https://arxiv.org/abs/2002.06275>`_

Decoder [GPT]
=========================================================================================
.. note::
	* [jalammar.github.io] `The Illustrated GPT-2 <https://jalammar.github.io/illustrated-gpt2/>`_
	* [github.com] `karpathy/nanoGPT <https://github.com/karpathy/nanoGPT>`_
	* [cameronrwolfe.substack.com] `Decoder-Only Transformers: The Workhorse of Generative LLMs <https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse>`_
	* [openai.com] `GPT-2: Language Models are Unsupervised Multitask Learners <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>`_
	* [openai.com] `GPT-3: Language Models are Few-Shot Learners <https://arxiv.org/abs/2005.14165>`_

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
