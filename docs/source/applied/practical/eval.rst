######################################################################################
Evaluation
######################################################################################
.. contents:: Table of Contents
	:depth: 3
	:local:
	:backlinks: none

***********************************************************************
Offline Evaluation
***********************************************************************
Golden Set Curation
=======================================================================
- Criteria for selection

	- Coverage: Includes all relevant feature distributions.
	- Accuracy: Labels verified by experts.
	- Diversity: Edge cases, rare conditions.
- Update frequency?
   
	- Periodically (e.g., quarterly) or when drift is detected.
- How to balance representation?

	- Maintain real-world distribution while oversampling rare cases.

Metric
=======================================================================
- ROC-AUC: Measures ability to distinguish classes across all thresholds; useful when class balance is not extreme.
- PR-AUC: Focuses on positive class performance (precision vs recall); useful when positives are rare.
- When to prefer ROC-AUC vs PR-AUC?

	- ROC-AUC: When positives and negatives are balanced.
	- PR-AUC: When positives are rare (e.g., fraud detection, rare disease prediction).

Slice-based Performance Evaluation
=======================================================================
How to choose slices for evaluation?

	- Numerical features: Quantile-based bins (e.g., age groups).
	- Categorical features: Stratify by value distribution.
	- Temporal features: Time-based slices (e.g., recent vs past).
	- Edge cases: Identify rare but critical scenarios.

When is a model ready for production?

	- Stable performance across test & validation sets.
	- Performs better than baseline (existing model or heuristic).
	- Low failure rate in stress tests (edge cases, adversarial inputs).

Model Evaluation Beyond AUC
=======================================================================
- Calibration: Platt scaling, isotonic regression.
- Expected Calibration Error (ECE): Ensuring confidence scores are well-calibrated.
- Robustness Testing: Adversarial robustness, stress testing with synthetic data.

***********************************************************************
Online Evaluation
***********************************************************************
- A/B Testing
- Interleaving vs non-interleaving

**************************************************************************************
LLM App Evaluation
**************************************************************************************
Practical
=========================================================================================
* [github.com] `The LLM Evaluation guidebook <https://github.com/huggingface/evaluation-guidebook>`_
* [confident.ai] `LLM Evaluation Metrics: The Ultimate LLM Evaluation Guide <https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation>`_
* [confident-ai.com] `How to Evaluate LLM Applications: The Complete Guide <https://www.confident-ai.com/blog/how-to-evaluate-llm-applications>`_
* [arize.com] `The Definitive Guide to LLM App Evaluation <https://arize.com/llm-evaluation/overview/>`_
* [arize.com] `RAG Evaluation <https://arize.com/blog-course/rag-evaluation/>`_
* [guardrailsai.com] `Guardrails AI Docs <https://www.guardrailsai.com/docs>`_

Academic
=========================================================================================
* [acm.org] `A Survey on Evaluation of Large Language Models <https://dl.acm.org/doi/pdf/10.1145/3641289>`_
* [arxiv.org] `The Responsible Foundation Model Development Cheatsheet: A Review of Tools & Resources <https://arxiv.org/abs/2406.16746>`_
* [arxiv.org] `Retrieving and Reading: A Comprehensive Survey on Open-domain Question Answering <https://arxiv.org/pdf/2101.00774>`_
* Evaluation of instruction tuned/pre-trained models

	* MMLU

		* [arxiv.org] `Measuring Massive Multitask Language Understanding <https://arxiv.org/pdf/2009.03300>`_
		* Dataset: https://huggingface.co/datasets/cais/mmlu
	* Big-Bench

		* [arxiv.org] `Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models <https://arxiv.org/pdf/2206.04615>`_
		* Dataset: https://github.com/google/BIG-bench
