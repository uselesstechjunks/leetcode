#######################################################################
Overview of ML Design Choices
#######################################################################
.. contents:: Table of Contents
   :depth: 3
   :local:
   :backlinks: none

***********************************************************************
Training
***********************************************************************
Large-Scale ML & Distributed Training
=======================================================================
- Parallelization: Data parallelism vs model parallelism.
- Gradient Accumulation: Handling large batch sizes.
- Federated Learning: Privacy-preserving distributed learning.
- ML Monitoring & Logging: Model drift detection, feature monitoring, data pipelines.
- Serving at Scale: TFX, Ray Serve, TorchServe, Kubernetes-based deployments.

Fine-Tuning & LLMs
=======================================================================
- Efficient Fine-Tuning: LoRA, QLoRA, adapters, prompt tuning.
- Memory-Efficient Training: Flash Attention, ZeRO Offloading, activation checkpointing.
- Inference Optimization: KV caching, speculative decoding, grouped-query attention.
- Long-Context Adaptation: RoPE interpolation, Hyena operators, recurrent memory transformers.
- Safety & Alignment: RLHF, constitutional AI, preference tuning.

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
Deployment
***********************************************************************
Model Productionization & Scaling
=======================================================================
- Latency vs Accuracy Tradeoffs: Quantization, distillation, pruning.
- Efficient Inference: TensorRT, ONNX, model sharding, mixed precision training.
- Retraining Strategies: Online learning, active learning, incremental updates.
- Data Drift and Concept Drift: Detection techniques, adaptive retraining pipelines.
- A/B Testing and Shadow Deployment: Canary rollouts, offline vs online evaluation.

Applied Causal Inference & Uplift Modeling
=======================================================================
- Causal ML in Production: A/B testing pitfalls, Simpson's paradox.
- Uplift Modeling: Net lift estimation for interventions.
- DoWhy & Causal Discovery: Counterfactual analysis in ML pipelines.

Retraining
=======================================================================
#. How often to retrain?
   
   - Depends on drift: Frequent updates if data shifts, otherwise periodic (weekly, monthly, quarterly).
#. Periodic vs Continuous Training?

   - Periodic: Easier to manage, avoids instability.
   - Continuous: Needed when real-time adaptation is required (e.g., dynamic pricing, recommendation systems).
