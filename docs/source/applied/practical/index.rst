#######################################################################
Practical ML
#######################################################################
.. contents:: Table of Contents
   :depth: 3
   :local:
   :backlinks: none

***********************************************************************
Training
***********************************************************************
Data and Feature Engineering
=======================================================================
- Feature Selection: Mutual information, SHAP, correlation-based selection.
- Feature Transformation: Log transforms, binning, embedding representations.
- Handling Noisy Labels: Weak supervision, label smoothing, relabeling strategies.
- Dealing with Class Imbalance: SMOTE, focal loss, balanced batch sampling.
- Bias and Fairness: Bias detection, de-biasing strategies, fairness-aware training.

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
ROC-AUC and PR-AUC
=======================================================================
Model Evaluation Beyond AUC
=======================================================================
- Calibration: Platt scaling, isotonic regression.
- Expected Calibration Error (ECE): Ensuring confidence scores are well-calibrated.
- Decision Curves: Precision-recall tradeoff visualization.
- Robustness Testing: Adversarial robustness, stress testing with synthetic data.
- Error Analysis: Slice-based performance evaluation, confusion matrix breakdown.

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
