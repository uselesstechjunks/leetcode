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
Feature Transformation
-----------------------------------------------------------------------
Scaling and Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Standardization

   - Transform features to have a mean of 0 and standard deviation of 1. 
   - Applicable to: Continuous numerical variables.
- Min-Max Scaling

   - Rescale features to a fixed range (e.g., [0, 1]). 
   - Applicable to: Continuous numerical variables.
- Robust Scaling

   - Use the median and interquartile range (IQR) to scale, robust to outliers. 
   - Applicable to: Continuous numerical variables, especially with outliers.

Log Transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Apply logarithmic transformation to reduce skewness in data with large values. 
- Applicable to: Continuous numerical variables with positive skew (e.g., income, population).

Binning and Discretization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Convert continuous variables into categorical bins (e.g., age groups). 
- Applicable to: Continuous numerical variables.

One-Hot Encoding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Convert categorical variables into binary vectors. 
- Applicable to: Categorical variables (nominal).

Ordinal Encoding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Assign integer values to ordered categories. 
- Applicable to: Ordinal categorical variables.

Polynomial Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Generate polynomial and interaction features to capture non-linear relationships. 
- Applicable to: Continuous numerical variables.

Handling Missing Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Impute missing values using mean, median, or more sophisticated methods like KNN or model-based imputation.
- Applicable to: Any type of variable with missing data (both continuous and categorical).

Label Design
-----------------------------------------------------------------------
Label Encoding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Convert categorical labels into numerical format (typically used in classification).
- Applicable to: Categorical labels (nominal or ordinal).

Log Transformation (for regression)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Apply a log transformation to skewed labels for regression tasks (e.g., predicting income, house prices).
- Applicable to: Continuous numerical labels.

Binarization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Convert continuous labels into binary values (e.g., thresholding for classification).
- Applicable to: Continuous labels for binary classification.

Less Labels
-----------------------------------------------------------------------
Weak Supervision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* [medium.com] `Weak Supervision — Learn From Less Information <https://npogeant.medium.com/weak-supervision-learn-from-less-information-dcc8fe54e2a5>`_
* [stanford.edu] `Weak Supervision: A New Programming Paradigm for Machine Learning <https://ai.stanford.edu/blog/weak-supervision/>`_

Semi Supervised Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* [maddevs.io] `Semi-Supervised Learning Explained: Techniques and Real-World Applications <https://maddevs.io/blog/semi-supervised-learning-explained/>`_
* [ruder.io] `An overview of proxy-label approaches for semi-supervised learning <https://www.ruder.io/semi-supervised/>`_

Noisy Labels
-----------------------------------------------------------------------
Label Smoothing 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Instead of using hard labels (e.g., 0 or 1), use smoothed labels (e.g., 0.9 and 0.1) to make the model more robust to noisy labels.

Noise Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Human-in-the-loop Use human feedback to verify or correct labels in the dataset.
Confidence-based Filtering Remove samples with low model confidence or high disagreement between multiple annotators.

Outlier Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Apply algorithms (e.g., Isolation Forest, Z-score method) to detect outliers in the dataset and remove instances with highly suspicious labels.

[TODO]
-----------------------------------------------------------------------
- Feature Selection: Mutual information, SHAP, correlation-based selection.
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
