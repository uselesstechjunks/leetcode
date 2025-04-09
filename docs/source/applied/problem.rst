#####################################################################
Problem Understanding
#####################################################################
*********************************************************************
Level 1: Data
*********************************************************************
This influences whether you use supervised learning, self-supervised pretraining, weak supervision, pseudo-labeling, etc.

	- What modalities are available? (images, text, user logs)
	- Are the labels clean or noisy?
	- How many labeled examples?
	- Any weak, inferred, or behavioral signals I can use?

*********************************************************************
Level 2: Task & Output
*********************************************************************
This defines your loss function, architecture head, evaluation metrics.

	- Is it single-label, multi-label, or ranking?
	- Flat or hierarchical labels?
	- Is the output interpretable or purely latent (like embeddings)?

*********************************************************************
Level 3: System & Constraints
*********************************************************************
This decides model complexity, serving choices, retraining strategies.

	- Is inference real-time or offline?
	- Do we need to support retrieval, tagging, or classification at scale?
	- Can we retrain frequently?
	- Is personalization or user feedback part of the loop?

*********************************************************************
Examples
*********************************************************************
Manual vs Inferred Labels
=====================================================================
- Manual labels (e.g., labeller says: "this is a shoe") → High precision, good for supervised learning.
- Inferred labels (e.g., product clicked after search for "shoes"): → Noisy but abundant. May require:
	- Self-supervised pretraining
	- Positive-unlabeled learning
	- Label smoothing
	- Confidence-based sampling
- If labels are inferred, you can’t blindly fine-tune a classifier. You may overfit to noise, so you’d bring in regularization, semi-supervised learning, or label cleaning.
