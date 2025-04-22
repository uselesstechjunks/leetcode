###########################################################################
Model
###########################################################################
.. contents:: Table of Contents
	:depth: 3
	:local:
	:backlinks: none

***************************************************************************
Retrieval
***************************************************************************
Two Tower
===========================================================================
Int Tower
===========================================================================
***************************************************************************
Ranking
***************************************************************************
DLRM
===========================================================================
DeepFM/DCN
===========================================================================
MTL
===========================================================================
PHASE 1: Foundational Understanding
---------------------------------------------------------------------------
#. Suppose you’re building a ranking model for a job portal. You want to predict both click and apply. Give two principled reasons why modeling these together in an MTL setup is better than training separate models.
#. You're training an MTL model for click and purchase on an ecommerce feed. The purchase signal is extremely sparse and delayed. If you remove the click task and train on purchase alone, the model fails to converge meaningfully. Mechanistically, what role did the click task play in making training work? (Be specific about gradient behavior, representation learning, and optimization landscape.)
#. MTL is often said to "regularize" a sparse task using dense tasks. What does regularization mean in this context, and why is it more effective than traditional methods like L2 or dropout when learning from sparse labels?
#. You're modeling click, like, and comment in a UGC app. You suspect comment is suffering from overfitting. It's sparse, noisy, and high variance across users. How might the click and like tasks stabilize learning for comment? Be precise about how this affects representations, gradients, and generalization.

PHASE 2: MTL Design Choices
---------------------------------------------------------------------------
Architecture Design
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
- Shared-bottom vs Task-Specific Towers
	#. Why might shared-bottom work well for a pair like click and like, but poorly for click and purchase? Explain in terms of representation bias, gradient dynamics, and task semantics.
	#. You're using a shared-bottom model for click, like, and comment. You notice that click and like metrics improve, but comment does not — even though comment is semantically close to like. Upon inspection, you find that comment gradients are small and noisy. Explain how this can happen despite semantic proximity, and outline two architectural fixes (other than PCGrad or reweighting).
- Soft parameter sharing – MMoE: Multi-Gate Mixture-of-Experts
	#. In MMoE, what kind of behaviors do experts tend to specialize in during training? Assume tasks are diverse — e.g., click, comment, share, purchase. What properties of the tasks or data influence this specialization?
	#. Give two scenarios where using MMoE might be overkill or even counterproductive compared to a simpler architecture.
	#. You’ve trained an MMoE model with 4 experts and 3 tasks: click, add-to-cart, and purchase. Post-training: click task distributes its gate weights evenly across all experts. purchase uses only Expert 2 and Expert 3. add-to-cart uses Expert 2 most, but sometimes Expert 1. What does this pattern suggest about your task distribution and expert behavior? What would you do next to interpret or optimize further?
- Entropy Regularization on Gates and Task-Conditioned Gating
	#. You’re training MMoE with 4 experts for 6 tasks. After training: One expert is never used by any task (mean gate weight ≈ 0). Two experts are heavily used by all tasks. Performance gains are marginal compared to shared-bottom. What’s going wrong? How would you fix it?
	#. Your gates produce near-uniform outputs across all experts — for all tasks. There’s no clear differentiation between expert usage. What does this suggest about your gating input or network? What architectural changes would you explore?
	#. Your setup has 5 related tasks (click, like, view, hover, share). When trained with MMoE: All gates collapse to one expert. All tasks converge, but show slight overfitting. Expert activations are indistinguishable across tasks. What’s your diagnosis? What would you change in the architecture or task grouping?

PHASE 3: Data Pipeline Decisions
---------------------------------------------------------------------------
Label Characteristics
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Sampling Strategy
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PHASE 4: Learning Dynamics and Stabilization
---------------------------------------------------------------------------
Loss Balancing Strategies
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PHASE 5: Debugging and Failure Modes
---------------------------------------------------------------------------
Symptoms and Diagnoses
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PHASE 6: Domain-Specific Considerations
---------------------------------------------------------------------------
BST
===========================================================================
