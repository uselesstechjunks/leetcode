###################################################################################
Causal Inference
###################################################################################

***********************************************************************************
Explanatory Models
***********************************************************************************
* Explanatory modeling aims to understand relationships within data and explain why certain outcomes occur. 
* It focuses on identifying causal relationships and understanding the underlying mechanisms.
* Methodology: 

   * Models are evaluated based on their interpretability and how well they explain the relationships between variables. 
   * Statistical significance of features and coefficients is often considered.
* Techniques: 

   * Linear models (linear regression, generalized linear models)
   * Structural equation modeling (SEM)
   * Bayesian networks, and 
   * Certain types of decision trees can be used.
* Feature Selection: Features are selected based on their ability to explain the phenomenon being studied. Domain knowledge and causal inference play a significant role.

***********************************************************************************
Causal Inference
***********************************************************************************
Resources
===================================================================================
* [Blog] `Causal Inference for The Brave and True <https://matheusfacure.github.io/python-causality-handbook/landing-page.html>`_
* [Blog] `ML beyond Curve Fitting: An Intro to Causal Inference and do-Calculus <https://www.inference.vc/untitled/>`_

    * Note: This blog has some excellent post on other topics as well. Do check them out. 
    * Example: https://www.inference.vc/variational-inference-with-implicit-probabilistic-models-part-1-2/
* [UCL] `Arthur Gretton's Course <https://www.gatsby.ucl.ac.uk/~gretton/coursefiles/columbiaCourse23.html>`_
* STATS 361: Causal Inference 

    - `Notes <https://web.stanford.edu/~swager/stats361.pdf>`_
* ECON 293 / MGTECON 634: Machine Learning and Causal Inference

    - `Lecture Videos <https://www.youtube.com/playlist?list=PLxq_lXOUlvQAoWZEqhRqHNezS30lI49G->`_

* [CMU 36-708] `Statistical Methods for Machine Learning <https://www.stat.cmu.edu/~larry/=sml/>`_; `Causation <https://www.stat.cmu.edu/~larry/=sml/Causation.pdf>`_
* [MIT 6.S897] `Machine Learning for Healthcare <https://youtube.com/playlist?list=PLUl4u3cNGP60B0PQXVQyGNdCyCTDU1Q5j&si=FHRX57NhPGrayv8D>`_

    * [Lecture Notes and Papers] `Causal Inference and Reinforcement Learning <https://mlhc19mit.github.io/>`_
* [Larry Wasserman] `Articles on arXiv <https://arxiv.org/a/wasserman_l_1.html>`_
* [Larry Wasserman] `Blog: Normal Deviate - Causation <https://normaldeviate.wordpress.com/2012/06/18/48/>`_
* [CMU] `Machine Learning for Structured Data <https://www.cs.cmu.edu/~mgormley/courses/10418/schedule.html>`_
* [Workshop] `Bayesian Causal Inference <https://bcirwis2021.github.io/index.html>`_

Papers
-----------------------------------------------------------------------------------
* `Uri Shalit papers <https://scholar.google.com/citations?user=aeGDj-IAAAAJ&hl=en>`_
* `MEMENTO: Neural Model for Estimating Individual Treatment Effects for Multiple Treatments <https://dl.acm.org/doi/pdf/10.1145/3511808.3557125>`_

Course Plan
===================================================================================
1. Causal Inference Basics:
-----------------------------------------------------------------------------------
   - *Concepts*: Understand fundamental concepts such as causality, counterfactuals, treatment effects, and causal graphs (Directed Acyclic Graphs - DAGs).
   - *Techniques*: Familiarize yourself with techniques like matching methods, instrumental variables, difference-in-differences, and regression discontinuity designs.
   - Papers

        * Instrumental Variables in Causal Inference and Machine Learning: A Survey
        * Matching Methods for Causal Inference: A Review and a Look Forward
        * Regression discontinuity design

Resources for Causal Inference:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   - *Books*:

     - "Causal Inference: The Mixtape" by Scott Cunningham (freely available online).
     - "Mostly Harmless Econometrics: An Empiricist's Companion" by Joshua D. Angrist and Jörn-Steffen Pischke.
   - *Courses*:

     - Coursera offers courses like "Causal Inference" by University of Pennsylvania and "Econometrics: Methods and Applications" by Erasmus University Rotterdam.
     - edX has courses such as "Causal Diagrams: Draw Your Assumptions Before Your Conclusions" by Harvard University.
   - *Online Resources*:

     - Blogs and tutorials from platforms like Towards Data Science and Medium often have introductory articles on causal inference.

2. Advanced Regression Techniques:
-----------------------------------------------------------------------------------
   - *Topics*: Brush up on advanced regression methods that are commonly used for explanatory modeling, such as generalized linear models (GLMs), hierarchical linear models (HLMs), and Bayesian regression.

Resources for Advanced Regression:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   - *Books*:

     - "Bayesian Data Analysis" by Andrew Gelman et al.
     - "Regression Modeling Strategies" by Frank E. Harrell Jr.
   - *Courses*:

     - Platforms like Coursera and edX offer courses on Bayesian statistics and regression modeling.

3. Interpretable Machine Learning:
-----------------------------------------------------------------------------------
   - *Techniques*: Explore methods that enhance model interpretability, such as feature importance techniques (e.g., SHAP values, permutation importance), partial dependence plots, and model-agnostic approaches (e.g., LIME).

Resources for Interpretable Machine Learning:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   - *Books*:

     - "Interpretable Machine Learning" by Christoph Molnar.
   - *Courses*:

     - Look for courses on interpretable machine learning on platforms like Coursera or edX.

4. Domain Knowledge in Supply Chain and Inventory Management:
-----------------------------------------------------------------------------------
   - *Understand*: Review key concepts in inventory management, supply chain operations, and optimization techniques commonly used in e-commerce and retail.

Additional Tips:
-----------------------------------------------------------------------------------
- *Practice Problem Solving*: Solve case studies or practice problems related to inventory management and causal analysis.
- *Mock Interviews*: Conduct mock interviews with peers or mentors to practice explaining your approach to developing explanatory models.

Example Scenario to Explore:
-----------------------------------------------------------------------------------
- Consider how you would design a study to determine the causal impact of a new inventory management policy on key performance metrics (e.g., cost efficiency, inventory turnover) using causal inference techniques.
