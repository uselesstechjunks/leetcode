##########################################################################
Amazon
##########################################################################
**************************************************************************
Understanding the Product
**************************************************************************
Homepage
==========================================================================
Non-logged in users
--------------------------------------------------------------------------
UX Design
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PCT:

.. note::
	* Icon image: One per page
	* Segments: Interleaved

		* Mixture of categories: list of tiles, each tile with 4 or less products (depends on thumbnail size)
		* Category specific: list of horizontally scrollable product
Mobile:

.. note::
	* TODO

UX Layout:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
	* Top: Icon image of a product
	* List: Each contains a Segment Title (usually split by a title-separator) and a set of images
	
		* Mixture of categories [4 or less product listsings per category block, scrollable row]
		* Category specific [1 product listing at each row, scrollable row]
		* Category specific
	* Segment 2

		* Mixture of categories
		* Category specific
		* Category specific
	* Segment 3
	
		* Mixture of categories
		* Category specific
		* Category specific
	* Segment 4

		* Mixture of categories
		* Category specific
	
Logged in users
---------------------------------------------------------------------------

Product Page
==========================================================================

**************************************************************************
Important Topics
**************************************************************************
.. warning::
	* Design query embeddings for ads - downstream task: ranking, classification, automted ad generation
	
		* Design a query-keyword/query-product matching algorithm from scratch
		* Design a system for dynamic ad generation system based on user query
		* Finetune using ratings, reviews, purchase data
	* Design listing embeddings
	
		* Use them in recommendation system (currently visiting a product, suggest new products)
		* Finetune using ratings, reviews, purchase data
	* Create a home-page recommendation for non-logged in users (we know geolocation, time-of-the-day)
	* Design user embeddings
	
		* Create a personalized home-page recommendation ("suggested items for you" page, without query - based on user history)

**************************************************************************
Sample Questions
**************************************************************************
Shared by Recruiter
==========================================================================
ML Breadth
--------------------------------------------------------------------------
Expectation: Candidates should demonstrate a solid understanding of standard methods relevant to their scientific field. A good measure of suitable breadth includes the ability to discuss concepts/methods commonly covered in relevant graduate-level university courses and apply these methods to construct a functional, scalable system. 

Additionally, familiarity with concepts such as experimental design, system evaluation, and optimal decision making across various scientific domains is important. The evaluation process can incorporate the following approaches:

Methods Survey: An assessment of the candidate's knowledge of techniques includes:

- How do you identify and address overfitting?
- Can you develop a query embedding for Amazon teams?
- Explain ensemble algorithms (e.g., Random forest; handling features and data; reducing variance).
- What methods can be used to split a decision tree?
- Which metrics would you utilize in a classification problem?
- How do you handle imbalanced datasets?
- What loss function is suitable for measuring multi-label problems?
- Suppose you need to determine a threshold for a classifier predicting customer sign-up for Prime. What criteria could be used to determine this threshold?
- In a model with one billion positive samples and 200,000 negative samples, what would you examine to ensure its quality before deployment?
- Describe the training process for a Context-awareness entity ranking model.

ML Depth
---------------------------------------------------------------------------
Expectation: Candidates are expected to exhibit mastery in their specific area of expertise, preferably assessed by a recognized authority in the field. They should demonstrate the ability to discern methodological trade-offs, contextualize solutions within both classical and contemporary research, and possess familiarity with the nuanced skill of devising solutions within their domain. Ideally, they would have a track record of publications in their field. The assessment process should delve into the following aspects:

- Methods: Candidates should provide detailed insights into the methodologies employed in their research and projects, including rationale for their choices (such as highlighting strengths and weaknesses of methods and justifying their selection).
- Innovation vs Practicality: Assessment should explore candidates' past projects to gauge their level of creativity and pragmatism.
- Deep Dives: Evaluation should examine whether candidates delved deeply into projects where relevant, such as investigating outliers, misclassified examples, and edge cases.
- Model Evaluation: Candidates should elaborate on how they evaluated their models, including rationale behind specific trade-offs and methods used to identify key model dynamics.
- Fundamentals: Assessment should cover candidates' understanding of the fundamental principles in their field.

Scrapped from the Internet
==========================================================================
Data Preprocessing and Handling:
--------------------------------------------------------------------------
1. How would you handle missing or corrupted data in a dataset?
2. How would you find thresholds for a classifier?
3. What are some ways to split a tree in a decision tree algorithm?
4. How does pruning work in Decision Trees?
5. What methods would you employ to forecast sales figures for Samsung phones?

Supervised Learning:
--------------------------------------------------------------------------
1. State the applications of supervised machine learning in modern businesses.
2. How will you determine which machine learning algorithm to use for a classification problem?
3. How does the Amazon recommendation engine work when recommending other things to buy?
4. Differentiate between logistic regression and support vector machines.
5. Give an example of using logistic regression over SVM and vice versa.
6. What does the F1 score represent?
7. How do the results change if we use logistic regression over the decision tree in a random forest?
8. Describe linear regression vs. logistic regression.
9. How would you define log loss in the context of model evaluation?
10. Could you discuss the key assumptions that govern linear regression models and explain the significance of taking these assumptions into account when interpreting statistical results?

Ensemble Learning:
--------------------------------------------------------------------------
1. Explain the ensemble learning technique in machine learning.
2. Differentiate between bagging and boosting.
3. What distinguishes the model performance between bagging and boosting?
4. Can you elaborate on how gradient boost is used in machine learning and how it works?
5. How does the assumption of error in linear regression influence the accuracy of our models, and what does it entail?
6. How do you perceive the role of DMatrix in XGBoost, and how does it differ from other gradient boosting data structures?

Clustering and Dimensionality Reduction:
--------------------------------------------------------------------------
1. How is KNN different from K-means clustering?
2. Explain the K-means and K Nearest Neighbor algorithms and differentiate between them.
3. How are PCA with a polynomial kernel and a single layer autoencoder related?
4. Differentiate between Lasso and Ridge regression.
5. Explain ICA, CCA, and PCA.
6. State some ways of reducing dimensionality.
7. How would you get a CCA objective function from PCA?

Model Evaluation and Performance:
--------------------------------------------------------------------------
1. Considering that you already have labeled data for your clustering project, what are some of the methods that you can use to evaluate model performance?
2. What does an ROC curve tell you about a model’s performance?
3. Could you define the concepts of overfitting and underfitting in machine learning, and explain their relevance in model development?

Deep Learning and Neural Networks:
--------------------------------------------------------------------------
1. Can you elaborate on what an attention model entails?
2. Can you differentiate between batch normalization and instance normalization and their respective uses?
3. Can you walk me through the functioning of a 1D CNN?
4. Can you describe the difference in application between RNNs and LSTMs?

Miscellaneous:
--------------------------------------------------------------------------
1. Design an Email Spam Filter.
2. What steps would you take to ensure a scalable, efficient architecture for Bing’s image search system?
3. How can you perform a dot product operation on two sparse matrices?
4. Walk me through a Monte Carlo simulation to estimate Pi.

**************************************************************************
Interview Experience (Scrapped from the Internet)
**************************************************************************
Science Breadth
==========================================================================
In the ML Breadth round, the focus was on assessing the depth of my understanding across machine learning concepts. I encountered a mix of theoretical questions and practical scenarios related to applied science at Amazon. It tested my ability to grasp a broad spectrum of ML topics, showcasing the importance of a well-rounded foundation in machine learning. This would include topics in supervised and unsupervised learning 

.. note::
	* KNN, logistic regression, SVM, Naive Bayes, Decision Trees, Random Forests, Ensemble Models, Boosting, 
	* Regression, Clustering, Dimensionality Reduction
	* Feature Engineering, Overfitting, Regularization, best practices for hyperparameter tuning, Evaluation metrics
	* Neural Networks, RNNs, CNNs, Transformers.

Science Depth
==========================================================================
The Science Depth segment involved a resume deep dive, where detailed questions probed into my past work experiences. This round aimed to uncover the depth of my expertise in specific areas, emphasizing the practical application of my knowledge. This would entail understanding the tradeoffs made during the project, the different design decisions, results and impact on the organization and understanding how successful was the project at solving the problem at hand using business metrics if required. Nitty gritty details of implementation are enquired during the interview and its important to take a look at past projects and know every little detail of it and study its impact.

Science Application
==========================================================================
The Machine Learning Case Study in the domain of the job role provided a practical challenge to assess my ability to apply theoretical knowledge to real-world scenarios. This segment gauged my problem-solving skills within the context of the job, giving me an opportunity to showcase my ability to translate theoretical concepts into actionable solutions. This would entail first understanding the business problem, and then methodically come up with steps for problem formulation and a solid reason to go for a machine learning based solution. The next part would be to come up with the data collection, feature engineering and talk about the different machine learning models and finally talk about evaluation metrics, training strategies and understanding the business metric and A/B testing the model to understand feasibility for replacing the existing model.

Leadership Principles
==========================================================================
The Behavioral Style questions in the Leadership Principles round were designed to evaluate my alignment with Amazon’s core leadership principles. Through scenarios drawn from my past work experiences, I was assessed for various leadership skills. This round, often conducted by a bar raiser, held significant importance in determining my suitability for the role, underscoring Amazon’s commitment to strong leadership qualities. A strong emphasis is given on the STAR format — Situation, Task, Action and Result hence it’s very important to follow this structure when answering any scenario based question.

Coding
==========================================================================
The Coding segment comprised LeetCode-style Data Structures and Algorithms questions. This component tested my coding proficiency and problem-solving abilities. Topics would include 

.. note::
	* Data Structures
		* Arrays, Hash maps, Graphs, Trees, Heaps, Linked List, Stack, Queue
	* Algorithms
		* Binary Search, Sliding Window, Two Pointer, Backtracking, Recursion, Dynamic Programming, Greedy. 
	* Data Manipulation libraries
		* Pandas and SQL.
	* Coding concepts from Machine Learning, Probability and Statistics.

Tech Talk
==========================================================================
An intriguing component of the interview process was the Tech Talk, a platform for me to showcase one of my previous projects. This session involved a 45-minute presentation, allowing me to delve into the details of the project, its objectives, methodologies employed, and, most importantly, the outcomes achieved. This presentation was a chance to demonstrate my communication skills, presenting complex technical information in an accessible manner. Following the presentation, the last 15 minutes were dedicated to a Q&A session facilitated by the panelists.

**************************************************************************
Links
**************************************************************************
.. note::
	* `Amazon Interview Experience for Applied Scientist <https://www.geeksforgeeks.org/amazon-interview-experience-for-applied-scientist/>`_
	* `Amazon data scientist interview (questions, process, prep) <https://igotanoffer.com/blogs/tech/amazon-data-science-interview>`_
	* `Amazon | Senior Applied Scientist L6 | Seattle <https://leetcode.com/discuss/compensation/685178/amazon-senior-applied-scientist-l6-seattle>`_
	* `Leadership Principles <https://www.amazon.jobs/content/en/our-workplace/leadership-principles>`_
