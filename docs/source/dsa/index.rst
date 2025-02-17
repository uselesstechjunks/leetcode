#################################################################################
Data Structures & Algorithms
#################################################################################
.. contents:: Table of Contents
   :depth: 3
   :local:
   :backlinks: none

*********************************************************************************
Interview Prep
*********************************************************************************
During Interview
=================================================================================
.. attention::
	#. Evaluate trade-offs.  
	#. Ask clarifying questions before coding.  
	#. Write the core logic.  
	#. Check boundary conditions.  
	#. Create test cases before dry running.  
	#. Ensure a smooth implementation.  
	#. Solve 3-4 follow-ups.  
	#. Optimize time and space complexity.  
	#. Use clear variable names and correct syntax.  
	#. Wrap up efficiently.

Topics
=================================================================================
.. attention::

	Linked-List, Bit-Manipulation, Stacks & Queues, Binary Search, Heaps, Greedy Algorithms, Dynamic Programming, Vectors/ArrayLists, Big O Time and Space, Sorting, Two Pointers, Sliding Window, Union-Find, String Manipulations, Trees and Graphs, BFS/DFS, Recursion, Back-Tracking, Hashing, Trie, Segment Trees & Binary Indexed Trees.

Bag of Tricks
=================================================================================
Count something
---------------------------------------------------------------------------------
#. Can we count compliment instead?

Find something
---------------------------------------------------------------------------------
#. Ordered?

	#. Values explicit - vanilla Binary search.
	#. Values NOT explicit 

		#. Values bounded? Binary search on range. Condition check O(T(n)) -> total T(n)lg(W), W=precision
		#. Bounded either above/below? One way binary search from one end - move from i -> 2i or i -> i/2
		#. Target forms a convex function? Optimal exists at root? 
	
			#. Can compute gradient? GD.
			#. Can compute Hessian? Newton.
#. Unordered? (a) Linear search (b) divide & conquer (c) use bookkeeping techniques.

	#. Hashmap (freq count, detect earlier occurance, obtain earliest/latest occurance)
	#. Stack (maintains insert seq in rev + can maintain first k inserted + latest in O(1))
	#. Queue (maintains insert seq + can maintain last k inserted + earliest/latest in O(1))
	#. Dequeue (maintains insert seq + can maintain first+last k inserted + earliest/latest in O(1))
	#. BST (all earlier values searchable in O(lg n) - doesn't maintain insert seq)
	#. Order statistics tree (???)
	#. Heap (smallest/largest values from earlier range in O(1) + can maintain k smallest/largest - doesn't maintain insert seq)
	#. Cartesian tree (RMQ tasks) - size extendable, range min at root O(1). Constructive requires stack. Unbalanced.
	#. Monotonic stack - maintains longest monotonic subsequence from min (max) (including curr) ending at curr

		#. Before appending, all larger (smaller) values than curr are popped
		#. After appending, top is range min (max) of [S[-2] + 1, top], S[-2] is range min of [S[-2], top]
		#. Corresponds to the rightmost branch of a Cartesian tree
		#. Bot is range min (max) for [0, top] (i.e., root of the Cartesian tree)
	#. Monotonic queue - maintains longest monotonic subsequence from min (max) (including curr) ending at curr

		#. Before appending, all larger (smaller) values than curr are removed from back
		#. After appending, back is range min (max) of [Q[-2] + 1, back], Q[-2] is range min of [S[-2], back]
		#. Corresponds to the rightmost branch of a Cartesian tree
		#. Front is range min (max) for [0, back] (i.e., root of the Cartesian tree)
	#. Min (max) stack (maintains range min (max) for [0, curr] at top + keeps all elements + obtain in O(1))
	#. Min (max) queue (maintains range min (max) for [0, curr] at back + keeps all elements + obtain in O(1))
	#. Segment tree (RSQ/RMQ, all subarray sums with prefix/suffix/sum in tree) - mutable, extends to 2d
	#. Interval tree (find value in range)
	#. Multidimensional - KD tree
	#. Binary indexed tree (???) - mutable
	#. Sparse table (RMQ)	
	#. Union find (equivalence classes)
	#. Trie (prefix matching)
	#. String hashing - Rabin Karp
#. Make bookkeeping faster - sqrt decomposition

Modify something
---------------------------------------------------------------------------------
#. Two pointers + swap
#. Dutch national flag

Schedule something
---------------------------------------------------------------------------------
#. Priority queue + optional external dict for value - greedy
#. [Tarjan][Kahn] Topological sort

Assign something
---------------------------------------------------------------------------------
#. Two pointers
#. [Kuhn] Maximal bipartite matching

Optimise something
---------------------------------------------------------------------------------
#. DP - Classic problems

	#. 0-1 knapsack
	#. Complete knapsack
	#. Multiple knapsack
	#. Monotone queue optimization
	#. Subset sum
	#. Longest common subsequence
	#. Longest increasing subsequence (LIS)
	#. Longest palindromic subsequence
	#. Rod cutting
	#. Edit distance
	#. Counting paths in a 2D array
	#. Longest Path in DAG
	#. Divide and conquer DP
	#. Knuth's optimisation
	#. ASSP [Floyd Warshall]
#. Greedy 

	#. Two pointers
	#. Sliding window
	#. Shortest path - SSSP [Dijkstra][Bellman Ford]
	#. Lightest edge - MST [Prim][Kruskal]

Check connectivity, grouping & cyclic dependencies
---------------------------------------------------------------------------------
#. Tortoise & hare algorithm
#. BFS for bipartite detection
#. DFS with edge classification, union-find
#. Lowest common ancestor - tree/graph - [Euler's tour],[Tarjan],[Farach-Colton and Bender]
#. Connected components
#. Articulation vertex and biconneted components
#. [Kosaraju] Strongly connected components
#. Eulerian circuit for cycle visiting all vertices

Combinatorial problems 
---------------------------------------------------------------------------------
#. Backtracking

Design problems 
---------------------------------------------------------------------------------
#. Mostly bookkeeping

Validate/parse something
---------------------------------------------------------------------------------
#. Stack

Involves intervals
---------------------------------------------------------------------------------
#. Sort them - overlap check left-end >= right-start
#. Sort by start - benefit (???)
#. Sort by end - benefit (???)

Common Problems
---------------------------------------------------------------------------------
#. Subarray Sum

	#. Keep track of earlier seen prefix sums
	#. [Kadane] Keep prefix or drop prefix (DP) when processing current
	#. Divide and conquer with (a) max prefix/suffix and (b) sum
	#. If mutable, Segment Tree, Binary Indexed Tree
	#. If monotonic, VLW
#. LCA
#. RSQ/RMQ
#. MEX
#. LCS
#. Order stat

*********************************************************************************
Revision Deck
*********************************************************************************
Important Questions
=================================================================================
.. toctree::
	:maxdepth: 1

	qb

Deep Dives
=================================================================================
.. toctree::
	:maxdepth: 1

	range
	graph
	dsasol

*********************************************************************************
Resources
*********************************************************************************
Fundamentals
================================================================================
.. attention::

	* [neetcode.io] `Neetcode <https://neetcode.io/practice>`_ 
	* [cp-algorithms.com] `Algorithms for Competitive Programming <https://cp-algorithms.com/>`_

Code Patterns
================================================================================
.. note::

	* [algo.monster] `Templates <https://algo.monster/templates>`_
	* [github.io] `LC cheatsheet <https://jwl-7.github.io/leetcode-cheatsheet/>`_

Problem Solving Patterns
================================================================================
Sliding Window
---------------------------------------------------------------------------------
.. important::
	* [leetcode.com] `Sliding Window Technique: A Comprehensive Guide <https://leetcode.com/discuss/interview-question/3722472/mastering-sliding-window-technique-a-comprehensive-guide>`_
	* [geeksforgeeks.org] `Sliding Window Technique <https://www.geeksforgeeks.org/window-sliding-technique/>`_
	* [leetcode.com] `Sliding window with frequency counts <https://leetcode.com/problems/subarrays-with-k-different-integers/solutions/235002/one-code-template-to-solve-all-of-these-problems/>`_

Two Pointers
---------------------------------------------------------------------------------
.. important::
	* [leetcode.com] `Solved all two pointers problems in 100 days. <https://leetcode.com/discuss/study-guide/1688903/Solved-all-two-pointers-problems-in-100-days>`_
	* [reddit.com] `Two-Pointer Technique, an In-Depth Guide <https://www.reddit.com/r/leetcode/comments/18g9383/twopointer_technique_an_indepth_guide_concepts/?rdt=59240>`_

Mostly Useful
================================================================================
.. important::
	* [takeuforward.org] `Strivers A2Z DSA Course/Sheet <https://takeuforward.org/strivers-a2z-dsa-course/strivers-a2z-dsa-course-sheet-2/>`_
	* [leetracer.com] `Screen Questions - (Company Tags Last Updated: 11-02-24) <https://leetracer.com/screener>`_	
	* [github.com] `LeetCode-Questions-CompanyWise <https://github.com/krishnadey30/LeetCode-Questions-CompanyWise/blob/master/google_6months.csv>`_
	* [leetcode.com] `Top Google Questions <https://leetcode.com/problem-list/7p55wqm/>`_
	* [leetcode.com] `Top 100 Liked <https://leetcode.com/studyplan/top-100-liked/>`_
	* [reddit.com] `I passed Meta E6 Hiring Committee (Screen+FullLoop). My thoughts, advice, tips. <https://www.reddit.com/r/leetcode/comments/1c7fs3o/i_passed_meta_e6_hiring_committee_screenfullloop/?share_id=jeNswSOERGx8GXDy02DBq&utm_name=androidcss>`_
	* [reddit.com] `3 parts of mastering LC with SCC example <https://www.reddit.com/r/leetcode/comments/1hye4hy/comment/m6pucmj/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button>`_
	* [reddit.com] `Here's a master list of problem keyword to algorithm patterns <https://www.reddit.com/r/leetcode/comments/1f9bejz/heres_a_master_list_of_problem_keyword_to/?share_id=_p0H75FfOq1zSO0yBWj8v&utm_name=androidcss>`_

Might be Useful
================================================================================
.. note::
	* [leetcode.com] `Lessons from My Google L5 Interview Experience: Tips and suggestion <https://leetcode.com/discuss/interview-question/6147892/Lessons-from-My-Google-L5-Interview-Experience%3A-Tips-and-suggestion>`_
	* [leetcode.com] `Google Onsite <https://leetcode.com/discuss/interview-question/849947/google-onsite>`_
	* [docs.google] `Leetcode Questions - public - solved by Siddhesh <https://docs.google.com/spreadsheets/d/1KkCeOIBwUFfKrHGGZe_6EJRCIqaM6MJBo0uSIMSD9bs/edit?gid=782922309#gid=782922309>`_
	* [hellointerview.com] `Data Structures and Algorithms <https://www.hellointerview.com/learn/code>`_
	* [leetcodethehardway.com] `LeetCode The Hard Way <https://leetcodethehardway.com/tutorials/category/basic-topics>`_
	* [deriveit.org] `DeriveIt <https://deriveit.org/coding>`_	
	* [docs.google] `DSA Resource Problem Set <https://docs.google.com/spreadsheets/d/1hwvHbRargzmbErRYGU2cjxf4PR8GTOI-e1R9VqOVQgY/edit?gid=481396158#gid=481396158>`_
	* [leetcode.com] `14 Patterns to Ace Any Coding Interview Question <https://leetcode.com/discuss/study-guide/4039411/14-Patterns-to-Ace-Any-Coding-Interview-Question>`_
	* [educative.io] `Grokking the Coding Interview Patterns <https://www.educative.io/courses/grokking-coding-interview>`_
	* [educative.io] `3 month coding interview preparation bootcamp <https://www.educative.io/blog/coding-interivew-preparation-bootcamp>`_
	* [educative.io] `Data Structures for Coding Interviews in Python <https://www.educative.io/courses/data-structures-coding-interviews-python>`_
	* [github.com] `Grokking the Coding Interview Patterns for Coding Questions <https://github.com/dipjul/Grokking-the-Coding-Interview-Patterns-for-Coding-Questions>`_
	* [designgurus.io] `Grokking the Coding Interview: Patterns for Coding Questions <https://www.designgurus.io/course/grokking-the-coding-interview>`_
	* [github.com] `Tech Interview Handbook <https://github.com/yangshun/tech-interview-handbook>`_
	* [github.com] `Interview_DS_Algo <https://github.com/MAZHARMIK/Interview_DS_Algo>`_
	* [geeksforgeeks.org] `Must Do Coding Questions for Companies <https://www.geeksforgeeks.org/must-do-coding-questions-for-companies-like-amazon-microsoft-adobe/>`_
	* [geeksforgeeks.org] `Must Do Coding Questions Company-wise <https://www.geeksforgeeks.org/must-coding-questions-company-wise/>`_	
	* [interview-prep-pro.vercel.app] `Interview Prep Pro <https://interview-prep-pro.vercel.app/>`_

Base
================================================================================
1. Arrays and Strings
---------------------------------------------------------------------------------
Key Concepts: Sliding window, two pointers, prefix sum, Kadane’s algorithm, string manipulation.

Problems to Practice:

* Longest substring without repeating characters
* Maximum subarray sum (Kadane’s algorithm)
* Rotate array (in-place rotation)
* Valid palindrome
* String pattern matching (e.g., KMP algorithm, Rabin-Karp)

2. Hashing
---------------------------------------------------------------------------------
Key Concepts: Hash maps, sets, frequency counts, collision handling.

Problems to Practice:

* Two-sum problem variants.
* Longest substring with at most k distinct characters.
* Group anagrams.
* Subarray with a given sum (hash map for prefix sums).

3. Linked Lists
---------------------------------------------------------------------------------
Key Concepts: Fast and slow pointers, reversing, merging, detecting cycles.

Problems to Practice:

* Reverse a linked list.
* Merge two sorted linked lists.
* Detect and remove cycle in a linked list.
* Intersection of two linked lists.
* Flatten a multilevel doubly linked list.

4. Trees and Graphs
---------------------------------------------------------------------------------
Key Concepts:

* Trees: Traversals (DFS, BFS), recursion, binary search tree properties.
* Graphs: Representations (adjacency list/matrix), DFS, BFS, Dijkstra, union-find.

Problems to Practice:

* Binary tree level order traversal.
* Lowest common ancestor (LCA).
* Validate binary search tree.
* Number of islands (DFS/BFS).
* Shortest path in a graph (Dijkstra’s algorithm).
* Detect cycle in an undirected graph (union-find).
	
5. Recursion and Backtracking
---------------------------------------------------------------------------------
Key Concepts: Base case, recursive stack, pruning.

Problems to Practice:

* Permutations and combinations.
* N-Queens problem.
* Sudoku solver.
* Subset sum problem.
* Word search in a grid.

6. Dynamic Programming
---------------------------------------------------------------------------------
Key Concepts: Memoization, tabulation, state definition, transitions.

Problems to Practice:

* Longest increasing subsequence.
* Longest common subsequence.
* 0/1 Knapsack problem.
* Coin change problem.
* Edit distance (Levenshtein distance).

7. Sorting and Searching
---------------------------------------------------------------------------------
Key Concepts: Merge sort, quicksort, binary search (with variations).

Problems to Practice:

* Search in a rotated sorted array.
* Median of two sorted arrays.
* Kth largest element in an array.
* Closest pair of points.

8. Stacks and Queues
---------------------------------------------------------------------------------
Key Concepts: Monotonic stack, deque (double-ended queue), LRU cache.

Problems to Practice:

* Valid parentheses.
* Largest rectangle in histogram.
* Sliding window maximum.
* Implement a queue using two stacks.

9. Bit Manipulation
---------------------------------------------------------------------------------
Key Concepts: XOR, bit shifts, masking, counting set bits.

Problems to Practice:

* Single number (XOR-based solution).
* Subsets using bit masks.
* Reverse bits.
* Count the number of 1 bits (Hamming weight).

10. Math and Geometry
---------------------------------------------------------------------------------
Key Concepts: GCD, LCM, modular arithmetic, Euclidean algorithm.

Problems to Practice:

* Check if a number is prime.
* Find GCD/LCM of two numbers.
* Count primes up to n (Sieve of Eratosthenes).
* Water trapped after rainfall (two-pointer approach).
