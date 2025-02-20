#################################################################################
Data Structures & Algorithms
#################################################################################
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

*********************************************************************************
Topics
*********************************************************************************
Linked-List, Bit-Manipulation, Stacks & Queues, Binary Search, Heaps, Greedy Algorithms, Dynamic Programming, Vectors/ArrayLists, Big O Time and Space, Sorting, Two Pointers, Sliding Window, Union-Find, String Manipulations, Trees and Graphs, BFS/DFS, Recursion, Back-Tracking, Hashing, Trie, Segment Trees & Binary Indexed Trees.

Core Algorithms
=====================================================================
.. attention::
	- Binary Search
	- Prefix Sum
	- Two Pointers
	- Sliding Window
	- Divide & Conquer
	- DP
	- Scheduling
	- DFS/BFS

*********************************************************************************
Bag of Tricks
*********************************************************************************
Goal: Map the problem to known tasks.

Thought Process
=====================================================================
.. note::
	#. Does it form a group, chain, tree, graph? - union find, parent-child hashmap (set if parent-child is implicit).
	#. Does it have a range? - binary search.
	#. Does it have monotonic property? - binary search/VLW.
	#. What bookkeeping is required? What involves recomputation? What else can we track to avoid it? - hashmap, bst, stack, queue, heap.
	#. Can we solve it in parts and combine the results? - divide and conquer, recursion, DP.
	#. What choices can be greedily eliminated? - two pointers, greedy, quicksort partitioning.

Find something
=================================================================================
Types of Queries
---------------------------------------------------------------------------------
#. OGQ - Optimal Goal Query: VLW - variable length window + aux bookkeeping (monotonic goal), FLW - fixed length window (works for non-monotone)
#. RSQ - Range Sum Query: :math:`\sum(l,r)`: Prefix sum, BIT, segment tree
#. MSQ - Maximum Sum Query: :math:`[0,n)->\max(\sum(l,r))`: Prefix sum->BIT, VLW->Kadane, divide and conquer->segment tree
#. RMQ - Range Min Query: :math:`\min(l,r)`: [unordered] monotonic stack, monotonic queue (VLW), Cartesian tree, segment tree, [ordered] binary search, BST (VLW)
#. RFQ - Range Frequency Query: :math:`c(l,r,key)`: Dict, segment tree
#. EEQ - Earlier Existance Query: set, dict, bitmap
#. LSQ - Latest Smaller Query: :math:`\max(l | l<r, v(l)<v(r))`: Monotonic stack (v1), Cartesian tree
#. ESQ - Earliest Smaller Query: :math:`\min(l | l<r, v(l)<v(r))`: Monotonic stack (v2), (???) inversions/pointers?
#. SEQ - Smallest Earlier Query: :math:`\min(v(l) | l<r, v(l)<v(r))`: pointer, bst, heap
#. TKQ - Top K Query: heap
#. RIQ - Range Intersect Query: Given point, find ranges that contains it: Interval tree
#. ROQ - Range Overlap Query: Find intervals that overlaps with given range: Sorting + binary search, sorting + stack
#. ECQ - Equivalence Class Query: Whether (x,y) belonds to the same group: Union find, dict for parent-child
#. MEX - Minimum Excluded Element: (???)
#. LCS - Longest common/increasing/palindromic subsequence: VLW, DP
#. RUQ - Range Update Query: Prefix sum->BIT (+delta at begin, -delta at end), segment tree

General Techniques
---------------------------------------------------------------------------------
Ordered
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. Values explicit - vanilla Binary search.
#. Values NOT explicit 

	#. Values bounded? Binary search on range. Condition check O(T(n)) -> total T(n)lg(W), W=precision
	#. Bounded either above/below? One way binary search from one end - move from i -> 2i or i -> i/2
	#. Target forms a convex function? Optimal exists at root? 

		#. Can compute gradient? GD.
		#. Can compute Hessian? Newton.
Unordered
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. Linear search
#. Divide & conquer 
#. Use bookkeeping techniques

Bookkeeping
---------------------------------------------------------------------------------
#. KV map - multiple O(1) use-cases

	- freq counts - histogram delta >= 0, distinct count >=k, min freq count >= k, majority-non majority count (max freq - sum V)
	- detect earlier occurance, obtain earliest/latest occurance with paired value
	- parent-child relation (stores parent/child pointer), alternative to union-find
#. Stack (maintains insert seq in rev + can maintain first k inserted + latest in O(1))
#. Queue (maintains insert seq + can maintain last k inserted + earliest/latest in O(1))
#. Dequeue (maintains insert seq + can maintain first+last k inserted + earliest/latest in O(1))
#. BST (all earlier values searchable in O(lg n) - doesn't maintain insert seq) - sortedcontainers.SortedList
#. Order statistics tree (???)
#. Heap (smallest/largest values from earlier range in O(1) + can maintain k smallest/largest - doesn't maintain insert seq)
#. Cartesian tree (RMQ tasks) - heap with insert seq: range min at root O(1). Constructive requires stack. Unbalanced.
#. Monotonic stack - 2 types 

	#. Type I: RMQ (Range Min/Max Query): Simulates Cartesian tree.

		#. Maintains longest monotonic subsequence from min (max) (including curr) ending at curr.
		#. At finish, corresponds to the rightmost branch of a Cartesian tree.
		#. Everything that has ever been on the stack is the min of some range. This covers all possible range mins.
		#. For every curr, all larger (smaller) values are popped - curr is RM of everything since popped.
		#. Once pushed, top is range min (max) of [S[-2]+1, top]. S[-2] is range min of [S[-3]+1, top]		
		#. Bot is range min (max) for [0, top] (i.e., root of the Cartesian tree)
		#. Each value gets to be at the stack at some point.
	#. Type II: ESQ (Earliest Smaller/Larger Query)

		#. Maintains longest monotonic subsequence from first element.
		#. Everything that comes after, only pushed onto the stack if it's larger (smaller)
#. Monotonic queue - Same as monotonic stack except it works for sliding window as we can skip ranges by popping root (at front).
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

Count something
=================================================================================
#. Can we count compliment instead?

Modify something
=================================================================================
#. Two pointers + swap
#. Dutch national flag

Schedule something
=================================================================================
#. Priority queue + optional external dict for value - greedy
#. [Tarjan][Kahn] Topological sort

Assign something
=================================================================================
#. Two pointers
#. [Kuhn] Maximal bipartite matching

Optimise something
=================================================================================
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
=================================================================================
#. Tortoise & hare algorithm
#. BFS for bipartite detection
#. DFS with edge classification, union-find
#. Lowest common ancestor - tree/graph - [Euler's tour],[Tarjan],[Farach-Colton and Bender]
#. Connected components
#. Articulation vertex and biconneted components
#. [Kosaraju] Strongly connected components
#. Eulerian circuit for cycle visiting all vertices

Combine something
=================================================================================
#. Backtracking

Design something 
=================================================================================
#. Mostly bookkeeping

Validate something
=================================================================================
#. Paring problems - Stack
#. Regex problems - DP

Involves intervals
=================================================================================
#. Sort them - overlap check left-end >= right-start
#. Sort by start - benefit (???)
#. Sort by end - benefit (???)

*********************************************************************************
Common Problems
*********************************************************************************
#. Maximum Sum Query: [0,n)->max(Sum(l,r))

	#. Keep track of earlier seen prefix sums
	#. [Kadane] Keep prefix or drop prefix (DP) when processing current
	#. Divide and conquer with (a) max prefix/suffix and (b) sum
	#. If mutable, Segment Tree, Binary Indexed Tree
	#. If monotonic, VLW

*********************************************************************************
Revision Deck
*********************************************************************************
Keep in Mind
=================================================================================
.. toctree::
	:maxdepth: 1

	interview

Important Questions
=================================================================================
.. toctree::
	:maxdepth: 1

	ql
	qr

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
	* [bigocheatsheet.com] `Know Thy Complexities! <http://bigocheatsheet.com/>`_
	* [cp-algorithms.com] `Algorithms for Competitive Programming <https://cp-algorithms.com/>`_

Code Patterns
================================================================================
.. note::

	* [algo.monster] `Templates <https://algo.monster/templates>`_
	* [github.io] `LC cheatsheet <https://jwl-7.github.io/leetcode-cheatsheet/>`_

Sliding Window
================================================================================
.. note::
	- fixed length

		- fixed sum with constant extra bookkeeping
		- fixed sum with auxiliary data structures
	- variable length

		- fixed sum with constant extra bookkeeping - aggregate >= value
		- fixed sum with auxiliary data structures - frequency, prefix sums -> dict, monotonic queue, bst
.. attention::
	- sequential grouping
	- sequential criteria - longest, smallest, contained, largest, smallest

.. important::
	* [youtube.com] `Sliding Window Mental Model <https://www.youtube.com/watch?v=MK-NZ4hN7rs>`_
	* [leetcode.com] `Sliding Window Technique: A Comprehensive Guide <https://leetcode.com/discuss/interview-question/3722472/mastering-sliding-window-technique-a-comprehensive-guide>`_
	* [geeksforgeeks.org] `Sliding Window Technique <https://www.geeksforgeeks.org/window-sliding-technique/>`_
	* [leetcode.com] `Sliding window with frequency counts <https://leetcode.com/problems/subarrays-with-k-different-integers/solutions/235002/one-code-template-to-solve-all-of-these-problems/>`_
	* [leetcode.com] `Lee's solutions for sliding window <https://leetcode.com/problems/find-the-longest-equal-subarray/solutions/3934172/java-c-python-one-pass-sliding-window-o-n>`_

Two Pointers
================================================================================
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
*********************************************************************************
Fundamental Questions
*********************************************************************************
1. Arrays and Strings
================================================================================
Key Concepts: Sliding window, two pointers, prefix sum, Kadane’s algorithm, string manipulation.

Problems to Practice:

* Longest substring without repeating characters
* Maximum subarray sum (Kadane’s algorithm)
* Rotate array (in-place rotation)
* Valid palindrome
* String pattern matching (e.g., KMP algorithm, Rabin-Karp)

2. Hashing
================================================================================
Key Concepts: Hash maps, sets, frequency counts, collision handling.

Problems to Practice:

* Two-sum problem variants.
* Longest substring with at most k distinct characters.
* Group anagrams.
* Subarray with a given sum (hash map for prefix sums).

3. Linked Lists
================================================================================
Key Concepts: Fast and slow pointers, reversing, merging, detecting cycles.

Problems to Practice:

* Reverse a linked list.
* Merge two sorted linked lists.
* Detect and remove cycle in a linked list.
* Intersection of two linked lists.
* Flatten a multilevel doubly linked list.

4. Trees and Graphs
================================================================================
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
================================================================================
Key Concepts: Base case, recursive stack, pruning.

Problems to Practice:

* Permutations and combinations.
* N-Queens problem.
* Sudoku solver.
* Subset sum problem.
* Word search in a grid.

6. Dynamic Programming
================================================================================
Key Concepts: Memoization, tabulation, state definition, transitions.

Problems to Practice:

* Longest increasing subsequence.
* Longest common subsequence.
* 0/1 Knapsack problem.
* Coin change problem.
* Edit distance (Levenshtein distance).

7. Sorting and Searching
================================================================================
Key Concepts: Merge sort, quicksort, binary search (with variations).

Problems to Practice:

* Search in a rotated sorted array.
* Median of two sorted arrays.
* Kth largest element in an array.
* Closest pair of points.

8. Stacks and Queues
================================================================================
Key Concepts: Monotonic stack, deque (double-ended queue), LRU cache.

Problems to Practice:

* Valid parentheses.
* Largest rectangle in histogram.
* Sliding window maximum.
* Implement a queue using two stacks.

9. Bit Manipulation
================================================================================
Key Concepts: XOR, bit shifts, masking, counting set bits.

Problems to Practice:

* Single number (XOR-based solution).
* Subsets using bit masks.
* Reverse bits.
* Count the number of 1 bits (Hamming weight).

10. Math and Geometry
================================================================================
Key Concepts: GCD, LCM, modular arithmetic, Euclidean algorithm.

Problems to Practice:

* Check if a number is prime.
* Find GCD/LCM of two numbers.
* Count primes up to n (Sieve of Eratosthenes).
* Water trapped after rainfall (two-pointer approach).
