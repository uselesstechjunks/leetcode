================================================================================
Data Structures
================================================================================
#. Segment Tree:

	- Sum, min, max, and custom range queries.
	- Lazy propagation for range updates.
	- Variants like mergeable segment trees.
#. Fenwick Tree (Binary Indexed Tree):

	- Point updates and prefix/range queries.
	- Multidimensional Fenwick Trees.
#. Sparse Table:

	- Efficient for immutable data (static range queries like min, max, or GCD).
#. Order Statistics Tree (Augmented BST or Fenwick Tree with Order Statistics):

	- Find kth smallest element.
	- Count of elements less than or greater than a given value.
#. RMQ (Range Minimum Query):

	- Hybrid solutions combining segment tree and sparse table for efficiency.
#. Wavelet Tree:

	- Handles range frequency queries and range kth order statistics.
#. Mo’s Algorithm:

	- Square-root decomposition for offline range queries.
#. Merge Sort Tree:

	- Efficient for range queries involving sorted data.
#. Interval Tree and KD-Tree:

	- For multidimensional range queries.

================================================================================
Algorithms
================================================================================
#. Divide-and-Conquer approaches (e.g., inversion count with merge sort).
#. Sliding window techniques (efficient for specific range problems).
#. Two-pointer methods for range problems in sorted data.
#. Offline processing for batch queries using Mo's Algorithm or persistent data structures.

================================================================================
Example Problems
================================================================================
Order Statistics
--------------------------------------------------------------------------------
#. Kth Largest/Smallest Element in a Stream:

	- Maintain the top k elements in a stream of data.
	- Example: Leverage min-heaps or order statistics trees.

#. Find the Median of a Running Stream:

	- Use two heaps (max-heap and min-heap) for efficiency.

#. Count of Smaller/Larger Numbers After Self:

	- Given an array, for each element, count how many elements are smaller/larger to its right.
	- Solution: Fenwick Tree, segment tree, or merge sort.

#. Find the Kth Largest Element in an Unsorted Array:

	- Variants where you cannot sort directly (e.g., use Quickselect).

#. kth Element in the Cartesian Product

	- Problem: Given two sorted arrays :math:`A` and :math:`B`, find the :math:`k`-th smallest element in the Cartesian product of :math:`A \times B`. 
	- Hints: Use a min-heap with tuples to track possible combinations efficiently.

#. Median in a Sliding Window

	- Problem: Given an array of integers and a sliding window of size :math:`k`, find the median of each window as it slides from left to right.
	- Hints: Use two heaps (max-heap and min-heap) to dynamically maintain the window.

#. Inversion Count in Subarrays

	- Problem: For an array :math:`A`, process :math:`q` queries of the form :math:`(L, R)` where you need to count the number of inversions in the subarray :math:`A[L:R]`.
	- Hints: Use a segment tree with merge-sort logic at each node.

#. Range k-th Smallest Element

	- Problem: Given an array and :math:`q` queries of the form :math:`(L, R, k)`, find the :math:`k`-th smallest element in the range :math:`[L, R]`.
	- Hints: Use a merge sort tree or wavelet tree for efficient query processing.

#. Count of Numbers in Range with a Given Frequency

	- Problem: Given an array and :math:`q` queries of the form :math:`(L, R, F)`, count how many numbers in the range :math:`[L, R]` appear exactly :math:`F` times.
	- Hints: Use Mo’s Algorithm with frequency tracking or segment trees with custom nodes.

Range Query Problems
--------------------------------------------------------------------------------
#. Range Sum Query with Updates:

	- Solve using segment trees or Fenwick trees with range updates.

#. Range Minimum/Maximum Query:

	- Solve using segment trees, sparse tables, or hybrid methods.

#. Dynamic Range Median Queries:

	- Maintain a dynamic dataset and answer queries for the median of a range.

#. Range XOR Query:

	- Solve using segment trees.

#. Sum of Range Products:

	- Given an array, answer the sum of products of all pairs in the range [L, R].

#. Number of Distinct Elements in Range:

	- Use Mo’s Algorithm or a segment tree with a map structure.

#. Range Frequency Query:

	- Solve using a wavelet tree or merge sort tree.

#. Dynamic Range Median Queries

	- Problem: Maintain a dynamic array supporting:

		1. Insertion of an element.
		2. Deletion of an element.
		3. Querying the median of any range :math:`[L, R]`.
	- Hints: Combine balanced BST or heaps with a range query structure like segment trees.

#. Range XOR with Updates

	- Problem: Given an array of integers, process the following operations efficiently:

		1. Update the :math:`i`-th element to :math:`x`.
		2. Query the XOR of elements in the range :math:`[L, R]`.
	- Hints: Use a segment tree with XOR as the operation and point updates.

#. Maximum Frequency in a Range

	- Problem: Given an array and :math:`q` queries of the form :math:`(L, R)`, find the most frequent number in the range :math:`[L, R]`.
	- Hints: Use a segment tree with frequency maps stored at each node.

#. Maximum Subarray Sum in a Range

	- Problem: Process queries of the form :math:`(L, R)`, where you must find the maximum subarray sum in the range :math:`[L, R]`.
	- Hints: Augment the segment tree to store max subarray sums and handle overlapping subranges efficiently.

#. Range Updates with a Custom Function

	- Problem: Design a data structure to efficiently handle:

		1. Updates: Apply a custom function :math:`f(x)` to all elements in the range :math:`[L, R]`.
		2. Queries: Retrieve the sum of all elements in the range :math:`[L, R]`.
	- Hints: Use a segment tree with lazy propagation where :math:`f(x)` can be propagated efficiently.

Hybrid Problems
--------------------------------------------------------------------------------
#. Dynamic Skyline Problem:

	- Given a list of intervals, dynamically insert or delete intervals and determine the current skyline.

#. Maximum Sum Rectangle in a 2D Matrix:

	- Use a 1D segment tree approach for optimal results.

#. Range GCD Query:

	- Find the GCD of elements in the range [L, R] using a segment tree or sparse table.

#. Number of Rectangles Containing a Point

	- Problem: You are given a list of :math:`n` rectangles (defined by two opposite corners) and :math:`q` points. For each point, count how many rectangles contain it.
	- Hints: Use a segment tree or 2D Fenwick Tree to maintain active ranges as you sweep through one coordinate.

#. Dynamic Skyline

	- Problem: Maintain the skyline (maximum height of buildings seen from a distance) as you dynamically add and remove buildings.
	- Hints: Use an interval tree or segment tree to handle dynamic range updates efficiently.

#. Count Subarrays with Given Sum in Range

	- Problem: For :math:`q` queries :math:`(L, R, S)`, count how many contiguous subarrays in the range :math:`[L, R]` have a sum equal to :math:`S`.
	- Hints: Use prefix sums with a Fenwick Tree to count valid subarray sums efficiently.

#. Maximum Overlap of Intervals

	- Problem: Given a list of intervals, process :math:`q` queries to find the maximum overlap of intervals in a given range :math:`[L, R]`.
	- Hints: Use a difference array combined with prefix sums or a segment tree for dynamic updates.

#. Submatrix Sum Queries

	- Problem: Given a 2D grid, process:

		1. Updates: Add a value to all elements in a submatrix.
		2. Queries: Find the sum of elements in any submatrix.
	- Hints: Use a 2D Fenwick Tree or segment tree for efficient query and update operations.
