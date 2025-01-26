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

Hybrid Problems
--------------------------------------------------------------------------------
#. Dynamic Skyline Problem:

	- Given a list of intervals, dynamically insert or delete intervals and determine the current skyline.

#. Maximum Sum Rectangle in a 2D Matrix:

	- Use a 1D segment tree approach for optimal results.

#. Range GCD Query:

	- Find the GCD of elements in the range [L, R] using a segment tree or sparse table.
