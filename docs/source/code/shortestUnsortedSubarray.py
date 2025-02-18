def findUnsortedSubarray(self, nums: List[int]) -> int:
	# Note:
	# if we sort the numbers, then it is easy to find the solution
	# which involves identifying interval [i,j] where i is the
	# leftmost index where sorted[i] differs from nums[i] and j
	# is the rightmost index where sorted[j] differs from nums[j].
	#
	# The trick here is to identify i and j without sorting via understanding inversions.
	# 
	# Note that when we sort it, the leftmost point where nums[i] != sorted[i]
	# is where the first inversion happens - nums[i] must have gotten swapped
	# with someting smaller on the right.
	# Similarly, for the rightmost point.
	#
	# We define two intervals: [0,j] and [i,n-1] and take their intersections.
	# (a) Find [0,j] => where j is the rightmost index which is inverted.
	# (b) Find [i,n-1] => where i is the leftmost index which is inverted.
	#
	# For inversions, keep in mind:
	# if nums[p] <-> nums[q] is inverted for some p < q, and nums[r] > nums[p]
	# for p < r < q, then nums[r] <-> nums[p] is also inverted.
	#
	# Therefore, when traversing from left to right, we just need to keep 
	# track of the max we've seen from the left and find inversions on the 
	# right just by comparing it with max.
	#
	# Similarly, when traversing from right to left, we just need to keep
	# track of the min we've seen from right, and find inversions on the left
	# by comparisng only with it
	n = len(nums)
	maxLeft = min(nums)-1
	last = -2
	for i in range(n):
		maxLeft = max(maxLeft, nums[i])
		if maxLeft > nums[i]:
			last = i
	minRight = max(nums)+1
	first = -1
	for i in range(n-1,-1,-1):
		minRight = min(minRight, nums[i])
		if minRight < nums[i]:
			first = i
	return last-first+1