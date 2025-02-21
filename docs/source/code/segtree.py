from random import randint
from collections import Counter

class SegmentTree:

	def __init__(self, nums, combine):
		def build(root_index, range_left, range_right):
			nonlocal nums
			if range_left > range_right:
				return
			if range_left == range_right:
				self.tree[root_index] = nums[range_left]
				return
			range_mid = (range_left + range_right) // 2
			left_index, right_index = 2 * root_index, 2 * root_index + 1
			build(left_index, range_left, range_mid)
			build(right_index, range_mid + 1, range_right)
			self.tree[root_index] = self.combine(self.tree[left_index], self.tree[right_index])

		self.n = len(nums)
		self.tree = [0] * 4 * self.n
		self.combine = combine
		build(1, 0, self.n-1)

	def update(self, index: int, val: int) -> None:
		def impl(root_index, range_left, range_right, insert_index, val):
			if range_left > range_right:
				return
			if range_left == range_right:
				self.tree[root_index] = val
				return
			range_mid = (range_left + range_right) // 2
			left_index, right_index = 2 * root_index, 2 * root_index + 1
			if insert_index <= range_mid:
				impl(left_index, range_left, range_mid, insert_index, val)
			else:
				impl(right_index, range_mid + 1, range_right, insert_index, val)
			self.tree[root_index] = self.combine(self.tree[left_index], self.tree[right_index])
		
		impl(1, 0, self.n-1, index, val)

	def sumRange(self, left: int, right: int) -> int:
		def impl(root_index, range_left, range_right, left, right):
			if range_left > range_right:
				return 0
			if range_left == left and range_right == right:
				return self.tree[root_index]
			range_mid = (range_left + range_right) // 2
			left_index, right_index = 2 * root_index, 2 * root_index + 1
			if right <= range_mid:
				return impl(left_index, range_left, range_mid, left, right)
			elif range_mid < left:
				return impl(right_index, range_mid+1, range_right, left, right)
			return impl(left_index, range_left, range_mid, left, range_mid) + impl(right_index, range_mid + 1, range_right, range_mid + 1, right)
		return impl(1, 0, self.n-1, left, right)

class RangeSum(SegmentTree):
    def __init__(self, arr):
        super().__init__(arr=arr, combine=lambda x,y: x+y)

class RangeMin(SegmentTree):
    def __init__(self, arr):
        super().__init__(arr=arr, combine=lambda x,y: min(x, y))

class RangeFrequency(SegmentTree):
    def __init__(self, arr):
        counts = [(x,1) for x in arr]
        def combine(x, y):
            if x[0] < y[0]:
                return x
            if x[0] > y[0]:
                return y
            return (x[0], x[1]+y[1])
        super().__init__(arr=counts, combine=combine)

class RangeOrderStatistics(SegmentTree):
    def __init__(self, arr):
        counts = [1 if x == 0 else 0 for x in arr]
        super().__init__(arr=counts, combine=lambda x,y: x+y)
    def find_kth_idx(self, k):
        def impl(i, tl, tr, k):
            if k > self.tree[i]:
                return -1
            if tl == tr:
                return tl
            tm = tl+(tr-tl)//2
            if k <= self.tree[2*i]:
                return impl(2*i, tl, tm, k)
            else:
                return impl(2*i+1, tm+1, tr, k-self.tree[2*i])
        return impl(1, 0, self.n-1, k)
