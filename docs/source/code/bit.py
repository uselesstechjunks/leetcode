from random import shuffle

class BinaryIndexedTree:
	def __init__(self, nums):
		def build():
			for index in range(len(self)):
				self.add(index, nums[index])

		self.n = len(nums)
		self.nums = nums
		self.bit = [0] * self.n
		build()
	
	def add(self, index, delta):
		""" adds a delta to a current index"""
		if delta == 0: return
		while index < len(self):
			self.bit[index] += delta
			index = index | (index + 1)

	def update(self, index, value):
		prev = self.nums[index]
		self.nums[index] = value
		self.add(index, value-prev)

	def sum(self, left, right):
		def sum(right):
			""" returns the sum of the range [0, right] inclusive"""
			res = 0
			while right >= 0:
				res += self[right]
				right = (right & (right + 1)) - 1
			return res

		return sum(right) - sum(left-1)

	def __len__(self):
		return self.n

	def __getitem__(self, index):
		return self.bit[index] if index >= 0 and index < self.n else None

	def __repr__(self):
		return '[{}]'.format(', '.join([str(self[i]) for i in range(len(self))]))

def range_sum(nums, left, right):
	res = 0
	for i in range(left, right + 1):
		res += nums[i]
	return res

def test1():
	n = 4
	
	# create raw data
	nums = list(range(n))
	shuffle(nums)

	# create binary indexed tree to precompute the partial range sums
	bit = BinaryIndexedTree(nums)

	# enumerate all possible subranges and validate with range sum computed by brute force
	for i in range(n):
		for j in range(i,n):
			assert(range_sum(nums, i, j) == bit.sum(i, j))

def test2():
	""" ["NumArray","update","update","update","sumRange","update","sumRange","update","sumRange","sumRange","update"] """
	""" [[[7,2,7,2,0]],[4,6],[0,2],[0,9],[4,4],[3,8],[0,4],[4,1],[0,3],[0,4],[0,4]] """
	""" [null,null,null,null,6,null,32,null,26,27,null] """
	nums = [7,2,7,2,0]
	bit = BinaryIndexedTree(nums)
	bit.update(4, 6)
	bit.update(0, 2)
	bit.update(0, 9)
	assert(bit.sum(4, 4) == 6)
	bit.update(3, 8)
	assert(bit.sum(0, 4) == 32)
	bit.update(4, 1)
	assert(bit.sum(0, 3) == 26)
	assert(bit.sum(0, 4) == 27)
	bit.update(0, 4)

if __name__ == '__main__':
	test1()
	test2()