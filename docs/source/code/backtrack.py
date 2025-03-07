def permute(self, nums: List[int]) -> List[List[int]]:
	def backtrack(index, n):
		nonlocal res
		if index == n:
			res.append(nums[:])
			return
		for i in range(index, n):
			nums[i], nums[index] = nums[index], nums[i]
			backtrack(index + 1, n)
			nums[i], nums[index] = nums[index], nums[i]
	res = []
	backtrack(0, len(nums))
	return res

def combine(self, n: int, k: int) -> List[List[int]]:
	def optimized():
		""" prevents wasteful subset formation with additional state information """
		# [1,2,3]      curr                start   range 
		# remaining=2, []                      1-> [1,2]
		# remaining=1, [1],        [2]         2-> [2,3], 3 -> [3]
		# remaining=0, [1,2],[1,3],[2,3]
		def backtrack(curr, start, remaining):
			nonlocal res
			if remaining == 0:
				res.append(curr[:])
				return
			for num in range(start, n - remaining + 2):
				curr.append(num)
				backtrack(curr, num + 1, remaining - 1)
				curr.pop()
		res = []
		backtrack([], 1, k)
		return res

	def simple():
		""" In this formulation, subsets get wasted, e.g. [3] """
		# [1,2,3]                    start    range 
		# k=0, []                      1-> [1,2,3]
		# k=1, [1],        [2], [3]    2-> [2,3], 3->[3]
		# k=2, [1,2],[1,3],[2,3]
		def backtrack(curr, start):
			nonlocal res
			if len(curr) == k:
				res.append(curr[:])
				return
			for num in range(start, n + 1):
				curr.append(num)
				""" NOTE: NOT START + 1"""
				backtrack(curr, num + 1)
				curr.pop()
		res = []
		backtrack([], 1)
		return res

def generateParenthesis(self, n: int) -> List[str]:
	# n = 3               left   right
	# (                     1      0
	# ((,      ()           1      1
	# (((,((), ()(          2      1
	# (((,(()(,()((,()()
	res, stack = [], []
	def backtrack(stack, n, left_count, right_count):
		if right_count == n:
			res.append(''.join(stack))
			return
		if left_count < n:
			stack.append('(')
			backtrack(stack, n, left_count + 1, right_count)
			stack.pop()
		if left_count > right_count:
			stack.append(')') # why is this always valid?
			backtrack(stack, n, left_count, right_count + 1)
			stack.pop()
	backtrack(stack, n, 0, 0)
	return res
	
def subsets(self, nums: List[int]) -> List[List[int]]:
	def forward():
		# [1,2,3]
		# []
		# [1],[2],[3]
		# [1,2],[1,3],[2,3]
		# [1,2,3]
		def backtrack(curr, index):
			nonlocal res
			if index == len(nums):
				res.append(curr[:])
				return
			# without nums[index]
			backtrack(curr, index + 1)
			# with nums[index]
			curr.append(nums[index])
			backtrack(curr, index + 1)
			curr.pop()
		res = []
		backtrack([], 0)
		return res
	def backward():
		# [1,2,3]
		# []
		# [3],[]
		# [2,3],[2],[3],[]
		# [1,2,3],[1,2],[1,3],[1],[2,3],[2],[3],[]
		def backtrack(index):
			if index == len(nums):
				return [[]]
			res = backtrack(index + 1)
			n = len(res)
			for i in range(n):
				curr = copy.deepcopy(res[i])
				curr.append(nums[index])
				res.append(curr)
			return res
		return backtrack(0)
