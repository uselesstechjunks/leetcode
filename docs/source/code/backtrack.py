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
	# n = 4, k = 2
	# []
	# [1] [2] [3] ?? till n-k+1
	# [[1,2],[1,3],[1,4]],[[2,3],[2,4]],[[3,4]]
	def backtrack(curr, start, k):
		nonlocal res
		if k == 0:
			res.append(curr[:])
			return
		for i in range(start, n-k+2):
			curr.append(i)
			backtrack(curr, i+1, k-1)
			curr.pop()
	res, curr = [], []
	backtrack(curr, 1, k)
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