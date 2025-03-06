def findTargetSumWays(self, nums: List[int], target: int) -> int:
	"""
	IMPORTANT!! NEED TO KEEP IN MIND THAT RUNNING SUM HAS TO BE IN THE STATE.
	WE CANNOT MEMOIZE/USE DP IF WE KEEP THE RECURSIVE CALL STRUCTURE REDUCING
	TARGET. THAT METHOD IS NOT STATELESS AS COUNT CHANGES ACROSS CALLS.
	"""
	def dp(self, nums: List[int], target: int) -> int:
		""" Note: This still fails a few testcases but it's kept here to convey they key idea """
		""" Note: This can be further optimized removing the first dimension as it only depends on the previous """
		n = len(nums)
		# f(i, j) = number of ways to obtain target j after seeing i nums
		# f(0, 0) = 1
		# f(i+1, j) = f(i, j-nums[i]) + f(i, j+nums[i])
		# range for j: [-sum(abs(nums)), sum(abs(nums))]
		# range for i: 0...n-1
		# gotta be careful with offset
		max_sum = sum([abs(num) for num in nums])
		max_index = 2 * max_sum + 1

		dp = [[0] * (max_index + 1) for _ in range(n + 1)]
		# dp[-][max_sum] actually represents curr sum = 0
		dp[0][max_sum] = 1

		for i in range(n):
			for j in range(nums[i], max_index - nums[i] + 1):
				dp[i+1][j] = dp[i][j-nums[i]] + dp[i][j+nums[i]]

		return dp[n][target + max_sum]
	
	def memoized():
		# the idea is - every time we reach target 0 after seeing n numbers, we increase the count
		# memo dimension: index X target
		# note that the state here is current_sum as opposed to target
		# the reason is - to make the function calls stateless, we cannot use
		# reduced target as every time we call the same target the overall count
		# increases. on the other hand, current sum with the same index remains same
		# and hence can be memoized
		n = len(nums)
		memo = [{} for _ in range(n)]
		# memo[index][target] = count of ways to reach target after index numbers

		def dfs(index, curr, target):
			nonlocal memo, n
			# check if goal is reached
			if index == n:
				return 1 if curr == target else 0
			if curr in memo[index]:
				# already computed, calling the same function again doesn't change the value
				return memo[index][curr]
			else:
				left = dfs(index + 1, curr + nums[index], target)
				right = dfs(index + 1, curr - nums[index], target)
				# memoize for later calls
				memo[index][curr] = left + right
				return memo[index][curr]

		return dfs(0, 0, target)

	def recursive():
		# the idea is - every time we reach target 0 after seeing
		# n numbers, we increase the count
		count = 0

		def dfs(index, target):
			nonlocal count
			n = len(nums)
			# check if goal is reached
			if index == n:
				if target == 0:
					count += 1
			else:
				dfs(index + 1, target - nums[index])
				dfs(index + 1, target + nums[index])
		dfs(0, target)
		return count
	
	return memoized()
