def findTargetSumWays(self, nums: List[int], target: int) -> int:
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