def uniquePaths(self, m: int, n: int) -> int:
	def grid():
		dp = [[0] * n for _ in range(m)]
		for i in range(m):
			dp[i][0] = 1
		for j in range(n):
			dp[0][j] = 1
		for i in range(1, m):
			for j in range(1, n):
				dp[i][j] = dp[i-1][j] + dp[i][j-1]
		return dp[-1][-1]
	
	def optimized():
		# f(i,j) = f(i-1,j) + f(i,j-1)
		# remove the first diemsion
		# f(j) = f(j) + f(j-1) <- as long as f(j-1) is from the current iteration
		# and f(j) is from the previous iteration
		# so we need to fill this from left to right
		dp = [1] * n
		for i in range(1, m):
			for j in range(1, n):
				dp[j] = dp[j] + dp[j-1]
		return dp[-1]
	
	return optimized()