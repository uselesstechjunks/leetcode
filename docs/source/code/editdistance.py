def minEditDistance(self, word1: str, word2: str) -> int:
	""" 
	Key idea: While designing transition rule for the three operations
	keep in mind about the next character that we'd check after each of them
	"""
	# f(i, j) = edit distance after reading i characters from
	# word1 and j characters from word2
	# transition rule:
	# f(i, j) = f(i-1, j-1) if word1[i] == word2[j] -> no edit
	#
	# insert: f(i, j) = f(i-1, j) + 1 
	# because after insert, ith character would move to the next pos
	# and word1[i] (inserted char) == word2[j].
	# next character to check: word1[i], word2[j+1]
	#
	# delete: f(i, j) = f(i, j-1) + 1
	# next character to check: word1[i+1], word2[j]
	#
	# replace: f(i, j) = f(i-1, j-1) + 1
	# next character to check: word1[i+1], word2[j+1]
	m, n = len(word1), len(word2)
	dp = [[0] * (n+1) for _ in range(m+1)]
	""" Key: IMPORTANT TO INITIALIZE PROPERLY """
	for i in range(m + 1):
		dp[i][0] = i # all delete
	for j in range(n + 1):
		dp[0][j] = j # all insert
	for i in range(m):
		for j in range(n):
			if word1[i] == word2[j]:
				dp[i+1][j+1] = dp[i][j]
			else:
				dp[i+1][j+1] = min(dp[i][j+1], dp[i+1][j], dp[i][j]) + 1
	return dp[-1][-1]