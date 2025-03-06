def longestPalindromeSubseq(self, s: str) -> int:
	def lps(s):
		# dp[i][j] = length of longest palindromic subsequence considering s[i...j]
		# s[i]s[i+1...j-1]s[j]
		# dp[i][j] = dp[i+1][j-1] + 2 if s[i] == s[j]
		# dp[i][j] = max(dp[i+1][j], dp[i][j-1])
		# we need to know later rows and earlier columns. 
		# need to fill from bottom to top, left to right
		# optim: only need one extra row
		n = len(s)
		next_row, curr_row = [0] * n, [0] * n
		for i in range(n-1, -1, -1):
			curr_row[i] = 1 # important because every s[i...i] is a palindrome
			for j in range(i+1, n):
				if s[i] == s[j]:
					curr_row[j] = next_row[j-1] + 2
				else:
					curr_row[j] = max(next_row[j], curr_row[j-1])
			for k in range(n):
				next_row[k] = curr_row[k]
				curr_row[k] = 0
		return next_row[-1]

	def lcs(s, t):
		m, n = len(s), len(t)
		prev_row, curr_row = [0] * (n+1), [0] * (n+1)
		for i in range(m):
			for j in range(n):
				if s[i] == t[j]:
					curr_row[j+1] = prev_row[j] + 1
				else:
					curr_row[j+1] = max(prev_row[j+1], curr_row[j])
			for k in range(n+1):
				prev_row[k] = curr_row[k]
				curr_row[k] = 0
		return max(curr_row)
	
	# return lcs(s, s[::-1])

def longestPalindromeSubstr(self, s: str) -> int:
	# dp[i][j] = length of longest palindromic subsequence considering s[i...j]
	# s[i]s[i+1...j-1]s[j]
	# dp[i][j] = dp[i+1][j-1] + 2 if s[i] == s[j] and s[i+1...j-1] is a palindrome => dp[i+1][j-1] == j-i-1
	# dp[i][j] = max(dp[i+1][j], dp[i][j-1])
	# we need to know later rows and earlier columns. 
	# need to fill from bottom to top, left to right
	# optim: only need one extra row
	n = len(s)
	next_row, curr_row = [0] * n, [0] * n
	left, right = 0, 0
	max_len = 1
	for i in range(n-1, -1, -1):
		curr_row[i] = 1 # important because every s[i...i] is a palindrome
		for j in range(i+1, n):
			if s[i] == s[j] and next_row[j-1] == j-i-1:
				curr_row[j] = next_row[j-1] + 2
				if curr_row[j] > max_len:
					max_len = curr_row[j]
					left, right = i, j
			else:
				curr_row[j] = max(next_row[j], curr_row[j-1])
		for k in range(n):
			next_row[k] = curr_row[k]
			curr_row[k] = 0
	return s[left:right + 1]
