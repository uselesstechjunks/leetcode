""" bad approach - needs fixing """
def wordBreak(self, s: str, wordDict: List[str]) -> bool:
	def dp():
		# f(j) = whether the string s[0...j] is valid or not
		# transition rule:
		# f(j) = f(j-k) and s[k+1..j] is a word in dict for any k
		n = len(s)
		words = set(wordDict)
		dp = [False] * n
		for last in range(n): # s[0...last]
			first = last # s[first.last]
			while first >= 0:
				word = s[first:last+1]
				if word in words and (first == 0 or dp[first-1]):
					dp[last] = True
					break
				first -= 1
		return dp[-1]

	def trie():
		pass
	return dp()
