def use_dp():
	# state
	# lis[i] is the longest increasing subsequence ENDING with nums[i]
	# update rule
	# lis[i] = max(lis[j]) + 1 if nums[j] < nums[i] else 1
	n = len(nums)
	dp = [1] * n
	for i in range(1, n):
		for j in range(i):
			if nums[j] < nums[i]:
				dp[i] = max(dp[i], dp[j] + 1)
	return max(dp)

def build_subseq():
	# TODO
	# uses patience sorting