def withoutExtraSpace(self, nums) -> int:
	# use the array index itself as the set using index as lookup key
	# traverse from left to right
	# when we see a negative number, ignore
	# when we see a positive number:
	#   case 0: number at its correct index. do nothing and move right
	#   case 1: number is larger than index, need to move left
	#           overwrite if there's negative at it's correct index
	#           swap if there's a positive at it's correct index
	#           if swapped, that positive must have been from case 2 earlier
	#           if swapped, do not increase the current index
	#   case 2: number if smaller than index, need to move right
	#           if there exists some number on the right should should be sitting here
	#           if would be swapped in a later stage
	# once processed the whole array, traverse from left to right
	# return the first element that's not in its right place
	curr_index, length = 0, len(nums)-1
	while curr_index < length:
		if nums[curr_index] <= 0:
			curr_index += 1
			continue
		correct_index = nums[curr_index]-1
		if correct_index > curr_index and correct_index < length:
			nums[curr_index], nums[correct_index] = nums[correct_index], nums[curr_index]
		elif correct_index < curr_index and correct_index >= 0:
			nums[curr_index], nums[correct_index] = nums[correct_index], nums[curr_index]
	print(nums)
	index = 0
	while index < length:
		if nums[index] <= 0:
			return index + 1
		index += 1
	return length+1

def withExtraSpace(self, nums) -> int:
	nums = set(nums)
	mex = 1
	while mex in nums:
		mex += 1
	return mex

def firstMissingPositive(self, nums: List[int]) -> int:
	return self.withoutExtraSpace(nums)