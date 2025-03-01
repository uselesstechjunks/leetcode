def findMin(self, nums: List[int]) -> int:
	left, right = 0, len(nums)-1
	while left < right and nums[left] > nums[right]:
		mid = (left + right) // 2
		if nums[left] > nums[mid]:
			right = mid
		else:
			left = mid + 1
	return nums[left]