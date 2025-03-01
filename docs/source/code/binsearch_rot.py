def search(self, nums: List[int], target: int) -> int:
	def direct_search():
		# search range: 0, n-1
		left, right = 0, len(nums) - 1
		while left <= right:
			mid = (left + right) // 2
			if nums[mid] == target:
				return mid
			# nums[left] <= nums[mid] -> min of array lies in [mid + 1, right]
			if nums[left] <= nums[mid]:
				if target < nums[mid]:
					# there can be elements in both parts which are smaller than target
					# if left itself is larger than target, we can eliminate left half
					if nums[left] > target:
						left = mid + 1
					else:
						right = mid - 1
				# if target > nums[mid]:
				else:
					# target has to be on the right side
					left = mid + 1
			# nums[left] > nums[mid] -> min of array lies in [left, mid]
			else:
				if target < nums[mid]:
					# target has to be on the left side
					right = mid - 1
				# if target > nums[mid]:
				else:
					# there are elements in both parts which are larger than target
					# if right itself is smaller than target, we can eliminate right half
					if nums[right] < target:
						right = mid - 1
					else:
						left = mid + 1
		return -1

	def indirect_search():
		def find_pivot():
			left, right = 0, len(nums) - 1
			while left != right and nums[left] > nums[right]:
				mid = (left + right) // 2
				# if nums[left] > nums[mid]:
				if nums[mid] < nums[right]:
					right = mid
				else:
					left = mid + 1
			return left

		def search_with_pivot(pivot_index):
			n = len(nums)
			left, right = 0, n - 1
			while left <= right:
				mid = (left + right) // 2
				pivot = (mid + pivot_index) % n
				if target == nums[pivot]:
					return pivot
				if target < nums[pivot]:
					right = mid - 1
				else:
					left = mid + 1
			return -1

		return search_with_pivot(find_pivot())

	direct = direct_search()
	indirect = indirect_search()

	assert(direct == indirect)
