class Solution:
    def bruteforce(self, nums):
        n, res = len(nums), 0
        for i in range(n):
            curr = inf
            for j in range(i,n):
                curr = min(curr, nums[j])
                res = max(res, curr * (j-i+1))
        return res
    def divideAndConquer(self, nums):
        def findMin(l, r):
            nonlocal nums
            currIdx = r
            for i in range(l,r):
                currIdx = i if nums[currIdx] > nums[i] else currIdx
            return currIdx
        def impl(l, r):
            nonlocal nums, res
            if l > r: return -1
            minIdx = findMin(l, r)
            curr = 0
            if minIdx != -1:
                left = impl(l, minIdx-1)
                right = impl(minIdx+1, r)
                curr = max(left, right, nums[minIdx] * (r-l+1))
                res = max(res, curr)
            return curr
        res = 0
        return impl(0, len(nums)-1)
    def monotonicStack(self, nums):
        """ simulates cartesian tree """
        # every time something is popped, it's the min of some range
        # those ranges cover all possible ranges exhaustively
        n, res = len(nums), 0
        stack = [-1]
        for i in range(n):
            while stack[-1] >= 0 and nums[stack[-1]] > nums[i]:
                curr = nums[stack.pop()]
                # popped in the range min of curr top+1 and i-1
                # print(f'range [{stack[-1]+1}:{i-1}] min={curr}')
                res = max(res, curr * (i-stack[-1]-1))
            stack.append(i)
        while stack[-1] >= 0:
            curr = nums[stack.pop()]
            # popped is the range min of top+1 and n-1
            # print(f'> range [{stack[-1]+1}:{n-1}] min={curr}')
            res = max(res, curr * (n-stack[-1]-1))
        return res
    def largestRectangleArea(self, heights: List[int]) -> int:
        return self.monotonicStack(heights)
        # return self.divideAndConquer(heights)
