class TreeNode:
    def __init__(self, idx):
        self.idx = idx
        self.left = None
        self.right = None

class CartesianTree:
    def __init__(self, nums):
        def construct(nums):
            stack = []
            for i in range(len(nums)):
                last = None
                while stack and nums[stack[-1].idx] > nums[i]:
                    last = stack.pop()
                curr = TreeNode(idx=i)
                curr.left = last
                if stack:
                    stack[-1].right = curr
                stack.append(curr)
            return stack[0]
        self.root = construct(nums)        

class Solution:
    def cartesianTreeSolution(self, heights):
        root = CartesianTree(heights).root
        def impl(root, l, r):
            nonlocal heights
            if root is None:
                return 0
            minHeight = heights[root.idx]
            currArea = (r-l+1) * minHeight
            leftMaxArea = impl(root.left, l, root.idx-1)
            rightMaxArea = impl(root.right, root.idx+1, r)
            return max(currArea, leftMaxArea, rightMaxArea)
        return impl(root, 0, len(heights)-1)
    
    def simplified(self, heights):
        stack = [-1]
        maxArea = 0
        for i in range(len(heights)):
            while stack[-1] > -1 and heights[stack[-1]] > heights[i]:                
                height = heights[stack.pop()]
                # everything else on stack is smaller than popped
                # popped is the shortest between [top+1, curr-1]
                width = i - stack[-1] - 1
                maxArea = max(maxArea, width*height)
            stack.append(i)
        while stack[-1] > -1:
            height = heights[stack.pop()]
            # top is the shortest between [top+1, end]
            width = len(heights) - stack[-1] - 1
            maxArea = max(maxArea, width*height)
        return maxArea

    def largestRectangleArea(self, heights: List[int]) -> int:
        return self.simplified(heights)