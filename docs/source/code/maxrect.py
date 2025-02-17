class Solution:
    def solve1d(self, nums):
        res, n = 0, len(nums)
        stack = [-1]
        for i in range(n):
            while stack[-1] >= 0 and nums[stack[-1]] > nums[i]:
                curr = nums[stack.pop()]
                l, r = stack[-1]+1, i-1
                res = max(res, curr * (r-l+1))
            stack.append(i)
        while stack[-1] >= 0:
            curr = nums[stack.pop()]
            l, r = stack[-1]+1, n-1
            res = max(res, curr * (r-l+1))
        return res
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        m, n = len(matrix), len(matrix[0]) if len(matrix) > 0 else 0
        mat = [[1 if matrix[i][j] == '1' else 0 for j in range(n)] for i in range(m)]
        print(mat)
        res = self.solve1d(mat[0])
        for i in range(1, m):
            for j in range(n):
                mat[i][j] = 0 if mat[i][j] == 0 else mat[i-1][j] + mat[i][j]
            res = max(res, self.solve1d(mat[i]))
        return res
