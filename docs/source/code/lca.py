# Binary tree structure
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def lca1(self, root, p, q):
        if root is None:
            return root
        if root == p or root == q:
            return root
        left = self.lca1(root.left, p, q)
        right = self.lca1(root.right, p, q)
        if left is not None and right is not None:
            return root
        return left if left is not None else right
    def lca2(self, root, p, q):
        if root is None:
            return None
        if p is None and q is None:
            return None
        if root == p:
            if q is None:
                return root
            left = self.lca2(root.left, None, q)
            right = self.lca2(root.right, None, q)
            if left is not None or right is not None:
                return root
        elif root == q:
            if p is None:
                return root
            left = self.lca2(root.left, p, None)
            right = self.lca2(root.right, p, None)
            if left is not None or right is not None:
                return root
        else:
            left = self.lca2(root.left, p, q)
            right = self.lca2(root.right, p, q)
            if left is not None and right is not None:
                return root
        return None

def test1():
    # Test Case
    root = TreeNode(3)
    root.left = TreeNode(5)
    root.right = TreeNode(1)
    root.left.left = TreeNode(6)
    root.left.right = TreeNode(2)
    root.right.left = TreeNode(0)
    root.right.right = TreeNode(8)
    root.left.right.left = TreeNode(7)
    root.left.right.right = TreeNode(4)

    # Nodes
    p = root.left  # Node with value 5
    q = root.right  # Node with value 1

    # Expected Output: 3
    return root, p, q

def test2():
    # Binary tree structure
    root = TreeNode(3)
    root.left = TreeNode(5)
    root.right = TreeNode(1)
    root.left.left = TreeNode(6)
    root.left.right = TreeNode(2)

    # Nodes
    p = root.left  # Node with value 5
    q = TreeNode(10)  # Node with value 10 (not present in the tree)

    # Expected Output: None (as `q` is not in the tree)
    return root, p, q

def test3():
    # Binary tree structure
    root = TreeNode(3)
    root.left = TreeNode(5)
    root.right = TreeNode(5)
    root.left.left = TreeNode(6)
    root.left.right = TreeNode(2)

    # Nodes
    p = root.left  # Node with value 5 (left subtree)
    q = root.right  # Node with value 5 (right subtree)

    # Expected Output: 3
    return root, p, q

if __name__ == '__main__':
    solution = Solution()
    root, p, q = test1()
    res = solution.lca2(root, p, q)
    print(res.val)