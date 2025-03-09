def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
	def dfs(root):
		nonlocal result
		if not root:
			return
		result.append(root.val)
		dfs(root.left)
		dfs(root.right)
	
	def iter(root):
		nonlocal result
		stack = [root] if root else []
		while stack:
			# stack top == current call stack
			node = stack.pop()
			result.append(node.val)

			""" remmeber to push right first """
			if node.right:
				stack.append(node.right)
			if node.left:
				stack.append(node.left)

def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
	def dfs(root):
		nonlocal result
		if not root:
			return
		dfs(root.left)
		result.append(root.val)
		dfs(root.right)
	
	def push_left(stack, root):
		while root:
			stack.append(root)
			root = root.left
	
	def iter(root):
		nonlocal result
		"""
		we need to simulate how the call stack works
		(a) push as many left elements as possible as init
		(b) popping means processing
		(c) once we pop, we need to move right and then do (a) again
		"""
		stack = []
		""" NOTE we have to do this before entering the loop """
		push_left(stack, root)

		while stack:
			node = stack.pop()
			result.append(node.val)
			node = node.right
			push_left(stack, node)

	result = []
	iter(root)
	return result