from collections import deque

class Solution:
	def kahn(self, n, edges):
        graph = [[] for _ in range(n)]
        indeg = [0] * n
        for u, v in edges:
            graph[u].append(v)
            indeg[v] += 1
        queue = deque([u for u in range(n) if indeg[u] == 0])
        discovered = [False] * n
        order = []
        while queue:
            u = queue.popleft()            
            order.append(u)
            for v in graph[u]:
                if discovered[v]: return [] # back edge, not a DAGs
                indeg[v] -= 1
                if indeg[v] == 0:
                    discovered[v] = True
                    queue.append(v)
        return order if len(order) == n else []

    def tarjan(self, n, edges):
        def dfs(graph, discovered, processed, u, order):
            discovered[u] = True
            for v in graph[u]:
                if not discovered[v]:
                    dfs(graph, discovered, processed, v, order)
                elif not processed[v]:
                    return # back edge, not a DAG
            processed[u] = True
            order.insert(0, u)
        graph = [[] for _ in range(n)]
        for u, v in edges:
            graph[u].append(v)
        discovered = [False] * n
        processed = [False] * n
        order = []
        for u in range(n):
            if not discovered[u]:
                dfs(graph, discovered, processed, u, order)
        return order if len(order) == n else []

    def tsort(self, n, edges, algorithm='tarjan'):
        if algorithm == 'tarjan':
            return self.tarjan(n, edges)
        if algorithm == 'kahn':
            return self.kahn(n, edges)