class Solution:
    def shortestPath(self, n: int, edges: List[List[int]], src: int) -> Dict[int, int]:
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
        d = [float('inf')] * n
        visited = [False] * n

        def argmin(arr, visited):
            minIdx = None
            for i in range(len(arr)):
                if not visited[i] and (minIdx is None or arr[minIdx] > arr[i]):
                    minIdx = i
            return minIdx

        d[src] = 0
        for _ in range(n):
            u = argmin(d, visited)
            for v, w in adj[u]:
                d[v] = min(d[v], d[u]+w)
            visited[u] = True
        
        return dict([(u, d[u] if d[u] < float('inf') else -1) for u in range(n)])