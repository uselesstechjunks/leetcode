import heapq
from sortedcontainers import SortedList
from typing import List, Dict

class Dijkstra:
    def construct(self, n, edges):
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
        return adj

    def basic_impl(self, n: int, edges: List[List[int]], src: int) -> Dict[int, int]:
        """
        In the basic implementation, the runtime is O((V+E)*V)
        """
        adj = self.construct(n, edges)
        dist = [float('inf')] * n
        visited = [False] * n

        def argmin(arr):
            nonlocal visited
            minIdx = None
            for i in range(len(arr)):
                if not visited[i] and (minIdx is None or arr[minIdx] > arr[i]):
                    minIdx = i
            return minIdx

        dist[src] = 0
        for _ in range(n):
            u = argmin(dist)
            for v, w in adj[u]:
                dist[v] = min(dist[v], dist[u]+w)
            visited[u] = True
        
        return {u: d if d < float('inf') else -1 for u, d in enumerate(dist)}
        
    def minheap_impl(self, n: int, edges: List[List[int]], src: int) -> Dict[int, int]:
        """
        Heap implementation is an improvement over basic implementation in terms of time
        complexity since the runtime is O((V+E) log E). However, since there are no
        easy ways to update the distance value and maintain the heap property at the same
        time, we need to maintain multiple entries for each vertex corresponding to distance
        values coming from relaxing different incident edges on them. Due to the heap property,
        the minimum of them is guaranteed to be processed first, and thereby all the other entries
        for that same vertex would be outdated and shouldn't be used. While it doesn't affect
        the processing, it does affect the space requirement.
        """
        adj = self.construct(n, edges)
        dist = [float('inf')] * n
        dist[src] = 0
        minheap = [(0, src)]
        while minheap:
            du, u = heapq.heappop(minheap)
            # check for reachability
            if du == float('inf'):
                break
            # check for outdated distance values
            if du > dist[u]:
                continue
            for v, w in adj[u]:
                dv = du + w
                if dist[v] > dv:
                    dist[v] = dv
                    heapq.heappush(minheap, (dv, v))
        return {u: d if d < float('inf') else -1 for u, d in enumerate(dist)}

    def tree_impl(self, n: int, edges: List[List[int]], src: int) -> Dict[int, int]:
        adj = self.construct(n, edges)
        dist = [float('inf')] * n
        dist[src] = 0
        sortedDist = SortedList([src], key=lambda x: dist[x])
	
        while sortedDist:
            u = sortedDist[0]
            sortedDist.remove(u)
            for v, w in adj[u]:
                sortedDist.discard(v)
                dist[v] = min(dist[v], dist[u] + w)
                if dist[v] < float('inf'):
                    sortedDist.add(v)
        
        return {u:d if d < float('inf') else -1 for u, d in enumerate(dist)}

class BellmanFord:
    def basic_impl(self, n, edges, src):
        dist = [float('inf')] * n
        dist[src] = 0
        for _ in range(n-1):
            for u, v, w in edges:
                dist[v] = min(dist[v], dist[u] + w)
        return {u: d if d < float('inf') else -1 for u, d in enumerate(dist)}

def test():
    sssp = Dijkstra()
    src = 0
    n = 5
    edges = [
        [0, 1, 1],
        [0, 2, 1],
        [1, 3, 1],
        [2, 3, 1],
        [3, 4, 1]
    ]
    print(sssp.basic_impl(n, edges, src))
    print(sssp.minheap_impl(n, edges, src))
    print(sssp.tree_impl(n, edges, src))
    sssp = BellmanFord()
    print(sssp.basic_impl(n, edges, src))

if __name__ == '__main__':
    test()
