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
        sortedDist = SortedList(list(range(n)), key=lambda u: dist[u])

        for _ in range(n):
            u = sortedDist[0]
            sortedDist.remove(u)
            # check for reachability
            if dist[u] == float('inf'):
                break
            for v, w in adj[u]:
                if v not in sortedDist:
                    continue
                sortedDist.remove(v)
                dist[v] = min(dist[v], dist[u] + w)
                sortedDist.add(v)
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

if __name__ == '__main__':
    test()

"""
### **Test Case 1: Simple Graph**
```plaintext
n = 5
edges = [
    [0, 1, 4],
    [0, 2, 2],
    [1, 2, 5],
    [1, 3, 10],
    [2, 3, 3],
    [3, 4, 1]
]
```
- **Source**: \( 0 \)
- **Expected Output**: {0: 0, 1: 4, 2: 2, 3: 5, 4: 6}

---

### **Test Case 2: Disconnected Graph**
```plaintext
n = 4
edges = [
    [0, 1, 1],
    [0, 2, 4]
]
```
- **Source**: \( 0 \)
- **Expected Output**: {0: 0, 1: 1, 2: 4, 3: -1}

---

### **Test Case 3: Graph with a Single Node**
```plaintext
n = 1
edges = []
```
- **Source**: \( 0 \)
- **Expected Output**: {0: 0}

---

### **Test Case 4: Large Dense Graph**
```plaintext
n = 4
edges = [
    [0, 1, 1],
    [0, 2, 2],
    [0, 3, 3],
    [1, 2, 1],
    [1, 3, 4],
    [2, 3, 1]
]
```
- **Source**: \( 0 \)
- **Expected Output**: {0: 0, 1: 1, 2: 2, 3: 3}

---

### **Test Case 5: Large Sparse Graph**
```plaintext
n = 6
edges = [
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 1],
    [3, 4, 5],
    [4, 5, 2]
]
```
- **Source**: \( 0 \)
- **Expected Output**: {0: 0, 1: 2, 2: 5, 3: 6, 4: 11, 5: 13}

---

### **Test Case 6: Cycle in the Graph**
```plaintext
n = 4
edges = [
    [0, 1, 1],
    [1, 2, 2],
    [2, 3, 3],
    [3, 0, 4]
]
```
- **Source**: \( 0 \)
- **Expected Output**: {0: 0, 1: 1, 2: 3, 3: 6}

---

### **Test Case 7: Multiple Paths with Different Weights**
```plaintext
n = 5
edges = [
    [0, 1, 10],
    [0, 2, 3],
    [1, 2, 1],
    [1, 3, 2],
    [2, 1, 4],
    [2, 3, 8],
    [3, 4, 7],
    [4, 0, 2]
]
```
- **Source**: \( 0 \)
- **Expected Output**: {0: 0, 1: 7, 2: 3, 3: 9, 4: 16}

---

### **Test Case 8: Negative Weights**
(Note: Dijkstra's algorithm assumes no negative edge weights.)
```plaintext
n = 4
edges = [
    [0, 1, 1],
    [1, 2, -2],
    [2, 3, 3]
]
```
- **Source**: \( 0 \)
- **Expected Output**: Not valid for Dijkstra (Bellman-Ford required).

---

### **Test Case 9: Graph with No Path to Some Nodes**
```plaintext
n = 6
edges = [
    [0, 1, 7],
    [0, 2, 9],
    [0, 5, 14],
    [1, 2, 10],
    [2, 3, 11],
    [3, 4, 6]
]
```
- **Source**: \( 0 \)
- **Expected Output**: {0: 0, 1: 7, 2: 9, 3: 20, 4: 26, 5: 14}

---

### **Test Case 10: Edge Case - Graph with Multiple Equal Shortest Paths**
```plaintext
n = 5
edges = [
    [0, 1, 1],
    [0, 2, 1],
    [1, 3, 1],
    [2, 3, 1],
    [3, 4, 1]
]
```
- **Source**: \( 0 \)
- **Expected Output**: {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}

---
"""
