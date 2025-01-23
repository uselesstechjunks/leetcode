from collections import deque

class Solution:
    def bfs1(self, digraph, src):
        n = len(digraph)
        queue = deque([src])
        discovered = [False] * n
        discovered[src] = True
        vertices = []
        edges = []
        while queue:
            u = queue.popleft()
            vertices.append(u)
            for v in digraph[u]:
                edges.append((u, v))
                if not discovered[v]:
                    discovered[v] = True
                    queue.append(v)
        return vertices, edges
    
    def bfs2(self, graph, src):
        n = len(graph)
        queue = deque([src])
        discovered = [False] * n
        processed = [False] * n
        discovered[src] = True
        vertices = []
        edges = []
        while queue:
            u = queue.popleft()            
            vertices.append(u)
            for v in graph[u]:
                if not processed[v]:
                    edges.append((u, v))
                if not discovered[v]:
                    discovered[v] = True
                    queue.append(v)
            processed[u] = True
        return vertices, edges

    def hasCycle(self, digraph, n):
        def dfs(digraph, discovered, processed, u):
            discovered[u] = True
            for v in digraph.get(u, []):
                if not discovered[v]:
                    if dfs(digraph, discovered, processed, v):
                        return True
                elif not processed[v]:
                    return True
            processed[u] = True
            return False
        discovered = [False] * n
        processed = [False] * n
        for u in range(n):
            if not discovered[u] and dfs(digraph, discovered, processed, u):
                return True
        return False
    
    def hasCycleUndirected(self, graph, n):
        def dfs(graph, discovered, parent, u):
            discovered[u] = True
            for v in graph.get(u, []):
                if parent == v:
                    continue
                if discovered[v]:
                    return True
                else:
                    if dfs(graph, discovered, u, v):
                        return True
            return False
        discovered = [False] * n
        for u in range(n):
            if not discovered[u] and dfs(graph, discovered, -1, u):
                return True
        return False

    def isBipartite(self, graph, n):
        colors = [None] * n
        def bfs(graph, colors, src):
            queue = deque([src])
            colors[src] = 0
            while queue:
                u = queue.popleft()
                for v in graph.get(u, []):
                    if colors[v] is None:
                        colors[v] = colors[u] ^ 1
                        queue.append(v)
                    elif colors[v] == colors[u]:
                        return False
            return True
        for u in range(n):
            if colors[u] is None and not bfs(graph, colors, u):
                return False
        return True
    
    def findBCC(self, graph, n):
        return None, None

def test1():
    digraph = {  
        0: [1, 2],  
        1: [3, 4],  
        2: [5, 6],  
        3: [],
        4: [],
        5: [],
        6: [] 
    }  
    start_node = 0
    return digraph, start_node

def test2():
    graph = {  
        0: [1, 3],
        1: [0, 2],
        2: [1, 3],
        3: [0, 2]
    }
    start_node = 0
    return graph, start_node

def cycleDirected():
    digraph, n = {}, 0 # Should return False
    digraph, n = {0: []}, 1  # Should return False
    digraph, n = {0: [0]}, 1  # Should return True
    digraph, n = {0: [1], 1: []}, 2  # Should return False
    digraph, n = {0: [1], 1: [0]}, 2  # Should return True
    digraph, n = {0: [1], 2: [3], 3: []}, 4  # Should return False
    digraph, n = {0: [1], 1: [2], 2: [0], 3: [4], 4: []}, 5  # Should return True
    digraph, n = {0: [1], 1: [2], 3: []}, 4  # Should return False
    digraph, n = {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: [2]}, 5  # Should return True
    return digraph, n

def cycleUndirected():
    graph, n = {}, 0  # Should return False
    graph, n = {0: []}, 1  # Should return False
    graph, n = {0: [0]}, 1  # Should return True
    graph, n = {0: [1], 1: [0]}, 2  # Should return False
    graph, n = {0: [1, 2], 1: [0, 2], 2: [0, 1]}, 3  # Should return True
    graph, n = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}, 4  # Should return False
    graph, n = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2], 4: [5], 5: [4, 6], 6: [5, 4]}, 7  # Should return True
    graph, n = {0: [1, 1], 1: [0, 0]}, 2  # Should return True
    graph, n = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2, 4], 4: [3, 5], 5: [4]}, 6  # Should return True
    graph, n = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}, 6  # Should return False
    return graph, n

def bipartite():
    graph, n = {}, 0 # Expected Output: True
    graph, n = {0: []}, 1 # Expected Output: True
    graph, n = {0: [1], 1: [0]}, 2, # Expected Output: True
    graph, n = {0: [1, 2], 1: [0, 2], 2: [0, 1]}, 3 # Expected Output: False
    graph, n = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}, 4 # Expected Output: True
    graph, n = {0: [1], 1: [0], 2: [3], 3: [2]}, 4 # Expected Output: True
    graph, n = {0: [1], 1: [0], 2: [3, 4], 3: [2, 4], 4: [2, 3]}, 5 # Expected Output: False
    graph, n = {0: [1, 2, 3, 4], 1: [0], 2: [0], 3: [0], 4: [0]}, 5 # Expected Output: True
    graph, n = {0: [0]}, 1 # Expected Output: False
    graph, n = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2], 4: [5], 5: [4, 6], 6: [5]}, 7 # Expected Output: True
    return graph, n

def bcc():
    graph, n = {}, 0 #  Expected Output: Bridges: [], Articulation Points: []
    graph, n = {0: []}, 1 #  Expected Output: Bridges: [], Articulation Points: []
    graph, n = {0: [1], 1: [0]}, 2 #  Expected Output: Bridges: [(0, 1)], Articulation Points: []
    graph, n = {0: [1], 1: [0, 2], 2: [1]}, 3 #  Expected Output: Bridges: [(1, 2), (0, 1)], Articulation Points: [1]
    graph, n = {0: [1, 2], 1: [0, 2], 2: [0, 1]}, 3 #  Expected Output: Bridges: [], Articulation Points: []
    graph, n = {0: [1], 1: [0, 2], 2: [1], 3: [4], 4: [3]}, 5 #  Expected Output: Bridges: [(0, 1), (1, 2), (3, 4)], Articulation Points: []
    graph, n = {0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: [1, 4], 4: [3]}, 5 #  Expected Output: Bridges: [(1, 3), (3, 4)], Articulation Points: [1, 3]
    graph, n = {0: [1], 1: [0, 2], 2: [1], 3: [4], 4: [3, 5], 5: [4]}, 6 #  Expected Output: Bridges: [(0, 1), (1, 2), (3, 4), (4, 5)], Articulation Points: []
    graph, n = {0: [1, 2, 3, 4], 1: [0], 2: [0], 3: [0], 4: [0]}, 5 #  Expected Output: Bridges: [(0, 1), (0, 2), (0, 3), (0, 4)], Articulation Points: [0]
    graph, n = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2, 4], 4: [3, 5], 5: [4]}, 6 #  Expected Output: Bridges: [(4, 5)], Articulation Points: [3, 4]
    return graph, n

if __name__ == '__main__':
    solution = Solution()
    digraph, start_node = test1()
    res = solution.bfs1(digraph, start_node)
    print(res)
    graph, start_node = test2()
    res = solution.bfs2(graph, start_node)
    print(res)
    digraph, n = cycleDirected()
    res = solution.hasCycle(digraph, n)
    print(res)
    graph, n = cycleUndirected()
    res = solution.hasCycleUndirected(graph, n)
    print(res)
    graph, n = bipartite()
    res = solution.isBipartite(graph, n)
    print(res)
    graph, n = bcc()
    res, res2 = solution.findBCC(graph, n)
    print(res, res2)