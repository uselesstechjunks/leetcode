import heapq

class UnionFind:
    def __init__(self, n):
        self.parents = list(range(n))
        self.size = [1] * n
        self.n = n
    def find(self, x):
        if self.parents[x] != x:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]
    def union(self, x, y):
        parent_x = self.find(x)
        parent_y = self.find(y)
        if self.size[parent_x] > self.size[parent_y]:
            self.parents[parent_y] = parent_x
            self.size[parent_x] += self.size[parent_y]
            parent = parent_x
        else:
            self.parents[parent_x] = parent_y
            self.size[parent_y] += self.size[parent_x]
            parent = parent_y
        return parent

class MST:
    def prim(self, n, m, edges):
        tree = []
        weight = 0
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u-1].append((v-1, w))
            adj[v-1].append((u-1, w))
        explored = [False] * n
        explored[0] = True
        minheap = [(w, 0, v) for v, w in adj[0]]
        heapq.heapify(minheap)
        while minheap:            
            w, u, v = heapq.heappop(minheap)
            if explored[v]:
                continue
            explored[v] = True
            weight += w
            tree.append((u+1, v+1))
            for y, w in adj[v]:
                if not explored[y]:
                    heapq.heappush(minheap, (w, v, y))
        if len(tree) == n-1:
            return weight, tree
        return -1, []

    def kruskal(self, n, m, edges):
        tree = []
        weight = 0
        uf = UnionFind(n+1)
        for u, v, w in sorted(edges, key=lambda x:x[2]):
            if uf.find(u) != uf.find(v):
                uf.union(u, v)
                tree.append((u, v))
                weight += w
        if len(tree) == n-1:
            return weight, tree
        return -1, []

def test1():
    mst = MST()
    n = 4
    m = 5
    edges = [
        [1, 2, 3],
        [1, 3, 4],
        [4, 2, 6],
        [3, 4, 5],
        [1, 4, 1]
    ]
    w, tree = mst.prim(n, m, edges)
    print(w)
    print(tree)
    w, tree = mst.kruskal(n, m, edges)
    print(w)
    print(tree)

def test2():
    mst = MST()
    n = 3
    m = 1
    edges = [
        [1, 2, 1]
    ]
    w, tree = mst.prim(n, m, edges)
    print(w)
    print(tree)
    w, tree = mst.kruskal(n, m, edges)
    print(w)
    print(tree)

def test3():
    mst = MST()
    n = 2
    m = 3
    edges = [
        [1, 2, 2],
        [1, 2, 1],
        [1, 2, 3]
    ]
    w, tree = mst.prim(n, m, edges)
    print(w)
    print(tree)
    w, tree = mst.kruskal(n, m, edges)
    print(w)
    print(tree)

def test4():
    mst = MST()
    n = 5
    m = 6
    connections = [
        [1, 2, 4],
        [2, 3, 3],
        [3, 4, 2],
        [4, 5, 6],
        [1, 5, 7],
        [2, 5, 1]
    ]
    w, tree = mst.prim(n, m, connections)
    print(w)
    print(tree)
    w, tree = mst.kruskal(n, m, connections)
    print(w)
    print(tree)
    
def test5():
    mst = MST()
    n = 4
    m = 3
    connections = [
        [1, 2, 5],
        [2, 3, 6],
        [3, 4, 2]
    ]
    w, tree = mst.prim(n, m, connections)
    print(w)
    print(tree)
    w, tree = mst.kruskal(n, m, connections)
    print(w)
    print(tree)

if __name__ == '__main__':
    test1()
    test2()
    test3()
    test4()
    test5()
