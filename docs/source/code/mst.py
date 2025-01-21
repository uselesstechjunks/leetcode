from ordered_set import OrderedSet

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

class Solution:
    def prim(self, n, m, edges):
        tree = []
        weight = 0
        selected = [False] * (n+1)
        candidates = []
        selected[1] = True

        for _ in range(n-1):
            edge = None
            for i in range(m):
                u, v, w = edges[i]
                if selected[u] ^ selected[v]:
                    if edge is None or edge[2] > w:
                        edge = edges[i]
            if edge is None: break
            u, v, w = edge
            selected[u] = True
            selected[v] = True
            weight += w
            tree.append((u,v))

        if len(tree) != n-1:
            return -1, []
        return weight, tree

    def kruskal(self, n, m, edges):
        edges.sort(key=lambda x:x[2])
        uf = UnionFind(n+1)
        tree = []
        weight = 0
        for u, v, w in edges:
            if uf.find(u) != uf.find(v):
                uf.union(u, v)
                weight += w
                tree.append((u, v))
        if len(tree) != n-1:
            return -1, []
        return weight, tree

def test1():
    solution = Solution()
    n = 4
    m = 5
    edges = [
        [1, 2, 3],
        [1, 3, 4],
        [4, 2, 6],
        [3, 4, 5],
        [1, 4, 1]
    ]
    w, tree = solution.prim(n, m, edges)
    print(w)
    print(tree)
    w, tree = solution.kruskal(n, m, edges)
    print(w)
    print(tree)

def test2():
    solution = Solution()
    n = 3
    m = 1
    edges = [
        [1, 2, 1]
    ]
    w, tree = solution.prim(n, m, edges)
    print(w)
    print(tree)
    w, tree = solution.kruskal(n, m, edges)
    print(w)
    print(tree)

def test3():
    solution = Solution()
    n = 2
    m = 3
    edges = [
        [1, 2, 2],
        [1, 2, 1],
        [1, 2, 3]
    ]
    w, tree = solution.prim(n, m, edges)
    print(w)
    print(tree)
    w, tree = solution.kruskal(n, m, edges)
    print(w)
    print(tree)

def test4():
    solution = Solution()
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
    w, tree = solution.prim(n, m, connections)
    print(w)
    print(tree)
    w, tree = solution.kruskal(n, m, connections)
    print(w)
    print(tree)
    
def test5():
    solution = Solution()
    n = 4
    m = 3
    connections = [
        [1, 2, 5],
        [2, 3, 6],
        [3, 4, 2]
    ]
    w, tree = solution.prim(n, m, connections)
    print(w)
    print(tree)
    w, tree = solution.kruskal(n, m, connections)
    print(w)
    print(tree)

if __name__ == '__main__':
    test1()
    test2()
    test3()
    test4()
    test5()