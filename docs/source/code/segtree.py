class SegmentTree:
    def __init__(self, arr, combine):
        self.combine = combine
        self.n = len(arr)
        self.tree = [0] * 4 * self.n
        def build(i, l, r):
            nonlocal arr
            if l == r:
                self.tree[i] = arr[l]
            else:
                m = l+(r-l)//2
                build(2*i, l, m)
                build(2*i+1, m+1, r)
                self.tree[i] = self.combine(self.tree[2*i], self.tree[2*i+1])
        build(1, 0, self.n-1)
    def query(self, l, r):        
        def impl(i, tl, tr, l, r):
            if tl == l and tr == r:
                return self.tree[i]
            tm = tl+(tr-tl)//2
            if r <= tm:
                return impl(2*i, tl, tm, l, r)
            if tm < l:
                return impl(2*i+1, tm+1, tr, l, r)
            return self.combine(impl(2*i, tl, tm, l, tm), impl(2*i+1, tm+1, tr, tm+1, r))
        return impl(1, 0, self.n-1, l, r)
    def update(self, k, val):
        def impl(i, tl, tr):
            nonlocal k, val
            if tl == tr:
                self.tree[i] = val
            else:
                tm = tl+(tr-tl)//2
                if k <= tm:
                    impl(2*i, tl, tm)
                else:
                    impl(2*i+1, tm+1, tr)
                self.tree[i] = self.combine(self.tree[2*i], self.tree[2*i+1])
        impl(1, 0, self.n-1)

class RSQ(SegmentTree):
    def __init__(self, arr):
        super().__init__(arr=arr, combine=lambda x,y: x+y)
    def __repr__(self):
        return f'RSQ=({self.tree})'

class RMQ(SegmentTree):
    def __init__(self, arr):
        super().__init__(arr=arr, combine=lambda x,y: min(x, y))

def test_rsq():
    size = 10
    nums = list(range(size))
    def calculate(l, r):
        nonlocal nums
        res = 0
        for i in range(l, r+1):
            res += nums[i]
        return res
    rsq = RSQ(nums)
    for l in range(size):
        for r in range(l, size):
            a = rsq.query(l, r)
            b = calculate(l, r)
            assert(a == b)
    nums[7] = -10
    rsq.update(7, -10)
    for l in range(size):
        for r in range(l, size):
            a = rsq.query(l, r)
            b = calculate(l, r)
            assert(a == b)

def test_rmq():
    size = 10
    nums = list(range(size))
    def calculate(l, r):
        nonlocal nums
        res = float('inf')
        for i in range(l, r+1):
            res = min(res, nums[i])
        return res
    rmq = RMQ(nums)
    for l in range(size):
        for r in range(l, size):
            a = rmq.query(l, r)
            b = calculate(l, r)
            assert(a == b)
    nums[7] = -10
    rmq.update(7, -10)
    for l in range(size):
        for r in range(l, size):
            a = rmq.query(l, r)
            b = calculate(l, r)
            assert(a == b)

if __name__ == '__main__':
    test_rsq()
    test_rmq()
