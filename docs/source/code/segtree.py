from random import shuffle, seed, randint

class SegmentTree:
    def __init__(self, arr):
        def build(arr, i, l, r):
            if l == r:
                self.tree[i] = arr[l]
            else:
                m = l+(r-l)//2
                build(arr, 2*i, l, m)
                build(arr, 2*i+1, m+1, r)
                self.tree[i] = self.tree[2*i] + self.tree[2*i+1]
        n = len(arr)
        self.tree = [0]*4*n
        self.size = n
        build(arr, 1, 0, n-1)        

    def sum(self, l, r):
        def search(i, tl, tr, l, r):
            if tl == l and tr == r:
                return self.tree[i]
            tm = tl+(tr-tl)//2
            if r <= tm:
                return search(2*i, tl, tm, l, r)
            if l > tm:
                return search(2*i+1, tm+1, tr, l, r)
            return search(2*i, tl, tm, l, tm) + search(2*i+1, tm+1, tr, tm+1, r)
        return search(1, 0, self.size-1, l, r)

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
                self.tree[i] = self.tree[2*i] + self.tree[2*i+1]
        impl(1, 0, self.size-1)

    def __repr__(self):
        return f'{self.tree}'

def test1():
    size = 100
    nums = list(range(size))
    seed(42)
    shuffle(nums)
    #print(nums)
    rsq = SegmentTree(nums)
    #print(rsq)
    for l in range(size):
        for r in range(l, size):
            #print(f'sum({l}:{r})')
            a = rsq.sum(l, r)
            b = 0
            for i in range(l, r+1):
                b += nums[i]
            assert(a == b)
            #print(f'a={a},b={b}')
    for _ in range(10):
        k = randint(0, size-1)
        val = randint(0, 8192)
        nums[k] = val
        #print(nums)
        rsq.update(k, val)
        #print(rsq)
        for l in range(size):
            for r in range(l, size):
                #print(f'sum({l}:{r})')
                a = rsq.sum(l, r)
                b = 0
                for i in range(l, r+1):
                    b += nums[i]
                #assert(a == b)
                #print(f'a={a},b={b}')

if __name__ == '__main__':
    test1()