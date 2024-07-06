"""
Sample output:
--------------------------------------
tensor([-6.1852,  4.9277, -7.1767, -5.7050,  3.5045,  1.6870,  5.5653, -2.3704,
         2.2934,  1.3846,  2.3040, -5.1264,  2.2573, -1.0059, -3.2234, -0.6346],
       grad_fn=<ViewBackward0>)
MultiHeadAttention
tensor([ -1.9919, -18.8973, -11.5190,  -3.5315, -25.3034, -26.6213,  26.3794,
        -19.0316, -11.3692, -25.6988,  -5.5753,  11.8805,   2.4946,  27.0999,
         -5.6713,   4.9562], grad_fn=<ViewBackward0>)
MultiHeadAttentionSequential
tensor([ -1.9919, -18.8973, -11.5190,  -3.5315, -25.3034, -26.6213,  26.3794,
        -19.0316, -11.3692, -25.6988,  -5.5753,  11.8805,   2.4946,  27.0999,
         -5.6713,   4.9562], grad_fn=<ViewBackward0>)
MaskedMultiHeadAttentionParallel
tensor([ -1.9919, -18.8973, -11.5190,  -3.5315, -25.3034, -26.6213,  26.3794,
        -19.0316, -11.3692, -25.6988,  -5.5753,  11.8805,   2.4946,  27.0999,
         -5.6712,   4.9562], requires_grad=True)
MultiHeadAttentionSequentialBatched
tensor([ -1.9919, -18.8973, -11.5190,  -3.5315, -25.3034, -26.6213,  26.3794,
        -19.0316, -11.3692, -25.6988,  -5.5753,  11.8805,   2.4946,  27.0999,
         -5.6713,   4.9562], requires_grad=True)
MaskedMultiHeadAttentionParallelBatched
tensor([ -1.9919, -18.8973, -11.5190,  -3.5315, -25.3034, -26.6213,  26.3794,
        -19.0316, -11.3692, -25.6988,  -5.5753,  11.8805,   2.4946,  27.0999,
         -5.6712,   4.9562], requires_grad=True)
MultiQueryAttentionSequentialBatched
tensor([  1.2325, -17.5478,   2.6610,  19.4804,   1.7672, -54.5261, -12.6656,
         -4.2605,   3.6807,   3.2058,  29.4352, -23.4512,   9.5014,  22.9171,
         50.5439,  11.1721], requires_grad=True)
MaskedMultiQueryAttentionParallelBatched
tensor([  1.2325, -17.5478,   2.6610,  19.4804,   1.7672, -54.5261, -12.6656,
         -4.2605,   3.6807,   3.2058,  29.4352, -23.4512,   9.5014,  22.9171,
         50.5439,  11.1721], requires_grad=True)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def attn(q,K,V):
    """
    Args:
        q: [k]
        K: [m,k]
        V: [m,v]
    Returns:
        y: [v]
    """
    logits = torch.einsum('k,mk->m',q,K)
    weights = F.softmax(logits, dim=0)
    y = torch.einsum('m,mv->v', weights, V)
    return y

def mha(q,K,V):
    """
    Args:
        q: [h,k]
        K: [h,m,k]
        V: [h,m,v]
    Returns:
        o: [h,v]
    """
    logits = torch.einsum('hk,hmk->hm',q,K)
    weights = F.softmax(logits, dim=-1)
    o = torch.einsum('hm,hmv->hv', weights, V)
    return o

def mha_par(Q,K,V,mask):
    """
    Args:
        Q: [h,n,k]
        K: [h,m,k]
        V: [h,m,v]
        mask: [n,m]
    Returns:
        O: [h,n,v]
    """
    logits = torch.einsum('hnk,hmk->hnm',Q,K)
    attn_mask = mask == 0
    logits.masked_fill_(attn_mask, float('-inf'))
    weights = F.softmax(logits, dim=-1)
    O = torch.einsum('hnm,hmv->hnv', weights, V)
    return O

def mha_batched(q,K,V):
    """
    Args:
        q: [b,h,k]
        K: [b,h,m,k]
        V: [b,h,m,v]
    Returns:
        O: [b,h,v]
    """
    logits = torch.einsum('bhk,bhmk->bhm',q,K)
    weights = F.softmax(logits, dim=-1)
    o = torch.einsum('bhm,bhmv->bhv', weights, V)
    return o

def mqa_batched(q,K,V):
    """
    Args:
        q: [b,h,k]
        k: [b,m,k]
        v: [b,m,v]
    Returns:
        o: [b,h,n,v]
    """
    logits = torch.einsum('bhk,bmk->bhm', q, K)
    weights = F.softmax(logits, dim=-1)
    o = torch.einsum('bhm,bmv->bhv', weights, V)
    return o

def mha_par_batched(Q,K,V,mask):
    """
    Args:
        Q: [b,h,n,k]
        K: [b,h,m,k]
        V: [b,h,m,v]
        mask: [n,m]
    Returns:
        O: [b,h,n,v]
    """
    logits = torch.einsum('bhnk,bhmk->bhnm', Q, K)
    attn_mask = mask == 0
    logits.masked_fill_(attn_mask, float('-inf'))
    weights = F.softmax(logits, dim=-1)
    O = torch.einsum('bhnm,bhmv->bhnv', weights, V)
    return O

def mqa_par_batched(Q,K,V,mask):
    """
    Args:
        Q: [b,h,n,k]
        K: [b,m,k]
        V: [b,m,v]
        mask: [n,m]
    Returns:
        O: [b,h,n,v]
    """
    logits = torch.einsum('bhnk,bmk->bhnm', Q, K)
    attn_mask = mask == 0
    logits.masked_fill_(attn_mask, float('-inf'))
    weights = F.softmax(logits, dim=-1)
    O = torch.einsum('bhnm,bmv->bhnv', weights, V)
    return O

class Attention(torch.nn.Module):
    def __init__(self, d, k, v):
        super(Attention, self).__init__()
        self.Wq = nn.Parameter(torch.randn(d,k))
        self.Wk = nn.Parameter(torch.randn(d,k))
        self.Wv = nn.Parameter(torch.randn(d,v))

    def forward(self, x, M):
        q = torch.einsum('d,dk->k', x, self.Wq)
        K = torch.einsum('md,dk->mk', M, self.Wk)
        V = torch.einsum('md,dv->mv', M, self.Wv)
        y = attn(q,K,V)
        return y

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, h, d, k, v):
        super(MultiHeadAttention, self).__init__()
        self.Wq = nn.Parameter(torch.randn(h,d,k))
        self.Wk = nn.Parameter(torch.randn(h,d,k))
        self.Wv = nn.Parameter(torch.randn(h,d,v))
        self.Wo = nn.Parameter(torch.randn(h,v,d))

    def forward(self, x, M):
        q = torch.einsum('d,hdk->hk', x, self.Wq)
        K = torch.einsum('md,hdk->hmk', M, self.Wk)
        V = torch.einsum('md,hdv->hmv', M, self.Wv)
        o = mha(q, K, V)
        y = torch.einsum('hv,hvd->d', o, self.Wo)
        return y

class MaskedMultiHeadAttentionParallel(torch.nn.Module):
    def __init__(self, h, d, k, v):
        super(MaskedMultiHeadAttentionParallel, self).__init__()
        self.Wq = nn.Parameter(torch.randn(h,d,k))
        self.Wk = nn.Parameter(torch.randn(h,d,k))
        self.Wv = nn.Parameter(torch.randn(h,d,v))
        self.Wo = nn.Parameter(torch.randn(h,v,d))

    def forward(self, X, M, mask):
        Q = torch.einsum('nd,hdk->hnk', X, self.Wq)
        K = torch.einsum('md,hdk->hmk', M, self.Wk)
        V = torch.einsum('md,hdv->hmv', M, self.Wv)
        O = mha_par(Q, K, V, mask)
        Y = torch.einsum('hnv,hvd->nd', O, self.Wo)
        return Y

class MaskedMultiHeadAttentionParallelBatched(torch.nn.Module):
    def __init__(self, h, d, k, v):
        super(MaskedMultiHeadAttentionParallelBatched, self).__init__()
        self.Wq = nn.Parameter(torch.randn(h,d,k))
        self.Wk = nn.Parameter(torch.randn(h,d,k))
        self.Wv = nn.Parameter(torch.randn(h,d,v))
        self.Wo = nn.Parameter(torch.randn(h,v,d))

    def forward(self, X, M, mask):
        Q = torch.einsum('bnd,hdk->bhnk', X, self.Wq)
        K = torch.einsum('bmd,hdk->bhmk', M, self.Wk)
        V = torch.einsum('bmd,hdv->bhmv', M, self.Wv)
        O = mha_par_batched(Q,K,V,mask)
        Y = torch.einsum('bhnv,hvd->bnd', O, self.Wo)
        return Y

class MultiHeadAttentionSequential(torch.nn.Module):
    def __init__(self, h, d, k, v):
        super(MultiHeadAttentionSequential, self).__init__()
        self.Wq = nn.Parameter(torch.randn(h,d,k))
        self.Wk = nn.Parameter(torch.randn(h,d,k))
        self.Wv = nn.Parameter(torch.randn(h,d,v))
        self.Wo = nn.Parameter(torch.randn(h,v,d))

    def forward(self, x, prev_K, prev_V):
        """
        Args:
            x: [d]
            prev_K: [h,m,k]
            prev_V: [h,m,v]
        Returns:
            y: [d]
            K: [h,m+1,k]
            V: [h,m+1,v]
        """
        q = torch.einsum('d,hdk->hk', x, self.Wq)
        K = torch.cat((prev_K, torch.einsum('d,hdk->hk', x, self.Wk).unsqueeze(-2)), dim=-2)
        V = torch.cat((prev_V, torch.einsum('d,hdv->hv', x, self.Wv).unsqueeze(-2)), dim=-2)
        o = mha(q, K, V)
        y = torch.einsum('hv,hvd->d', o, self.Wo)
        return y, K, V

class MultiHeadAttentionSequentialBatched(torch.nn.Module):
    def __init__(self, h, d, k, v):
        super(MultiHeadAttentionSequentialBatched, self).__init__()
        self.Wq = nn.Parameter(torch.randn(h,d,k))
        self.Wk = nn.Parameter(torch.randn(h,d,k))
        self.Wv = nn.Parameter(torch.randn(h,d,v))
        self.Wo = nn.Parameter(torch.randn(h,v,d))

    def forward(self, x, prev_K, prev_V):
        """
        Args:
            x: [b,d]
            prev_K: [b,h,m,k]
            prev_V: [b,h,m,v]
        Returns:
            y: [b,d]
            K: [b,h,m+1,k]
            V: [b,h,m+1,v]
        """
        q = torch.einsum('bd,hdk->bhk', x, self.Wq)
        K = torch.cat((prev_K, torch.einsum('bd,hdk->bhk', x, self.Wk).unsqueeze(-2)), dim=-2)
        V = torch.cat((prev_V, torch.einsum('bd,hdv->bhv', x, self.Wv).unsqueeze(-2)), dim=-2)
        o = mha_batched(q, K, V)
        y = torch.einsum('bhv,hvd->bd', o, self.Wo)
        return y, K, V

class MaskedMultiQueryAttentionParallelBatched(torch.nn.Module):
    def __init__(self, h, d, k, v):
        super(MaskedMultiQueryAttentionParallelBatched, self).__init__()
        self.Wq = nn.Parameter(torch.randn(h,d,k))
        self.Wk = nn.Parameter(torch.randn(d,k))
        self.Wv = nn.Parameter(torch.randn(d,v))
        self.Wo = nn.Parameter(torch.randn(h,v,d))

    def forward(self, X, M, mask):
        """
        Args:
            X: [b,n,d]
            M: [b,m,d]
            mask: [n,m]
        Returns:
            Y: [b,n,d]
        """
        Q = torch.einsum('bnd,hdk->bhnk', X, self.Wq)
        K = torch.einsum('bmd,dk->bmk', M, self.Wk)
        V = torch.einsum('bmd,dv->bmv', M, self.Wv)
        O = mqa_par_batched(Q,K,V,mask)
        Y = torch.einsum('bhnv,hvd->bnd', O, self.Wo)
        return Y

class MultiQueryAttentionSequentialBatched(torch.nn.Module):
    def __init__(self, h, d, k, v):
        super(MultiQueryAttentionSequentialBatched, self).__init__()
        self.Wq = nn.Parameter(torch.randn(h,d,k))
        self.Wk = nn.Parameter(torch.randn(d,k))
        self.Wv = nn.Parameter(torch.randn(d,v))
        self.Wo = nn.Parameter(torch.randn(h,v,d))

    def forward(self, x, prev_K, prev_V):
        """
        Args:
            x: [b,d]
            prev_K: [b,m,k]
            prev_V: [b,m,v]
        Returns:
            y: [b,d]
            K: [b,m+1,k]
            V: [b,m+1,v]
        """
        q = torch.einsum('bd,hdk->bhk', x, self.Wq)
        K = torch.cat((prev_K, torch.einsum('bd,dk->bk', x, self.Wk).unsqueeze(-2)), dim=-2)
        V = torch.cat((prev_V, torch.einsum('bd,dv->bv', x, self.Wv).unsqueeze(-2)), dim=-2)
        o = mqa_batched(q, K, V)
        y = torch.einsum('bhv,hvd->bd', o, self.Wo)
        return y, K, V

def test_mha(MultiHeadAttention, d, k, v, h, M):
    torch.manual_seed(42)
    model = MultiHeadAttention(h,d,k,v)
    y = model(M[-1],M)
    with torch.no_grad():
        print(f'MultiHeadAttention\n{y}')

def test_mha_seq(MultiHeadAttentionSequential, d, k, v, h, M):
    torch.manual_seed(42)
    model = MultiHeadAttentionSequential(h,d,k,v)
    prev_K = torch.FloatTensor(h,0,k)
    prev_V = torch.FloatTensor(h,0,v)
    y = None

    for x in M:
        y, prev_K, prev_V = model(x, prev_K, prev_V)
    
    with torch.no_grad():
        print(f'MultiHeadAttentionSequential\n{y}')

def test_mha_par(MaskedMultiHeadAttentionParallel, d, k, v, h, M, mask):
    torch.manual_seed(42)
    model = MaskedMultiHeadAttentionParallel(h,d,k,v)
    Y = model(M,M,mask)
    with torch.no_grad():
        print(f'MaskedMultiHeadAttentionParallel\n{Y[-1]}')

def test_mha_par_batched(MaskedMultiHeadAttentionParallelBatched, d, k, v, h, M, mask):
    torch.manual_seed(42)
    model = MaskedMultiHeadAttentionParallelBatched(h,d,k,v)
    X = M.unsqueeze(0)
    Y = model(X,X,mask)
    with torch.no_grad():
        Y = Y.squeeze(0)
        print(f'MaskedMultiHeadAttentionParallelBatched\n{Y[-1]}')

def test_mha_seq_batched(MultiHeadAttentionSequentialBatched, d, k, v, h, M):
    torch.manual_seed(42)
    model = MultiHeadAttentionSequentialBatched(h,d,k,v)
    prev_K = torch.FloatTensor(1,h,0,k)
    prev_V = torch.FloatTensor(1,h,0,v)
    y = None

    for x in M:
        y, prev_K, prev_V = model(x.unsqueeze(0), prev_K, prev_V)

    with torch.no_grad():
        y = y.squeeze(0)
        print(f'MultiHeadAttentionSequentialBatched\n{y}')

def test_mqa_par_batched(MaskedMultiQueryAttentionParallelBatched, d, k, v, h, M, mask):
    torch.manual_seed(42)
    model = MaskedMultiQueryAttentionParallelBatched(h,d,k,v)
    X = M.unsqueeze(0)
    Y = model(X,X,mask)
    with torch.no_grad():
        Y = Y.squeeze(0)
        print(f'MaskedMultiQueryAttentionParallelBatched\n{Y[-1]}')

def test_mqa_seq_batched(MultiQueryAttentionSequentialBatched, d, k, v, h, M):
    torch.manual_seed(42)
    model = MultiQueryAttentionSequentialBatched(h,d,k,v)
    prev_K = torch.FloatTensor(1,0,k)
    prev_V = torch.FloatTensor(1,0,v)
    y = None

    for x in M:
        y, prev_K, prev_V = model(x.unsqueeze(0), prev_K, prev_V)

    with torch.no_grad():
        y = y.squeeze(0)
        print(f'MultiQueryAttentionSequentialBatched\n{y}')

if __name__ == '__main__':
    d = 16
    k = 8
    m = 10
    v = 8
    h = 2

    M = torch.randn((m,d))
    # triangular mask mimicing the decoder
    # same code can be reused for encoder with all bits on
    mask = torch.tril(torch.ones(M.shape[0],M.shape[0]))

    #########################################
    # SHA
    #########################################
    model = Attention(d,k,d)
    y = model(M[0],M)
    print(y)

    #########################################
    # MHA
    #########################################
    test_mha(MultiHeadAttention, d, k, v, h, M)
    test_mha_seq(MultiHeadAttentionSequential, d, k, v, h, M)
    test_mha_par(MaskedMultiHeadAttentionParallel, d, k, v, h, M, mask)    
    test_mha_seq_batched(MultiHeadAttentionSequentialBatched, d, k, v, h, M)
    test_mha_par_batched(MaskedMultiHeadAttentionParallelBatched, d, k, v, h, M, mask)

    #########################################
    # MQA
    #########################################    
    test_mqa_seq_batched(MultiQueryAttentionSequentialBatched, d, k, v, h, M)
    test_mqa_par_batched(MaskedMultiQueryAttentionParallelBatched, d, k, v, h, M, mask)
