"""
Sample output:
--------------------------------------
tensor([ -3.0437,  -5.3354,  -2.7996,   4.7690,   6.1953,  -3.1872,   2.4339,
         -4.9126,  -0.5149,  -3.6056,   1.6128, -14.4580,  -2.2639,  -2.7896,
         -0.7055,   6.9216], grad_fn=<ViewBackward0>)
The following outputs should be the exact same with fixed seed.
MultiHeadAttention
tensor([-18.4005,  11.4970,   5.9840,   9.8845, -15.3181,   1.3615,  -2.5959,
         17.3029, -11.3590,  25.8750, -14.3187,  -3.3374,   2.2135, -13.3058,
         -1.9368, -15.8990], grad_fn=<ViewBackward0>)
MultiHeadAttentionSequential
tensor([-18.4005,  11.4970,   5.9840,   9.8845, -15.3181,   1.3615,  -2.5959,
         17.3029, -11.3590,  25.8750, -14.3187,  -3.3374,   2.2135, -13.3058,
         -1.9368, -15.8990], grad_fn=<ViewBackward0>)
MaskedMultiHeadAttentionParallel
tensor([-18.4005,  11.4970,   5.9840,   9.8845, -15.3181,   1.3615,  -2.5959,
         17.3029, -11.3590,  25.8750, -14.3187,  -3.3374,   2.2135, -13.3058,
         -1.9368, -15.8990], requires_grad=True)
MaskedMultiHeadAttentionParallelBatched
tensor([-18.4005,  11.4970,   5.9840,   9.8845, -15.3181,   1.3615,  -2.5959,
         17.3029, -11.3590,  25.8750, -14.3187,  -3.3374,   2.2135, -13.3058,
         -1.9368, -15.8990], requires_grad=True)
MultiHeadAttentionSequentialBatched
tensor([-18.4005,  11.4970,   5.9840,   9.8845, -15.3181,   1.3615,  -2.5959,
         17.3029, -11.3590,  25.8750, -14.3187,  -3.3374,   2.2135, -13.3058,
         -1.9368, -15.8990], requires_grad=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def attn(q,K,V):
    """
    args:
        q: [k]
        K: [m,k]
        V: [m,v]
    returns:
        y: [v]
    """
    logits = torch.einsum('k,mk->m',q,K)
    weights = F.softmax(logits, dim=0)
    y = torch.einsum('m,mv->v', weights, V)
    return y

def mha(q,K,V):
    """
    args:
        q: [h,k]
        K: [h,m,k]
        V: [h,m,v]
    returns:
        o: [h,v]
    """
    logits = torch.einsum('hk,hmk->hm',q,K)
    weights = F.softmax(logits, dim=-1)
    o = torch.einsum('hm,hmv->hv', weights, V)
    return o

def mha_par(Q,K,V,mask):
    """
    args:
        Q: [h,n,k]
        K: [h,m,k]
        V: [h,m,v]
        mask: [n,m]
    returns:
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
    args:
        q: [b,h,k]
        K: [b,h,m,k]
        V: [b,h,m,v]
    returns:
        O: [b,h,v]
    """
    logits = torch.einsum('bhk,bhmk->bhm',q,K)
    weights = F.softmax(logits, dim=-1)
    o = torch.einsum('bhm,bhmv->bhv', weights, V)
    return o

def mha_par_batched(Q,K,V,mask):
    """
    args:
        Q: [b,h,n,k]
        K: [b,h,m,k]
        V: [b,h,m,v]
        mask: [n,m]
    returns:
        O: [b,h,n,v]
    """
    logits = torch.einsum('bhnk,bhmk->bhnm', Q, K)
    attn_mask = mask == 0
    logits.masked_fill_(attn_mask, float('-inf'))
    weights = F.softmax(logits, dim=-1)
    O = torch.einsum('bhnm,bhmv->bhnv', weights, V)
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

    '''
    args:
        x: [d]
        prev_K: [h,m,k]
        prev_V: [h,m,v]
    returns:
        y: [d]
        K: [h,m+1,k]
        V: [h,m+1,v]
    '''
    def forward(self, x, prev_K, prev_V):
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

    '''
    args:
        x: [b,d]
        prev_K: [b,h,m,k]
        prev_V: [b,h,m,v]
    returns:
        y: [b,d]
        K: [b,h,m+1,k]
        V: [b,h,m+1,v]
    '''
    def forward(self, x, prev_K, prev_V):
        q = torch.einsum('bd,hdk->bhk', x, self.Wq)
        K = torch.cat((prev_K, torch.einsum('bd,hdk->bhk', x, self.Wk).unsqueeze(-2)), dim=-2)
        V = torch.cat((prev_V, torch.einsum('bd,hdv->bhv', x, self.Wv).unsqueeze(-2)), dim=-2)
        o = mha_batched(q, K, V)
        y = torch.einsum('bhv,hvd->bd', o, self.Wo)
        return y, K, V

if __name__ == '__main__':
    d = 16
    k = 8
    m = 10
    v = 8
    h = 2

    M = torch.randn((m,d))

    model = Attention(d,k,d)
    y = model(M[0],M)
    print(y)

    print('The following outputs should be the exact same with fixed seed.')

    torch.manual_seed(42)
    model = MultiHeadAttention(h,d,k,v)
    y = model(M[-1],M)
    with torch.no_grad():
        print(f'MultiHeadAttention\n{y}')

    torch.manual_seed(42)
    model = MultiHeadAttentionSequential(h,d,k,v)
    prev_K = torch.FloatTensor(h,0,k)
    prev_V = torch.FloatTensor(h,0,v)
    y = None

    for x in M:
        y, prev_K, prev_V = model(x, prev_K, prev_V)
    
    with torch.no_grad():
        print(f'MultiHeadAttentionSequential\n{y}')

    # triangular mask mimicing the decoder
    # same code can be reused for encoder with all bits on
    mask = torch.tril(torch.ones(M.shape[0],M.shape[0]))
    
    torch.manual_seed(42)
    model = MaskedMultiHeadAttentionParallel(h,d,k,v)
    Y = model(M,M,mask)
    with torch.no_grad():
        print(f'MaskedMultiHeadAttentionParallel\n{Y[-1]}')

    torch.manual_seed(42)
    model = MaskedMultiHeadAttentionParallelBatched(h,d,k,v)
    X = M.unsqueeze(0)
    Y = model(X,X,mask)
    with torch.no_grad():
        Y = Y.squeeze(0)
        print(f'MaskedMultiHeadAttentionParallelBatched\n{Y[-1]}')

    torch.manual_seed(42)
    model = MultiHeadAttentionSequentialBatched(h,d,k,v)
    prev_K = torch.FloatTensor(1,h,0,k)
    prev_V = torch.FloatTensor(1,h,0,v)
    Y = None

    for x in M:
        Y, prev_K, prev_V = model(x.unsqueeze(0), prev_K, prev_V)

    with torch.no_grad():
        y = Y.squeeze(0)
        print(f'MultiHeadAttentionSequentialBatched\n{y}')
