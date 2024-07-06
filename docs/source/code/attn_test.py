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
    weights = F.softmax(logits, dim=1)
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
        return attn(q,K,V)

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

    torch.manual_seed(42)
    model = MultiHeadAttention(h,d,k,v)
    y = model(M[-1],M)
    print(y)

    mask = torch.tril(torch.ones(M.shape[0],M.shape[0]))
    
    torch.manual_seed(42)
    model = MaskedMultiHeadAttentionParallel(h,d,k,v)
    y = model(M,M,mask)
    print(y)

    torch.manual_seed(42)
    model = MaskedMultiHeadAttentionParallelBatched(h,d,k,v)
    X = M.unsqueeze(0)
    y = model(X,X,mask)
    print(y)
