import torch
import torch.nn as nn
import torch.nn.functional as F

def mh_attn(q,K,V):
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
    o = torch.einsum('hm,hmv->hv',weights, V)
    return o

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
        o = mh_attn(q, K, V)
        y = torch.einsum('hv,hvd->d', o, self.Wo)
        return y

if __name__ == '__main__':
    d = 16
    k = 8
    m = 10
    v = 8
    h = 2

    M = torch.randn((m,d))

    model = MultiHeadAttention(h,d,k,v)
    y = model(M[0],M)
    print(y)
