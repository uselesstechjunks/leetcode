import torch
import torch.nn as nn
import torch.nn.functional as F

def mh_attn_parallel(Q,K,V,mask):
    """
    args:
        Q: [h,n,k]
        K: [h,m,k]
        V: [h,m,v]
        mask: [n,m]
    returns:
        o: [h,n,v]
    """
    logits = torch.einsum('hnk,hmk->hnm',Q,K)
    attn_mask = mask == 0
    logits.masked_fill_(attn_mask, float('-inf'))
    weights = F.softmax(logits, dim=-1)
    o = torch.einsum('hnm,hmv->hnv', weights, V)
    return o

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
        o = mh_attn_parallel(Q, K, V, mask)
        y = torch.einsum('hnv,hvd->nd', o, self.Wo)
        return y

if __name__ == '__main__':
    d = 16
    k = 8
    m = 10
    v = 8
    h = 2

    M = torch.randn((m,d))

    mask = torch.tril(torch.ones(M.shape[0],M.shape[0])) # lower triangular matrix
    model = MaskedMultiHeadAttentionParallel(h,d,k,v)
    y = model(M,M,mask) # self attention
    print(y)
