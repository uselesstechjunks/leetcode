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
    y = torch.einsum('m,mv->v',weights, V)
    return y

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
