import torch
import torch.nn.functional as F

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, h, d, k, v):
        super(MultiHeadAttention, self).__init__()
        self.Wq = torch.nn.Parameter(torch.randn(h, d, k))
        self.Wk = torch.nn.Parameter(torch.randn(h, d, k))
        self.Wv = torch.nn.Parameter(torch.randn(h, d, v))
        self.Wo = torch.nn.Parameter(torch.randn(h, d, v))

    def forward(self, x, M):
         return self.multi_head_attn(x, M, self.Wq, self.Wk, self.Wv, self.Wo)    

    def multi_head_attn(self, x, M, Wq, Wk, Wv, Wo):
        """
        args:
            x: vector with shape [d]
            M: matrix with shape [m,d]
            Wq: matrix with shape [h,d,k]
            Wk: matrix with shape [h,d,k]
            Wv: matrix with shape [h,d,v]
            Wo: matrix with shape [h,d,v]
        returns:
            y: vector with shape [v]
        """
        q = torch.einsum('d,hdk->hk', x, Wq)
        K = torch.einsum('md,hdk->hmk', M, Wk)
        V = torch.einsum('md,hdv->hmv', M, Wv)
        logits = torch.einsum('hk,hmk->hm', q, K)
        weights = F.softmax(logits,dim=1)
        o = torch.einsum('hm,hmv->hv', weights, V)
        y = torch.einsum('hv,hdv->d', o, Wo)
        return y


if __name__ == '__main__':
    d = 6
    k = 8
    v = 4
    m = 10
    h = 32

    M = torch.randn(m,d)

    attn = MultiHeadAttention(h, d, k, v)
    y = attn(M[0], M)
    print(y)
