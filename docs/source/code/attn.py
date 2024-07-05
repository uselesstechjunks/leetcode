import torch
import torch.nn.functional as F

class Attention(torch.nn.Module):
	def __init__(self, d, k, v):
		super(Attention, self).__init__()
		self.Wq = torch.nn.Parameter(torch.randn(d, k))
		self.Wk = torch.nn.Parameter(torch.randn(d, k))
		self.Wv = torch.nn.Parameter(torch.randn(d, v))
	
	def forward(self, x, M):
		return self.weighted_dot_prod_attn(x, M, self.Wq, self.Wk, self.Wv)    
	
	def weighted_dot_prod_attn(self, x, M, Wq, Wk, Wv):
		"""
		args:
		x: vector with shape [d]
		M: matrix with shape [m,d]
		Wq: matrix with shape [d,k]
		Wk: matrix with shape [d,k]
		Wv: matrix with shape [d,v]
		returns:
		y: vector with shape [v]
		"""
		q = torch.einsum('d,dk->k', x, Wq)
		K = torch.einsum('md,dk->mk', M, Wk)
		V = torch.einsum('md,dv->mv', M, Wv)
		y = self.dot_prod_attn(q, K, V)
		return y
	
	def dot_prod_attn(self, q, K, V):    
		"""
		args:
		q: vector with shape [k]
		K: matrix with shape [m,k]
		V: matrix with shape [m,v]
		returns:
		y: vector with shape [v]
		"""
		logits = torch.einsum('k,mk->m', q, K)
		weights = F.softmax(logits,dim=0)
		y = torch.einsum('m,mv->v', weights, V)
		return y

if __name__ == '__main__':
	d = 6
	k = 8
	v = 4
	m = 10
	M = torch.randn(m,d)
	attn = Attention(d, k, v)
	y = attn(M[0], M)
	print(y)
