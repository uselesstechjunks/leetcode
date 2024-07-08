import torch

"""
Rule of thumb:
------------------------------------------------------------------------
(a) dimensions that appear in the output would appear in the outer-loop.
    we'll fill in for these dimensions element-wise.
(b) dimensions that appear in both the inputs (common dimensions) 
    are multiplied element-wise.
(c) dimensions that appear in the inputs but not on the output are summed over.
    this means for dimensions that satisfy both (b) and (c) are first multiplied
    and then summer over.
"""

def test_matmul():
    torch.manual_seed(42)
    X = torch.randn((4,5)) # matrix 4x5
    Y = torch.randn((5,3)) # matrix 5x3
    
    Expected = torch.einsum('ij,jk->ik', X, Y)

    # einsum impl
    Actual = torch.zeros_like(Expected)

    # output dimension
    for i in torch.arange(X.shape[-2]):
        # output dimension
        for k in torch.arange(Y.shape[-1]):
            # loop through common dimension j for element-wise product
            Actual[i,k] = torch.dot(X[i,:], Y[:,k])

    assert(torch.all(torch.isclose(Expected, Actual)))

def test_matmul_reduce_k():
    torch.manual_seed(42)
    X = torch.randn((4,5)) # matrix 4x5
    Y = torch.randn((5,3)) # matrix 5x3
    
    Expected = torch.einsum('ij,jk->i', X, Y)

    # einsum impl
    Actual = torch.zeros_like(Expected)

    # since k is free for Y, we sum over k and cache it
    sum_k_Y = torch.sum(Y,dim=-1)

    # output dimension
    for i in torch.arange(X.shape[-2]):        
        # loop through common dimension j for element-wise product
        Actual[i] = torch.dot(X[i,:], sum_k_Y)

    assert(torch.all(torch.isclose(Expected, Actual)))

def test_matmul_reduce_i():
    torch.manual_seed(42)
    X = torch.randn((4,5)) # matrix 4x5
    Y = torch.randn((5,3)) # matrix 5x3
    
    Expected = torch.einsum('ij,jk->k', X, Y)

    # einsum impl
    Actual = torch.zeros_like(Expected)

    # since i is free for X, we sum over i and cache it
    sum_i_X = torch.sum(X,dim=-2)

    # output dimension
    for k in torch.arange(Y.shape[-1]):        
        # loop through common dimension j for element-wise product
        Actual[k] = torch.dot(sum_i_X, Y[:,k])

    assert(torch.all(torch.isclose(Expected, Actual)))

def test_matmul_reduce_j():
    torch.manual_seed(42)
    X = torch.randn((4,5)) # matrix 4x5
    Y = torch.randn((5,3)) # matrix 5x3
    
    Expected = torch.einsum('ij,jk->j', X, Y)

    # einsum impl
    Actual = torch.zeros_like(Expected)

    # since i is free for X, we sum over i and cache it
    sum_i_X = torch.sum(X,dim=-2)
    # since k is free for Y, we sum over k and cache it
    sum_k_Y = torch.sum(Y,dim=-1)

    # output dimension
    for j in torch.arange(X.shape[-1]):        
        # loop through common dimension j for element-wise product
        Actual[j] = sum_i_X[j] * sum_k_Y[j]

    assert(torch.all(torch.isclose(Expected, Actual)))

def test_matmul_reduce_all():
    torch.manual_seed(42)
    X = torch.randn((4,5)) # matrix 4x5
    Y = torch.randn((5,3)) # matrix 5x3
    
    Expected = torch.einsum('ij,jk->', X, Y)

    # einsum impl
    Actual = torch.zeros_like(Expected)

    # since i is free for X, we sum over i and cache it
    sum_i_X = torch.sum(X,dim=-2)
    # since k is free for Y, we sum over k and cache it
    sum_k_Y = torch.sum(Y,dim=-1)

    # output dimension
    Actual = torch.dot(sum_i_X, sum_k_Y)

    assert(torch.all(torch.isclose(Expected, Actual)))

if __name__ == '__main__':
    test_matmul()
    test_matmul_reduce_k()
    test_matmul_reduce_i()
    test_matmul_reduce_j()
    test_matmul_reduce_all()
