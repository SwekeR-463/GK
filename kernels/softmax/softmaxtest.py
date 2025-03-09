import torch
from kernels.softmax.softmax import softmax

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


def test_softmax_kernel(size: tuple, atol = 1e-3, rtol = 1e-3, device=DEVICE):
    # creata input data
    torch.manual_seed(0)
    assert type(size) is tuple and len(size) == 2
    x = torch.randn(size[0], size[1], device=DEVICE)
    
    z_tri = softmax(x)
    z_ref = torch.softmax(x, axis=1)
    
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print(f'The maximum difference between torch and triton is {torch.max(torch.abs(z_ref - z_tri))}')
    print("PASSEDDDDDDDDD")
    
if __name__ == "__main__":
    test_softmax_kernel(size=(1823, 781))