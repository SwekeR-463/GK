import torch
from rope import rope

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


def test_rope_kernel(batch_size=2, seq_len=64, embed_dim=128, atol=1e-2, rtol=1e-1, device=DEVICE):
    torch.manual_seed(0)
    
    input = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float16, device=device)
    theta = torch.randn(embed_dim // 2, dtype=torch.float32, device=device)

    output_tri = rope(input, theta)

    output_ref = input.clone()
    for b in range(batch_size):
        for t in range(seq_len):
            for d in range(0, embed_dim, 2):
                angle = t * theta[d//2]
                cos_val = torch.cos(angle)
                sin_val = torch.sin(angle)
                x0, x1 = input[b, t, d], input[b, t, d+1]
                output_ref[b, t, d] = x0 * cos_val - x1 * sin_val
                output_ref[b, t, d+1] = x0 * sin_val + x1 * cos_val

    torch.testing.assert_close(output_tri, output_ref, atol=atol, rtol=rtol)
    print(f'Max difference: {torch.max(torch.abs(output_ref - output_tri))}')
    print("PASSED")

if __name__ == "__main__":
    test_rope_kernel()