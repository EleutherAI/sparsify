import torch
import torch.nn.functional as F

from sparsify.fused_encoder import fused_encoder


def test_fused_encoder():
    torch.manual_seed(42)

    N, D, M = 8192, 1024, 131_072  # batch_size, input_dim, output_dim
    k = 32

    # Example inputs
    device = "cuda"
    x = torch.randn(N, D, requires_grad=True, device=device)
    W = torch.randn(M, D, requires_grad=True, device=device)
    b = torch.randn(M, requires_grad=True, device=device)

    from time import monotonic

    start = monotonic()

    output = F.relu(F.linear(x, W, b))
    values_naive, indices_naive = torch.topk(output, k, dim=1, sorted=False)
    loss_naive = values_naive.sum()
    loss_naive.backward()

    torch.cuda.synchronize()
    print("Naive time:", monotonic() - start)

    x_grad_naive = x.grad.clone()
    W_grad_naive = W.grad.clone()
    b_grad_naive = b.grad.clone()

    # Zero out gradient buffers
    x.grad = None
    W.grad = None
    b.grad = None

    start = monotonic()

    # Forward pass
    values, indices, _ = fused_encoder(x, W, b, k, "topk")

    # Dummy loss (sum of top-k values)
    loss = values.sum()
    loss.backward()

    torch.cuda.synchronize()
    print("Fused time:", monotonic() - start)

    torch.testing.assert_close(values, values_naive)
    torch.testing.assert_close(indices, indices_naive)
    torch.testing.assert_close(loss, loss_naive)
    torch.testing.assert_close(x.grad, x_grad_naive)
    torch.testing.assert_close(W.grad, W_grad_naive)
    torch.testing.assert_close(b.grad, b_grad_naive)


if __name__ == "__main__":
    test_fused_encoder()
