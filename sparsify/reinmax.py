import torch


class ReinMaxCore(torch.autograd.Function):
    """
    `torch.autograd.Function` implementation of the ReinMax gradient estimator.
    """

    @staticmethod
    # @torch.compile(mode="max-autotune")
    def forward(
        ctx,
        logits: torch.Tensor,
        tau: torch.Tensor,
    ):
        y_soft = logits.softmax(dim=-1)
        sample = torch.multinomial(
            y_soft,
            num_samples=1,
            replacement=True,
        )
        one_hot_sample = torch.zeros_like(
            y_soft, memory_format=torch.legacy_contiguous_format
        ).scatter_(-1, sample, 1.0)
        ctx.save_for_backward(one_hot_sample, logits, y_soft, tau)
        return one_hot_sample, y_soft

    @staticmethod
    @torch.compile(mode="max-autotune")
    def backward(
        ctx,
        grad_at_sample: torch.Tensor,
        grad_at_p: torch.Tensor,
    ):
        one_hot_sample, logits, y_soft, tau = ctx.saved_tensors

        shifted_y_soft = 0.5 * ((logits / tau).softmax(dim=-1) + one_hot_sample)
        grad_at_input_1 = (2 * grad_at_sample) * shifted_y_soft
        grad_at_input_1 = grad_at_input_1 - shifted_y_soft * grad_at_input_1.sum(
            dim=-1, keepdim=True
        )

        grad_at_input_0 = (-0.5 * grad_at_sample + grad_at_p) * y_soft
        grad_at_input_0 = grad_at_input_0 - y_soft * grad_at_input_0.sum(
            dim=-1, keepdim=True
        )

        grad_at_input = grad_at_input_0 + grad_at_input_1
        return grad_at_input - grad_at_input.mean(dim=-1, keepdim=True), None


def reinmax(
    logits: torch.Tensor,
    tau: float,
):
    r"""
    Parameters
    ----------

    logits: ``torch.Tensor``, required
        The input Tensor for the softmax. Note that the softmax
        operation would be conducted along the
        last dimension.
    tau: ``float``, required
        The temperature hyper-parameter. Note note that reinmax
        prefers to set tau >= 1, while gumbel-softmax prefers to set tau < 1.
        For more details, please refer to their paper.

    Returns
    -------
    y_hard: ``torch.Tensor``
        The one-hot sample generated from ``multinomial(softmax(logits))``.
    y_soft: ``torch.Tensor``
        The output of the softmax function, i.e., ``softmax(logits)``.

    Example
    -------
    Below is an example replacing Straight-Through Gumbel-Softmax with ReinMax

    .. code-block:: python
        :linenos:
        :emphasize-added: 2
        :emphasize-removed: 1

        y_hard = torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True)
        y_hard, _ = reinmax.reinmax(logits, tau)

    Below is an example replacing Straight-Through with ReinMax

    .. code-block:: python
        :linenos:
        :emphasize-added: 4
        :emphasize-removed: 1,2,3

        y_hard = one_hot_multinomial(logits.softmax())
        y_soft_tau = (logits/tau).softmax()
        y_hard = y_soft_tau - y_soft_tau.detach() + y_hard
        y_hard, y_soft = reinmax.reinmax(logits, tau)
    """
    if tau < 1:
        raise ValueError(
            "ReinMax prefers to set the temperature (tau) larger or equal to 1."
        )
    shape = logits.size()
    logits = logits.view(-1, shape[-1])
    grad_sample, y_soft = ReinMaxCore.apply(logits, logits.new_empty(1).fill_(tau))
    return grad_sample.view(shape), y_soft.view(shape)
