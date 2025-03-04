import torch

import lovely_tensors as lt
lt.monkey_patch()


class Step(torch.autograd.Function):
    """
    Heaviside step function with custom backwards pass
    """

    @staticmethod
    def forward(ctx, pre_acts, threshold):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(pre_acts, threshold)
        return (pre_acts > threshold).sum((0, 1))

    @staticmethod
    def rectangle(x):
        """Kernel function for straight-through estimator"""
        return ((x > -0.5) & (x < 0.5)).to(x.dtype)

    @staticmethod
    def backward(ctx, output_grad, bandwidth = 0.001):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        pre_acts, threshold = ctx.saved_tensors
        
        # We don’t apply STE to x input
        x_grad = 0.0 * output_grad  

        # Pseudo-derivative of the Dirac delta component of the Heaviside function
        threshold_grad = (
            -(1.0 / bandwidth)
            * Step.rectangle((pre_acts - threshold) / bandwidth)
            * output_grad
        )

        print("threshold grad", threshold_grad, "x_grad", x_grad)

        return x_grad, threshold_grad


class JumpReLU(torch.autograd.Function):
    """
    JumpReLU function with custom backwards pass
    """

    @staticmethod
    def forward(ctx, pre_acts, threshold):
        mask = pre_acts > threshold

        did_fire = (mask.sum(0) > 0).detach()
        out = pre_acts * mask

        ctx.save_for_backward(pre_acts, threshold)
        return out, did_fire

    @staticmethod
    def backward(ctx, output_grad, did_fire_grad):
        bandwidth = 0.001

        pre_acts, threshold = ctx.saved_tensors

        # We don’t apply STE to x input
        pre_acts_grad = (pre_acts > threshold) * output_grad

        # Pseudo-derivative of the Dirac delta component of the JumpRelU function
        threshold_grad = (
            -(threshold / bandwidth)
            * Step.rectangle((pre_acts - threshold) / bandwidth)
            * output_grad
        )

        print("threshold", threshold, "bandwidth", bandwidth, "pre_acts", pre_acts, "output_grad", output_grad, "did_fire_grad", did_fire_grad)
        print("threshold grad", threshold_grad, "pre_acts_grad", pre_acts_grad)

        return pre_acts_grad, threshold_grad
            