"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        # 卷积情况下的初始化（shape不为None）
        # fan_in: 输入特征图数（I） * 感受野大小（K*K），fan_out: 输出特征图数（O） * 感受野大小，shape: 卷积核大小（KKIO）
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        kernel_shape = (kernel_size, kernel_size, in_channels, out_channels)
        self.weight = Parameter(
            init.kaiming_uniform(fan_in, fan_out, shape=kernel_shape, device=device, dtype=dtype, requires_grad=True)
        )
        if bias:
            bound = 1.0 / (in_channels * kernel_size ** 2) ** 0.5
            self.bias = Parameter(
                init.rand(out_channels, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True)
            )
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Calculate the appropriate padding to ensure input and output dimensions are the same
        # (in the stride=1 case, anyways)
        padding = (self.kernel_size - 1) // 2
        # NCHW -> NHWC
        conv_res = ops.conv(x.transpose(axes=(1, 2)).transpose(axes=(2, 3)), self.weight,
                            stride=self.stride, padding=padding)
        if self.bias:
            # (out_channels,) -> (N, H, W, out_channels)
            conv_res += self.bias.broadcast_to(conv_res.shape)
        # NHWC -> NCHW
        return conv_res.transpose(axes=(2, 3)).transpose(axes=(1, 2))
        ### END YOUR SOLUTION