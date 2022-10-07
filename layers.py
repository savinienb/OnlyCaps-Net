
from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


def _squash(input_tensor, dim=2):
    """
    Applies norm nonlinearity (squash) to a capsule layer.
    Args:
    input_tensor: Input tensor. Shape is [batch, num_channels, num_atoms] for a
        fully connected capsule layer or
        [batch, num_channels, num_atoms, height, width] or
        [batch, num_channels, num_atoms, height, width, depth] for a convolutional
        capsule layer.
    Returns:
    A tensor with same shape as input for output of this layer.
    """
    epsilon = 1e-12
    norm = torch.linalg.norm(input_tensor, dim=dim, keepdim=True)
    norm_squared = norm * norm
    return (input_tensor / (norm + epsilon)) * (norm_squared / (1 + norm_squared))


def _softsquash(input_tensor, dim=2):
    """
    Applies norm nonlinearity (squash) to a capsule layer.
    Args:
    input_tensor: Input tensor. Shape is [batch, num_channels, num_atoms] for a
        fully connected capsule layer or
        [batch, num_channels, num_atoms, height, width] or
        [batch, num_channels, num_atoms, height, width, depth] for a convolutional
        capsule layer.
    Returns:
    A tensor with same shape as input for output of this layer.
    """
    epsilon = 1e-12
    norm = torch.linalg.norm(input_tensor, dim=dim, keepdim=True)
    return (input_tensor / (norm + epsilon)) * F.softmax(norm, dim=1)


def _unitsquash(input_tensor, dim=2):
    """
    Applies norm nonlinearity (squash) to a capsule layer.
    Args:
    input_tensor: Input tensor. Shape is [batch, num_channels, num_atoms] for a
        fully connected capsule layer or
        [batch, num_channels, num_atoms, height, width] or
        [batch, num_channels, num_atoms, height, width, depth] for a convolutional
        capsule layer.
    Returns:
    A tensor with same shape as input for output of this layer.
    """
    epsilon = 1e-12
    norm = torch.linalg.norm(input_tensor, dim=dim, keepdim=True)
    return (input_tensor / (norm + epsilon)) 


def _update_routing(votes, biases, num_routing, routing, squash):
    """
    Sums over scaled votes and applies squash to compute the activations.
    Iteratively updates routing logits (scales) based on the similarity between
    the activation of this layer and the votes of the layer below.
    Args:
        votes: tensor, The transformed outputs of the layer below.
        biases: tensor, Bias variable.
        num_dims: scalar, number of dimmensions in votes. For fully connected
        capsule it is 4, for convolutional 2D it is 6, for convolutional 3D it is 7.
        num_routing: scalar, Number of routing iterations.
    Returns:
        The activation tensor of the output layer after num_routing iterations.
    """
    votes_shape = votes.size()

    logits_shape = list(votes_shape)
    logits_shape[3] = 1
    logits = torch.zeros(logits_shape, requires_grad=False, device=votes.device)
    
    if 'unit' in routing:
        route = 1./(torch.linalg.norm(votes, dim=2, keepdim=True)+1e-5)
        preactivate = torch.sum(votes * route, dim=1) + biases[None, ...]
        if (num_routing>0):
            activation = _squash(preactivate)
    elif 'sabour' in routing:
        for i in range(abs(num_routing)):
            route = F.softmax(logits, dim=2)
            preactivate = torch.sum(votes * route, dim=1) + biases[None, ...]

            if i + 1 < num_routing:
                distances = F.cosine_similarity(preactivate[:, None, ...], votes, dim=3)
                logits = logits + distances[:, :, :, None, ...]
            else:
                activation = (preactivate)
    else:
        raise ValueError  # "routing must be sabour or unit"
    
    if (num_routing>0):
        if "sabour" in squash:
            activation = _squash(preactivate)
        elif "unit" in squash:
            activation = _unitsquash(preactivate)
        elif "soft" in squash:
            activation = _softsquash(preactivate)
        else:
            raise ValueError  # "squash must be sabour, unit or soft"
    else:
        activation = preactivate
    return activation

class DepthwiseConv3d(nn.Module):
    """
    Performs 2D convolution given a 5D input tensor.
    This layer given an input tensor of shape
    `[batch, input_dim, input_atoms, input_height, input_width]` squeezes the
    first two dimmensions to get a 4D tensor as the input of torch.nn.Conv2d. Then
    splits the first dimmension and the second dimmension and returns the 6D
    convolution output.
    Args:
        kernel_size: scalar or tuple, convolutional kernels are [kernel_size, kernel_size].
        input_dim: scalar, number of capsules in the layer below.
        output_dim: scalar, number of capsules in this layer.
        input_atoms: scalar, number of units in each capsule of input layer.
        output_atoms: scalar, number of units in each capsule of output layer.
        stride: scalar or tuple, stride of the convolutional kernel.
        padding: scalar or tuple, zero-padding added to both sides of the input
        dilation: scalar or tuple, spacing between kernel elements
        share_weight: share transformation weight matrices between capsules in lower layer or not
    Returns:
        6D Tensor output of a 2D convolution with shape
        `[batch, input_dim, output_dim, output_atoms, out_height, out_width]`.
    """

    def __init__(
        self,
        kernel_size,
        input_dim,
        output_dim,
        input_atoms=8,
        output_atoms=8,
        stride=2,
        dilation=1,
        padding=0,
        share_weight=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_atoms = input_atoms
        self.output_atoms = output_atoms
        self.share_weight = share_weight

        if self.share_weight:
            self.conv2d = nn.Conv2d(
                input_atoms, output_dim * output_atoms, kernel_size, stride=stride, dilation=dilation, padding=padding
            )
        else:
            self.conv2d = nn.Conv2d(
                input_dim * input_atoms,
                input_dim * output_dim * output_atoms,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                groups=input_dim,
            )
        torch.nn.init.normal_(self.conv2d.weight, std=0.1)

    def forward(self, input_tensor):
        input_shape = input_tensor.size()

        if self.share_weight:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0] * self.input_dim, self.input_atoms, input_shape[-2], input_shape[-1]
            )
        else:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0], self.input_dim * self.input_atoms, input_shape[-2], input_shape[-1]
            )

        conv = self.conv2d(input_tensor_reshaped)
        conv_shape = conv.size()

        conv_reshaped = conv.view(
            input_shape[0], self.input_dim, self.output_dim, self.output_atoms, conv_shape[-2], conv_shape[-1]
        )
        return conv_reshaped


class DepthwiseDeconv3d(nn.Module):
    """
    Performs 2D deconvolution given a 5D input tensor.
    This layer given an input tensor of shape
    `[batch, input_dim, input_atoms, input_height, input_width]` squeezes the
    first two dimmensions to get a 4D tensor as the input of torch.nn.ConvTranspose2d. Then
    splits the first dimmension and the second dimmension and returns the 6D
    convolution output.
    Args:
        kernel_size: scalar or tuple, deconvolutional kernels are [kernel_size, kernel_size].
        input_dim: scalar, number of capsules in the layer below.
        output_dim: scalar, number of capsules in this layer.
        input_atoms: scalar, number of units in each capsule of input layer.
        output_atoms: scalar, number of units in each capsule of output layer.
        stride: scalar or tuple, controls the stride for the cross-correlation.
        padding: scalar or tuple, controls the amount of implicit zero-paddings on both sides for dilation * (kernel_size - 1) - padding number of points
        share_weight: share transformation weight matrices between capsules in lower layer or not
    Returns:
        6D Tensor output of a 2D deconvolution with shape
        `[batch, input_dim, output_dim, output_atoms, out_height, out_width]`.
    """

    def __init__(
        self, kernel_size, input_dim, output_dim, input_atoms=8, output_atoms=8, stride=2, padding=0, share_weight=False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_atoms = input_atoms
        self.output_atoms = output_atoms
        self.share_weight = share_weight

        if self.share_weight:
            self.deconv2d = nn.ConvTranspose2d(input_atoms, output_dim * output_atoms, kernel_size, stride, padding)
        else:
            self.deconv2d = nn.ConvTranspose2d(
                input_dim * input_atoms,
                input_dim * output_dim * output_atoms,
                kernel_size,
                stride,
                padding,
                groups=input_dim,
            )
        torch.nn.init.normal_(self.deconv2d.weight, std=0.1)

    def forward(self, input_tensor):
        input_shape = input_tensor.size()
        if self.share_weight:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0] * self.input_dim, self.input_atoms, input_shape[-2], input_shape[-1]
            )
        else:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0], self.input_dim * self.input_atoms, input_shape[-2], input_shape[-1]
            )

        deconv = self.deconv2d(input_tensor_reshaped)
        deconv_shape = deconv.size()

        deconv_reshaped = deconv.view(
            input_shape[0], self.input_dim, self.output_dim, self.output_atoms, deconv_shape[-2], deconv_shape[-1]
        )
        return deconv_reshaped



class DepthwiseCaps4d(nn.Module):
    """
    MODIFIED DEPTHWISE OPTIMIZED FOR OPERATION EFFICIENCY (SLOWER AND MORE MEMORY)
    Performs 3D convolution given a 6D input tensor.
    This layer given an input tensor of shape
    `[batch, input_dim, input_atoms, input_height, input_width, input_depth]` squeezes the
    first two dimmensions to get a 5D tensor as the input of torch.nn.Conv3d. Then
    splits the first dimmension and the second dimmension and returns the 7D
    convolution output.
    Args:
        kernel_size: scalar or tuple, convolutional kernels are [kernel_size, kernel_size, kernel_size].
        input_dim: scalar, number of capsules in the layer below.
        output_dim: scalar, number of capsules in this layer.
        input_atoms: scalar, number of units in each capsule of input layer.
        output_atoms: scalar, number of units in each capsule of output layer.
        stride: scalar or tuple, stride of the convolutional kernel.
        padding: scalar or tuple, zero-padding added to both sides of the input
        dilation: scalar or tuple, spacing between kernel elements
        share_weight: share transformation weight matrices between capsules in lower layer or not
    Returns:
        7D Tensor output of a 3D convolution with shape
        `[batch, input_dim, output_dim, output_atoms, out_height, out_width, out_depth]`.
    """

    def __init__(
        self,
        kernel_size,
        input_dim,
        output_dim,
        input_atoms=8,
        output_atoms=8,
        stride=2,
        dilation=1,
        padding=0,
        share_weight=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_atoms = input_atoms
        self.output_atoms = output_atoms
        self.share_weight = share_weight
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding


        self.depth = nn.Conv3d(
                self.input_dim , self.input_dim* self.output_dim,
                kernel_size=self.kernel_size,
                stride=self.stride,
                bias=False,
                dilation=dilation,
                padding=self.padding,
                groups=self.input_dim,
            )
        self.point = nn.Conv3d(
                self.input_dim * self.input_atoms * self.output_dim,
                self.input_dim * self.output_atoms * self.output_dim,
                kernel_size=1,
                stride=1,
                dilation=1,
                padding=0,
                groups=self.input_atoms,
            )
        torch.nn.init.normal_(self.depth.weight, std=0.1)
        torch.nn.init.normal_(self.point.weight, std=0.1)

    def forward(self, input_tensor):
        input_shape = input_tensor.size()

        input_tensor_reshaped = input_tensor.view(
                input_shape[0] , self.input_dim, self.input_atoms, input_shape[-3], input_shape[-2], input_shape[-1]
            )
        input_tensor_reshaped = torch.transpose(input_tensor_reshaped,1,2)
        
        input_tensor_reshaped = input_tensor_reshaped.reshape(
                input_shape[0] * self.input_atoms,self.input_dim, input_shape[-3], input_shape[-2], input_shape[-1]
            )
 
        depth = self.depth(input_tensor_reshaped) #output is input atoms, output dim, input dim
        depth_shape = depth.size()
        
        depth_tensor_reshaped = depth.view(
                input_shape[0] ,  self.input_atoms * self.output_dim * self.input_dim, depth_shape[-3], depth_shape[-2], depth_shape[-1])
        point = self.point(depth_tensor_reshaped) #output is output atom, output_dim, input dim 
        conv = point
        
        conv_shape = conv.size()
        conv = conv.view(
            input_shape[0],
            self.output_atoms,self.output_dim,
            self.input_dim,
            conv_shape[-3],
            conv_shape[-2],
            conv_shape[-1],
        )
        conv = torch.transpose(conv,1,3)
        conv_reshaped = conv.view(
            input_shape[0],
            self.input_dim,
            self.output_dim,
            self.output_atoms,
            conv_shape[-3],
            conv_shape[-2],
            conv_shape[-1],
        )
        return conv_reshaped

class DepthwiseCaps4d(nn.Module):
    """
    Performs 3D convolution given a 6D input tensor.
    This layer given an input tensor of shape
    `[batch, input_dim, input_atoms, input_height, input_width, input_depth]` squeezes the
    first two dimmensions to get a 5D tensor as the input of torch.nn.Conv3d. Then
    splits the first dimmension and the second dimmension and returns the 7D
    convolution output.
    Args:
        kernel_size: scalar or tuple, convolutional kernels are [kernel_size, kernel_size, kernel_size].
        input_dim: scalar, number of capsules in the layer below.
        output_dim: scalar, number of capsules in this layer.
        input_atoms: scalar, number of units in each capsule of input layer.
        output_atoms: scalar, number of units in each capsule of output layer.
        stride: scalar or tuple, stride of the convolutional kernel.
        padding: scalar or tuple, zero-padding added to both sides of the input
        dilation: scalar or tuple, spacing between kernel elements
        share_weight: share transformation weight matrices between capsules in lower layer or not
    Returns:
        7D Tensor output of a 3D convolution with shape
        `[batch, input_dim, output_dim, output_atoms, out_height, out_width, out_depth]`.
    """

    def __init__(
        self,
        kernel_size,
        input_dim,
        output_dim,
        input_atoms=8,
        output_atoms=8,
        stride=2,
        dilation=1,
        padding=0,
        share_weight=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_atoms = input_atoms
        self.output_atoms = output_atoms
        self.share_weight = False
        if self.share_weight:
            self.conv3d = nn.Conv3d(
                input_atoms,
                output_dim * output_atoms,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
            )
        else:
            self.conv3d = nn.Conv3d(
                input_dim * input_atoms,
                input_dim * output_dim * output_atoms,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                groups=input_dim,
            )
        
        point_weight = torch.empty(output_atoms,output_dim,input_dim,input_atoms,1,1,1)
        depth_weight = torch.empty(output_atoms,1,1,input_atoms,kernel_size, kernel_size, kernel_size)
        torch.nn.init.normal_(depth_weight, std=0.1)
        torch.nn.init.normal_(point_weight, std=0.1)
        weight_in = point_weight * depth_weight
        weight_in = weight_in.view( output_atoms*output_dim* self.input_dim, self.input_atoms, kernel_size, kernel_size,kernel_size)
        #print(weight_in.shape)
        #print(self.conv3d.weight.shape)
        self.conv3d.weight = nn.Parameter(weight_in)
        #self.conv3d.weight.view( output_atoms*output_dim* self.input_dim, self.input_atoms, kernel_size, kernel_size,kernel_size)
        #torch.nn.init.normal_(self.conv3d.weight, std=0.1)

    def forward(self, input_tensor):
        input_shape = input_tensor.size()
        
        if self.share_weight:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0] * self.input_dim, self.input_atoms, input_shape[-3], input_shape[-2], input_shape[-1]
            )
        else:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0], self.input_dim * self.input_atoms, input_shape[-3], input_shape[-2], input_shape[-1]
            )

        conv = self.conv3d(input_tensor_reshaped)
        conv_shape = conv.size()

        conv_reshaped = conv.view(
            input_shape[0],
            self.input_dim,
            self.output_dim,
            self.output_atoms,
            conv_shape[-3],
            conv_shape[-2],
            conv_shape[-1],
        )
        return conv_reshaped


class Caps4d(nn.Module):
    """
    Performs 3D convolution given a 6D input tensor.
    This layer given an input tensor of shape
    `[batch, input_dim, input_atoms, input_height, input_width, input_depth]` squeezes the
    first two dimmensions to get a 5D tensor as the input of torch.nn.Conv3d. Then
    splits the first dimmension and the second dimmension and returns the 7D
    convolution output.
    Args:
        kernel_size: scalar or tuple, convolutional kernels are [kernel_size, kernel_size, kernel_size].
        input_dim: scalar, number of capsules in the layer below.
        output_dim: scalar, number of capsules in this layer.
        input_atoms: scalar, number of units in each capsule of input layer.
        output_atoms: scalar, number of units in each capsule of output layer.
        stride: scalar or tuple, stride of the convolutional kernel.
        padding: scalar or tuple, zero-padding added to both sides of the input
        dilation: scalar or tuple, spacing between kernel elements
        share_weight: share transformation weight matrices between capsules in lower layer or not
    Returns:
        7D Tensor output of a 3D convolution with shape
        `[batch, input_dim, output_dim, output_atoms, out_height, out_width, out_depth]`.
    """

    def __init__(
        self,
        kernel_size,
        input_dim,
        output_dim,
        input_atoms=8,
        output_atoms=8,
        stride=2,
        dilation=1,
        padding=0,
        share_weight=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_atoms = input_atoms
        self.output_atoms = output_atoms
        self.share_weight = False

        if self.share_weight:
            self.conv3d = nn.Conv3d(
                input_atoms,
                output_dim * output_atoms,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
            )
        else:
            self.conv3d = nn.Conv3d(
                input_dim * input_atoms,
                input_dim * output_dim * output_atoms,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                groups=input_dim,
            )
        torch.nn.init.normal_(self.conv3d.weight, std=0.1)

    def forward(self, input_tensor):
        input_shape = input_tensor.size()
        
        if self.share_weight:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0] * self.input_dim, self.input_atoms, input_shape[-3], input_shape[-2], input_shape[-1]
            )
        else:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0], self.input_dim * self.input_atoms, input_shape[-3], input_shape[-2], input_shape[-1]
            )

        conv = self.conv3d(input_tensor_reshaped)
        conv_shape = conv.size()

        conv_reshaped = conv.view(
            input_shape[0],
            self.input_dim,
            self.output_dim,
            self.output_atoms,
            conv_shape[-3],
            conv_shape[-2],
            conv_shape[-1],
        )
        return conv_reshaped


class ConvSlimCapsule3D(nn.Module):
    """
    Builds a slim convolutional capsule layer.
    This layer performs 3D convolution given 6D input tensor of shape
    `[batch, input_dim, input_atoms, input_height, input_width, input_depth]`. Then refines
    the votes with routing and applies Squash non linearity for each capsule.
    Each capsule in this layer is a convolutional unit and shares its kernel over
    the position grid and different capsules of layer below. Therefore, number
    of trainable variables in this layer is:
        kernel: [kernel_size, kernel_size, kernel_size, input_atoms, output_dim * output_atoms]
        bias: [output_dim, output_atoms]
    Output of a conv3d layer is a single capsule with channel number of atoms.
    Therefore conv_slim_capsule_3d is suitable to be added on top of a conv3d layer
    with num_routing=1, input_dim=1 and input_atoms=conv_channels.
    Args:
        kernel_size: scalar or tuple, convolutional kernels are [kernel_size, kernel_size, kernel_size].
        input_dim: scalar, number of capsules in the layer below.
        output_dim: scalar, number of capsules in this layer.
        input_atoms: scalar, number of units in each capsule of input layer.
        output_atoms: scalar, number of units in each capsule of output layer.
        stride: scalar or tuple, stride of the convolutional kernel.
        padding: scalar or tuple, zero-padding added to both sides of the input
        dilation: scalar or tuple, spacing between kernel elements
        num_routing: scalar, number of routing iterations.
        share_weight: share transformation weight matrices between capsules in lower layer or not
    Returns:
        Tensor of activations for this layer of shape
        `[batch, output_dim, output_atoms, out_height, out_width, out_depth]`
    """

    def __init__(
        self,
        kernel_size,
        input_dim,
        output_dim,
        squash = 'unit',
        routing = 'unit',
        depthwise_capsule = True,
        input_atoms=8,
        output_atoms=8,
        stride=2,
        padding=0,
        dilation=1,
        num_routing=3,
        share_weight=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_atoms = output_atoms
        self.num_routing = num_routing
        self.biases = nn.Parameter(torch.nn.init.constant_(torch.empty(output_dim, output_atoms, 1, 1, 1), 0.1))
        self.depthwise_capsule = depthwise_capsule
        self.routing = routing
        self.squash = squash
        if depthwise_capsule:
            self.depthwise_conv4d = DepthwiseCaps4d(
                kernel_size=kernel_size,
                input_dim=input_dim,
                output_dim=output_dim,
                input_atoms=input_atoms,
                output_atoms=output_atoms,
                stride=stride,
                padding=padding,
                dilation=dilation,
                share_weight=False )
        else:
            self.depthwise_conv4d = Caps4d(
                kernel_size=kernel_size,
                input_dim=input_dim,
                output_dim=output_dim,
                input_atoms=input_atoms,
                output_atoms=output_atoms,
                stride=stride,
                padding=padding,
                dilation=dilation,
                share_weight=False)

    def forward(self, input_tensor):
        votes = self.depthwise_conv4d(input_tensor)
        return _update_routing(votes, self.biases, self.num_routing, self.routing, self.squash)

class Deconv4d(nn.Module):
    """
    Performs 3D deconvolution given a 6D input tensor.
    This layer given an input tensor of shape
    `[batch, input_dim, input_atoms, input_height, input_width, input_depth]` squeezes the
    first two dimmensions to get a 5D tensor as the input of torch.nn.ConvTranspose3d. Then
    splits the first dimmension and the second dimmension and returns the 7D
    convolution output.
    Args:
        kernel_size: scalar or tuple, deconvolutional kernels are [kernel_size, kernel_size, kernel_size].
        input_dim: scalar, number of capsules in the layer below.
        output_dim: scalar, number of capsules in this layer.
        input_atoms: scalar, number of units in each capsule of input layer.
        output_atoms: scalar, number of units in each capsule of output layer.
        stride: scalar or tuple, controls the stride for the cross-correlation.
        padding: scalar or tuple, controls the amount of implicit zero-paddings on both sides for dilation * (kernel_size - 1) - padding number of points
        share_weight: share transformation weight matrices between capsules in lower layer or not
    Returns:
        7D Tensor output of a 3D deconvolution with shape
        `[batch, input_dim, output_dim, output_atoms, out_height, out_width, out_depth]`.
    """

    def __init__(
        self, kernel_size, input_dim, output_dim, input_atoms=8, output_atoms=8, stride=2, padding=0, share_weight=True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_atoms = input_atoms
        self.output_atoms = output_atoms
        self.share_weight = share_weight

        if self.share_weight:
            self.deconv3d = nn.ConvTranspose3d(input_atoms, output_dim * output_atoms, kernel_size, stride, padding)
        else:
            self.deconv3d = nn.ConvTranspose3d(
                input_dim * input_atoms,
                input_dim * output_dim * output_atoms,
                kernel_size,
                stride,
                padding,
                groups=input_dim,
            )
        torch.nn.init.normal_(self.deconv3d.weight, std=0.1)

    def forward(self, input_tensor):
        input_shape = input_tensor.size()
        if self.share_weight:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0] * self.input_dim, self.input_atoms, input_shape[-3], input_shape[-2], input_shape[-1]
            )
        else:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0], self.input_dim * self.input_atoms, input_shape[-3], input_shape[-2], input_shape[-1]
            )
                
        
        deconv = self.deconv3d(input_tensor_reshaped)
        deconv_shape = deconv.size()
        deconv_reshaped = deconv.view(
            input_shape[0],
            self.input_dim,
            self.output_dim,
            self.output_atoms,
            deconv_shape[-3],
            deconv_shape[-2],
            deconv_shape[-1],
        )
        return deconv_reshaped

class DepthwiseDeconv4d(nn.Module):
    """
    Performs 3D deconvolution given a 6D input tensor.
    This layer given an input tensor of shape
    `[batch, input_dim, input_atoms, input_height, input_width, input_depth]` squeezes the
    first two dimmensions to get a 5D tensor as the input of torch.nn.ConvTranspose3d. Then
    splits the first dimmension and the second dimmension and returns the 7D
    convolution output.
    Args:
        kernel_size: scalar or tuple, deconvolutional kernels are [kernel_size, kernel_size, kernel_size].
        input_dim: scalar, number of capsules in the layer below.
        output_dim: scalar, number of capsules in this layer.
        input_atoms: scalar, number of units in each capsule of input layer.
        output_atoms: scalar, number of units in each capsule of output layer.
        stride: scalar or tuple, controls the stride for the cross-correlation.
        padding: scalar or tuple, controls the amount of implicit zero-paddings on both sides for dilation * (kernel_size - 1) - padding number of points
        share_weight: share transformation weight matrices between capsules in lower layer or not
    Returns:
        7D Tensor output of a 3D deconvolution with shape
        `[batch, input_dim, output_dim, output_atoms, out_height, out_width, out_depth]`.
    """

    def __init__(
        self, kernel_size, input_dim, output_dim, input_atoms=8, output_atoms=8, stride=2, padding=0, share_weight=True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_atoms = input_atoms
        self.output_atoms = output_atoms
        self.share_weight = share_weight

        self.depth = nn.ConvTranspose3d(
                self.input_dim , self.input_dim* self.output_dim,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
                padding=padding,
                groups=input_dim,
            )
        self.point = nn.Conv3d(
                self.input_dim * self.input_atoms * self.output_dim,
                self.input_dim * self.output_atoms * self.output_dim,
                kernel_size=1,
                stride=1,
                dilation=1,
                padding=0,
                groups=self.input_atoms,
            )
        torch.nn.init.normal_(self.depth.weight, std=0.1)
        torch.nn.init.normal_(self.point.weight, std=0.1)

    def forward(self, input_tensor):
        input_shape = input_tensor.size()
        if self.share_weight:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0] * self.input_dim, self.input_atoms, input_shape[-3], input_shape[-2], input_shape[-1]
            )
        else:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0], self.input_dim * self.input_atoms, input_shape[-3], input_shape[-2], input_shape[-1]
            )

        input_tensor_reshaped = input_tensor.view(
                input_shape[0] , self.input_dim, self.input_atoms, input_shape[-3], input_shape[-2], input_shape[-1]
            )
        input_tensor_reshaped = torch.transpose(input_tensor_reshaped,1,2)
        
        input_tensor_reshaped = input_tensor_reshaped.reshape(
                input_shape[0] * self.input_atoms,self.input_dim, input_shape[-3], input_shape[-2], input_shape[-1]
            )
 
        depth = self.depth(input_tensor_reshaped) #output is input atoms, output dim, input dim
        depth_shape = depth.size()
        
        depth_tensor_reshaped = depth.view(
                input_shape[0] ,  self.input_atoms * self.output_dim * self.input_dim, depth_shape[-3], depth_shape[-2], depth_shape[-1])
        point = self.point(depth_tensor_reshaped) #output is output atom, output_dim, input dim 
        deconv = point
        
        deconv_shape = deconv.size()
        deconv_reshaped = deconv.view(
            input_shape[0],
            self.output_atoms,self.output_dim,
            self.input_dim,
            deconv_shape[-3],
            deconv_shape[-2],
            deconv_shape[-1],
        )
        
        return deconv_reshaped


class DeconvSlimCapsule3D(nn.Module):
    """
    Builds a slim deconvolutional capsule layer.
    This layer performs 3D deconvolution given 6D input tensor of shape
    `[batch, input_dim, input_atoms, input_height, input_width, input_depth]`. Then refines
    the votes with routing and applies Squash non linearity for each capsule.
    Each capsule in this layer is a deconvolutional unit and shares its kernel over
    the position grid and different capsules of layer below. Therefore, number
    of trainable variables in this layer is:
        kernel: [kernel_size, kernel_size, kernel_size, input_atoms, output_dim * output_atoms]
        bias: [output_dim, output_atoms]
    Args:
        kernel_size: scalar or tuple, deconvolutional kernels are [kernel_size, kernel_size, kernel_size].
        input_dim: scalar, number of capsules in the layer below.
        output_dim: scalar, number of capsules in this layer.
        input_atoms: scalar, number of units in each capsule of input layer.
        output_atoms: scalar, number of units in each capsule of output layer.
        stride: scalar or tuple, controls the stride for the cross-correlation.
        padding: scalar or tuple, controls the amount of implicit zero-paddings on both sides for dilation * (kernel_size - 1) - padding number of points
        num_routing: scalar, number of routing iterations.
        share_weight: share transformation weight matrices between capsules in lower layer or not
    Returns:
        Tensor of activations for this layer of shape
        `[batch, output_dim, output_atoms, out_height, out_width, out_depth]`
    """

    def __init__(
        self,
        kernel_size,
        input_dim,
        output_dim,
        squash = 'unit',
        routing = 'unit',
        depthwise_capsule = True,
        input_atoms=8,
        output_atoms=8,
        stride=2,
        padding=0,
        num_routing=3,
        share_weight=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_routing = num_routing
        self.biases = nn.Parameter(torch.nn.init.constant_(torch.empty(output_dim, output_atoms, 1, 1, 1), 0.1))
        
        self.depthwise_capsule = depthwise_capsule
        self.routing = routing
        self.squash = squash
        if depthwise_capsule:
            self.depthwise_deconv4d = DepthwiseDeconv4d(
                kernel_size=kernel_size,
                input_dim=input_dim,
                output_dim=output_dim,
                input_atoms=input_atoms,
                output_atoms=output_atoms,
                stride=stride,
                padding=padding,
                share_weight=False
            )
        else:
            self.depthwise_deconv4d = Deconv4d(
                kernel_size=kernel_size,
                input_dim=input_dim,
                output_dim=output_dim,
                input_atoms=input_atoms,
                output_atoms=output_atoms,
                stride=stride,
                padding=padding,
                share_weight=False)

    def forward(self, input_tensor):
        votes = self.depthwise_deconv4d(input_tensor)
        return _update_routing(votes, self.biases, self.num_routing,self.routing, self.squash)


class MarginLoss(nn.Module):
    def __init__(self, margin=0.4, downweight=0.5, class_weight=None, reduction="mean"):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.downweight = downweight
        if class_weight is not None:
            self.register_buffer("class_weight", class_weight)
        else:
            self.class_weight = class_weight
        self.reduction = reduction

    def forward(self, raw_logits, labels):
        raw_logits_shape = raw_logits.size()
        num_dims = len(raw_logits_shape)
        if num_dims > 2:
            raw_logits = raw_logits.view(raw_logits_shape[0], raw_logits_shape[1], -1)
            labels = labels.view(raw_logits_shape[0], raw_logits_shape[1], -1)
        logits = raw_logits - 0.5
        positive_cost = labels * F.relu(self.margin - logits) ** 2
        negative_cost = (1 - labels) * F.relu(logits + self.margin) ** 2
        if self.class_weight is not None:
            if num_dims > 2:
                loss = (
                    torch.sum(
                        self.class_weight[None, :, None]
                        * (0.5 * positive_cost + self.downweight * 0.5 * negative_cost),
                        dim=1,
                    )
                    / torch.sum(self.class_weight)
                )
            else:
                loss = torch.sum(
                    self.class_weight[None, :] * (0.5 * positive_cost + self.downweight * 0.5 * negative_cost), dim=1
                ) / torch.sum(self.class_weight)
        else:
            loss = torch.sum(0.5 * positive_cost + self.downweight * 0.5 * negative_cost, dim=1)

        if self.reduction == "mean":
            return torch.mean(loss)
        else:
            pass

