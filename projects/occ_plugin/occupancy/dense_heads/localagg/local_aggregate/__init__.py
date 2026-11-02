#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch.nn as nn
import torch
from . import _C


def cast_bytes_to_float_preserve_bits(byte_tensor):
    """
    Creates a Float tensor that contains the exact raw bits of the byte_tensor.
    Used when C++ expects Float* but treats it as char*.
    """
    if byte_tensor.dtype != torch.uint8:
        return byte_tensor.float() 
        
    num_bytes = byte_tensor.numel()
    num_floats = (num_bytes + 3) // 4
    
    float_tensor = torch.zeros(num_floats, dtype=torch.float32, device=byte_tensor.device)
    float_tensor.view(torch.uint8)[:num_bytes] = byte_tensor
    
    return float_tensor

def cast_bytes_to_int_preserve_bits(byte_tensor):
    """
    Creates an Int32 tensor that contains the exact raw bits of the byte_tensor.
    Used when C++ expects Int* but treats it as char*.
    """
    if byte_tensor.dtype != torch.uint8:
        return byte_tensor.int() 

    num_bytes = byte_tensor.numel()
    num_ints = (num_bytes + 3) // 4
    
    int_tensor = torch.zeros(num_ints, dtype=torch.int32, device=byte_tensor.device)
    int_tensor.view(torch.uint8)[:num_bytes] = byte_tensor
    
    return int_tensor


class _LocalAggregate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pts,
        points_int,
        means3D,
        means3D_int,
        opacities,
        semantics,
        radii,
        cov3D,
        H, W, D
    ):

        pts = pts.float()
        means3D = means3D.float()
        opacities = opacities.float()
        semantics = semantics.float()
        cov3D = cov3D.float()
        
        points_int = points_int.int()
        means3D_int = means3D_int.int()
        radii = radii.int()

        args = (
            pts,
            points_int,
            means3D,
            means3D_int,
            opacities,
            semantics,
            radii,
            cov3D,
            H, W, D
        )
        
        # Invoke C++/CUDA rasterizer
        num_rendered, logits, geomBuffer, binningBuffer, imgBuffer = _C.local_aggregate(*args)
        
        # Keep relevant tensors for backward
        ctx.num_rendered = num_rendered
        ctx.H = H
        ctx.W = W
        ctx.D = D
        ctx.save_for_backward(
            geomBuffer, 
            binningBuffer, 
            imgBuffer, 
            means3D,
            pts,
            points_int,
            cov3D,
            opacities,
            semantics
        )
        return logits

    @staticmethod
    def backward(ctx, out_grad):
        num_rendered = ctx.num_rendered
        H = ctx.H
        W = ctx.W
        D = ctx.D
        geomBuffer, binningBuffer, imgBuffer, means3D, pts, points_int, cov3D, opacities, semantics = ctx.saved_tensors
        
        geomBuffer = cast_bytes_to_float_preserve_bits(geomBuffer)
        binningBuffer = cast_bytes_to_int_preserve_bits(binningBuffer)
        imgBuffer = cast_bytes_to_float_preserve_bits(imgBuffer)
        
        if pts.dtype != torch.float32: pts = pts.float()
        if means3D.dtype != torch.float32: means3D = means3D.float()
        if cov3D.dtype != torch.float32: cov3D = cov3D.float()
        if opacities.dtype != torch.float32: opacities = opacities.float()
        if semantics.dtype != torch.float32: semantics = semantics.float()
        if points_int.dtype != torch.int32: points_int = points_int.int()
        if out_grad.dtype != torch.float32: out_grad = out_grad.float()

        args = (
            geomBuffer,
            binningBuffer,
            imgBuffer,
            H, W, D,
            num_rendered,
            means3D,
            pts,
            points_int,
            cov3D,
            opacities,
            semantics,
            out_grad)
        
        means3D_grad, opacity_grad, semantics_grad, cov3D_grad = _C.local_aggregate_backward(*args)

        grads = (
            None,
            None,
            means3D_grad,
            None,
            opacity_grad,
            semantics_grad,
            None,
            cov3D_grad,
            None, None, None
        )

        return grads

class LocalAggregator(nn.Module):
    def __init__(self, scale_multiplier, H, W, D, pc_min, grid_size, 
                 inv_softmax=False):
        super().__init__()
        self.scale_multiplier = scale_multiplier
        self.H = H
        self.W = W
        self.D = D
        self.register_buffer('pc_min', torch.tensor(pc_min, dtype=torch.float).unsqueeze(0))
        self.grid_size = grid_size
        self.inv_softmax = inv_softmax      

    def forward(
        self, 
        pts,
        means3D, 
        opacities, 
        semantics, 
        scales, 
        cov3D): 

        assert pts.shape[0] == 1
        pts = pts.squeeze(0)
        assert not pts.requires_grad
        means3D = means3D.squeeze(0).float()
        opacities = opacities.squeeze(0).float()
        semantics = semantics.squeeze(0).float()
        cov3D = cov3D.squeeze(0).float()
        scales = scales.detach().squeeze(0)
        points_int = ((pts - self.pc_min) / self.grid_size).to(torch.int)
        means3D_int = ((means3D.detach() - self.pc_min) / self.grid_size).to(torch.int)
        radii = torch.ceil(scales.max(dim=-1)[0] * self.scale_multiplier / self.grid_size).to(torch.int)
        cov3D = cov3D.flatten(1)[:, [0, 4, 8, 1, 5, 2]]             

        logits = _LocalAggregate.apply(
            pts,
            points_int,
            means3D,
            means3D_int,
            opacities,
            semantics,
            radii,
            cov3D,
            self.H, self.W, self.D
        )

        if not self.inv_softmax:
            return logits 
        else:
            assert False