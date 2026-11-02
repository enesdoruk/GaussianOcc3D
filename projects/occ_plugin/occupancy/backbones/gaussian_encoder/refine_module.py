from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import Scale
import torch.nn as nn, torch
import torch.nn.functional as F
from torch import Tensor
from typing import NamedTuple
from projects.occ_plugin.utils import MixerModel


SIGMOID_MAX = 9.21024
LOGIT_MAX = 0.9999

def safe_sigmoid(tensor):
    tensor = torch.clamp(tensor, -9.21, 9.21)
    return torch.sigmoid(tensor)

def safe_inverse_sigmoid(tensor):
    tensor = torch.clamp(tensor, 1 - LOGIT_MAX, LOGIT_MAX)
    return torch.log(tensor / (1 - tensor))


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers

    
class GaussianPrediction(NamedTuple):
    means: Tensor
    scales: Tensor
    rotations: Tensor
    opacities: Tensor
    semantics: Tensor
    original_means: Tensor = None
    delta_means: Tensor = None



@MODELS.register_module()
class SparseGaussian3DRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        pc_range=None,
        scale_range=None,
        restrict_xyz=False,
        unit_xyz=None,
        refine_manual=None,
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        semantics_activation='softmax',
        xyz_activation="sigmoid",
        scale_activation="sigmoid",
        trans_dim = None,
        n_blocks = None,
        drop_path=0.1,
        **kwargs,
    ):
        super(SparseGaussian3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0
                
        self.output_dim = 10 + int(include_opa) + semantic_dim
        self.semantic_start = 10 + int(include_opa)
        self.semantic_dim = semantic_dim
        self.include_opa = include_opa
        self.semantics_activation = semantics_activation
        self.xyz_act = xyz_activation
        self.scale_act = scale_activation

        self.pc_range = pc_range
        self.scale_range = scale_range
        self.restrict_xyz = restrict_xyz
        self.unit_xyz = unit_xyz
        if restrict_xyz:
            assert unit_xyz is not None
            unit_prob = [unit_xyz[i] / (pc_range[i + 3] - pc_range[i]) for i in range(3)]
            if xyz_activation == "sigmoid":
                unit_prob = [4 * unit_prob[i] for i in range(3)]
            self.unit_sigmoid = unit_prob
        
        assert isinstance(refine_manual, list)
        self.refine_state = refine_manual
        assert all([self.refine_state[i] == i for i in range(len(self.refine_state))])
        
        self.rms_norm = False
        self.fused_add_norm = False
        self.drop_path = drop_path
        self.trans_dim = trans_dim
        
        self.n_blocks = n_blocks
        if self.n_blocks is not None:
            self.pos_embed_xy = nn.Sequential(nn.Linear(2, trans_dim), nn.LayerNorm(trans_dim), nn.ReLU(inplace=True))
            self.pos_embed_xz = nn.Sequential(nn.Linear(2, trans_dim), nn.LayerNorm(trans_dim), nn.ReLU(inplace=True))
            self.pos_embed_yz = nn.Sequential(nn.Linear(2, trans_dim), nn.LayerNorm(trans_dim), nn.ReLU(inplace=True))
            
            # 2. Resampling Layers
            # Encoder Downsample (C -> 2C)
            self.downsample_xy = nn.Sequential(nn.Conv1d(trans_dim, trans_dim * 2, 3, 2, 1), nn.BatchNorm1d(trans_dim * 2), nn.ReLU(inplace=True))
            self.downsample_xz = nn.Sequential(nn.Conv1d(trans_dim, trans_dim * 2, 3, 2, 1), nn.BatchNorm1d(trans_dim * 2), nn.ReLU(inplace=True))
            self.downsample_yz = nn.Sequential(nn.Conv1d(trans_dim, trans_dim * 2, 3, 2, 1), nn.BatchNorm1d(trans_dim * 2), nn.ReLU(inplace=True))
            
            # Decoder Upsample (2C -> C)
            self.upsample_xy = nn.Sequential(nn.ConvTranspose1d(trans_dim * 2, trans_dim, 4, 2, 1), nn.BatchNorm1d(trans_dim), nn.ReLU(inplace=True))
            self.upsample_xz = nn.Sequential(nn.ConvTranspose1d(trans_dim * 2, trans_dim, 4, 2, 1), nn.BatchNorm1d(trans_dim), nn.ReLU(inplace=True))
            self.upsample_yz = nn.Sequential(nn.ConvTranspose1d(trans_dim * 2, trans_dim, 4, 2, 1), nn.BatchNorm1d(trans_dim), nn.ReLU(inplace=True))

            # 3. Mamba Blocks (Explicitly defined for 3 stages)
            # Stage 1: High Res (C)
            self.enc_xy = MixerModel(d_model=trans_dim, n_layer=n_blocks[0], drop_path=0.1)
            self.enc_xz = MixerModel(d_model=trans_dim, n_layer=n_blocks[0], drop_path=0.1)
            self.enc_yz = MixerModel(d_model=trans_dim, n_layer=n_blocks[0], drop_path=0.1)
            
            # Stage 2: Bottleneck (2C)
            self.mid_xy = MixerModel(d_model=trans_dim*2, n_layer=n_blocks[1], drop_path=0.1)
            self.mid_xz = MixerModel(d_model=trans_dim*2, n_layer=n_blocks[1], drop_path=0.1)
            self.mid_yz = MixerModel(d_model=trans_dim*2, n_layer=n_blocks[1], drop_path=0.1)
            
            # Stage 3: High Res (C)
            self.dec_xy = MixerModel(d_model=trans_dim, n_layer=n_blocks[2], drop_path=0.1)
            self.dec_xz = MixerModel(d_model=trans_dim, n_layer=n_blocks[2], drop_path=0.1)
            self.dec_yz = MixerModel(d_model=trans_dim, n_layer=n_blocks[2], drop_path=0.1)

            # Norms
            self.norm_enc = nn.LayerNorm(trans_dim)
            self.norm_mid = nn.LayerNorm(trans_dim*2)
            self.norm_dec = nn.LayerNorm(trans_dim)

            # 4. Projections
            self.xy_proj = nn.Linear(trans_dim, 2)
            self.xz_proj = nn.Linear(trans_dim, 2)
            self.yz_proj = nn.Linear(trans_dim, 2)
            
        
        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            nn.Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim))


    def get_scan_indices(self, anchors, plane):
        x, y, z = anchors[..., 0], anchors[..., 1], anchors[..., 2]
        if plane == 'xy':   
            idx = torch.argsort(y, dim=-1, stable=True)
            sort_idx = torch.gather(torch.argsort(x.gather(-1, idx), dim=-1, stable=True), -1, idx)
        elif plane == 'xz': 
            idx = torch.argsort(z, dim=-1, stable=True)
            sort_idx = torch.gather(torch.argsort(x.gather(-1, idx), dim=-1, stable=True), -1, idx)
        elif plane == 'yz': 
            idx = torch.argsort(z, dim=-1, stable=True)
            sort_idx = torch.gather(torch.argsort(y.gather(-1, idx), dim=-1, stable=True), -1, idx)
        unsort_idx = torch.argsort(sort_idx, dim=-1)
        return sort_idx, unsort_idx
    
    
    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
    ):
        
        x = instance_feature + anchor_embed
        if x.dim() == 2: x = x.unsqueeze(0)
        
        output = self.layers(x)
        
        if self.n_blocks is not None:
            xyz = output[..., :3]
    
            geo = xyz.clone().detach()
            
            xy = self.pos_embed_xy(xyz[..., :2])
            xz = self.pos_embed_xz(xyz[..., [0, 2]])
            yz = self.pos_embed_yz(xyz[..., 1:3])
            
            B, N, C = xy.shape
            
            # ---------------- XY BRANCH ----------------
            sort_idx, unsort_idx = self.get_scan_indices(geo, 'xy')
            gather_idx = sort_idx.unsqueeze(-1).expand(B, N, C)
            restore_idx = unsort_idx.unsqueeze(-1).expand(B, N, C)
            
            feat = torch.gather(xy, 1, gather_idx) # (B, N, C)
            skip = feat.clone() # Save for residual
            feat = self.norm_enc(self.enc_xy(feat))
            feat = self.downsample_xy(feat.permute(0, 2, 1)).permute(0, 2, 1) # (B, N/2, 2C)
            feat = self.norm_mid(self.mid_xy(feat))
            feat = self.upsample_xy(feat.permute(0, 2, 1)).permute(0, 2, 1) # (B, N, C)
            feat = feat + skip # Skip Connection
            feat = self.norm_dec(self.dec_xy(feat)) 
            xy_out = torch.gather(feat, 1, restore_idx)
            
            # ---------------- XZ BRANCH ----------------
            sort_idx, unsort_idx = self.get_scan_indices(geo, 'xz')
            gather_idx = sort_idx.unsqueeze(-1).expand(B, N, C)
            restore_idx = unsort_idx.unsqueeze(-1).expand(B, N, C)
            
            feat = torch.gather(xz, 1, gather_idx)
            skip = feat.clone()
            feat = self.norm_enc(self.enc_xz(feat))
            feat = self.downsample_xz(feat.permute(0, 2, 1)).permute(0, 2, 1)
            feat = self.norm_mid(self.mid_xz(feat))
            feat = self.upsample_xz(feat.permute(0, 2, 1)).permute(0, 2, 1)
            feat = feat + skip
            feat = self.norm_dec(self.dec_xz(feat))
            xz_out = torch.gather(feat, 1, restore_idx)
            
            # ---------------- YZ BRANCH ----------------
            sort_idx, unsort_idx = self.get_scan_indices(geo, 'yz')
            gather_idx = sort_idx.unsqueeze(-1).expand(B, N, C)
            restore_idx = unsort_idx.unsqueeze(-1).expand(B, N, C)
            
            feat = torch.gather(yz, 1, gather_idx)
            skip = feat.clone()
            feat = self.norm_enc(self.enc_yz(feat))
            feat = self.downsample_yz(feat.permute(0, 2, 1)).permute(0, 2, 1)
            feat = self.norm_mid(self.mid_yz(feat))
            feat = self.upsample_yz(feat.permute(0, 2, 1)).permute(0, 2, 1)
            feat = feat + skip
            feat = self.norm_dec(self.dec_yz(feat))
            yz_out = torch.gather(feat, 1, restore_idx)

            # ---------------- FUSION ----------------
            xy_proj = self.xy_proj(xy_out)
            xz_proj = self.xz_proj(xz_out)
            yz_proj = self.yz_proj(yz_out)
            
            x_coord = (xy_proj[..., 0] + xz_proj[..., 0]) / 2 + xyz[...,0]
            y_coord = (xy_proj[..., 1] + yz_proj[..., 0]) / 2 + xyz[...,1]
            z_coord = (xz_proj[..., 1] + yz_proj[..., 1]) / 2 + xyz[...,2]
            
            mamba_xyz = torch.stack([x_coord, y_coord, z_coord], dim=-1)
            output = torch.cat([mamba_xyz, output[..., 3:]], dim=-1)
            
        if self.restrict_xyz:
            delta_xyz_sigmoid = output[..., :3]
            delta_xyz_prob = 2 * safe_sigmoid(delta_xyz_sigmoid) - 1
            delta_xyz = torch.stack([
                delta_xyz_prob[..., 0] * self.unit_sigmoid[0],
                delta_xyz_prob[..., 1] * self.unit_sigmoid[1],
                delta_xyz_prob[..., 2] * self.unit_sigmoid[2]
            ], dim=-1)
            output = torch.cat([delta_xyz, output[..., 3:]], dim=-1)
        
        if len(self.refine_state) > 0:
            refined_part_output = output[..., self.refine_state] + anchor[..., self.refine_state]
            output = torch.cat([refined_part_output, output[..., len(self.refine_state):]], dim=-1)

        if self.xyz_act == "sigmoid":
            xyz = output[..., :3]
        else:
            xyz = output[..., :3].clamp(min=1e-6, max=1-1e-6)
        
        if self.scale_act == "sigmoid":
            scale = output[..., 3:6]
        else:
            scale = output[..., 3:6].clamp(min=1e-6, max=1-1e-6)

        rot = torch.nn.functional.normalize(output[..., 6:10], dim=-1)
        output = torch.cat([xyz, scale, rot, output[..., 10:]], dim=-1)
        
        if self.xyz_act == 'sigmoid':
            xyz = safe_sigmoid(xyz)
        xxx = xyz[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        yyy = xyz[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        zzz = xyz[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        xyz = torch.stack([xxx, yyy, zzz], dim=-1)

        if self.scale_act == 'sigmoid':
            gs_scales = safe_sigmoid(scale)
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales
        
        semantics = output[..., self.semantic_start: (self.semantic_start + self.semantic_dim)]
        if self.semantics_activation == 'softmax':
            semantics = semantics.softmax(dim=-1)
        elif self.semantics_activation == 'softplus':
            semantics = F.softplus(semantics)
        
        gaussian = GaussianPrediction(
            means=xyz,
            scales=gs_scales,
            rotations=rot,
            opacities=safe_sigmoid(output[..., 10: (10 + int(self.include_opa))]),
            semantics=semantics
        )
        return output, gaussian #, semantics
