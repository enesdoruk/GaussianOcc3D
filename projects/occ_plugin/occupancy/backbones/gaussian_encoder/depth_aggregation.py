from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmengine.model import xavier_init, constant_init
from mmengine import build_from_cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math


@MODELS.register_module()
class DepthWiseDeformableSampler(BaseModule):
    def __init__(
        self,
        instance_dim: int,
        feat_dim: int,
        num_groups: int,
        num_levels: int,
        num_depth: int,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        pc_range: Optional[List[float]] = None,
    ):
        super().__init__()

        assert feat_dim % num_groups == 0
        self.instance_dim = instance_dim
        self.feat_dim = feat_dim
        self.num_groups = num_groups
        self.group_dims = feat_dim // num_groups
        self.num_levels = num_levels
        self.num_depth = num_depth
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = attn_drop

        assert kps_generator is not None
        kps_generator.setdefault("embed_dims", instance_dim)
        self.kps_generator = build_from_cfg(kps_generator, MODELS)
        self.num_pts = self.kps_generator.num_pts

        self.weights_fc = nn.Linear(
            instance_dim, num_groups * 1 * num_levels * self.num_pts
        )

        assert pc_range is not None
        self.register_buffer(
            "pc_range", torch.tensor(pc_range, dtype=torch.float32)
        )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)

    @staticmethod
    def world_to_bev_points(key_points, pc_range):
        """3D (x,y,z) → BEV (x,y) normalize [0,1]."""
        x_min, y_min, z_min, x_max, y_max, z_max = pc_range.tolist()
        x = key_points[..., 0]
        y = key_points[..., 1]
        u = (x - x_min) / (x_max - x_min + 1e-6)
        v = (y - y_min) / (y_max - y_min + 1e-6)
        return torch.stack([u, v], dim=-1)  # [B, N, P, 2]

    def _get_weights_single_depth(self, instance_feature, anchor_embed):
        """
        instance_feature, anchor_embed: [B, N, instance_dim]
        weights: [B, N, 1, num_levels, num_pts, num_groups]
        """
        bs, num_anchor = instance_feature.shape[:2]
        feat = instance_feature + anchor_embed  # [B, N, instance_dim]

        weights = (
            self.weights_fc(feat)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .reshape(
                bs,
                num_anchor,
                1,               
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = (torch.rand_like(weights) > self.attn_drop)
        else:
            mask = torch.ones_like(weights, dtype=torch.bool)
        return weights, mask

    def _feature_sampling_single_depth(
        self,
        feature_maps_d: List[torch.Tensor],  # [B, 1, feat_dim, H, W]
        key_points: torch.Tensor,           # [B, N, P, 3]
    ) -> torch.Tensor:
       
        bs, num_anchor, num_pts = key_points.shape[:3]

        # 3D -> BEV 2D
        points_2d = self.world_to_bev_points(key_points, self.pc_range)  # [B,N,P,2]
        points_2d = points_2d * 2 - 1  # [-1,1]

        feats_list = []
        for fm in feature_maps_d:
            # fm: [B,1,feat_dim,H,W] -> [B,feat_dim,H,W]
            fm2d = fm[:, 0]
            sampled = F.grid_sample(
                fm2d,
                points_2d,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )  # [B,feat_dim,N,P]
            sampled = sampled.permute(0, 2, 3, 1)  # [B,N,P,feat_dim]
            sampled = sampled.unsqueeze(2)         # [B,N,1,P,feat_dim]
            feats_list.append(sampled)

        features = torch.stack(feats_list, dim=3)
        return features

    def _get_level_single_depth(
        self,
        features: torch.Tensor,  # [B,N,1,L,P,feat_dim]
        weights: torch.Tensor,   # [B,N,1,L,P,num_groups]
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=3) 
        features = features.reshape(bs, num_anchor, self.num_pts, self.feat_dim)
        return features  # [B,N,num_pts,feat_dim]

    def forward(
        self,
        instance_feature: torch.Tensor,   # [B, N, instance_dim]
        anchor: torch.Tensor,             # [B, N, ...]
        anchor_embed: torch.Tensor,       # [B, N, instance_dim]
        feature_maps: List[torch.Tensor], # [B, D, feat_dim, H, W]
        metas: dict = None,
        **kwargs,
    ):
        bs, num_anchor, _ = instance_feature.shape
        D = self.num_depth

        key_points = self.kps_generator(anchor, instance_feature)  # [B,N,num_pts,3]

        depth_feats = []

        for d in range(D):
            fm_d_levels = [fm[:, d:d+1] for fm in feature_maps]

            feats_d = self._feature_sampling_single_depth(
                fm_d_levels, key_points
            )  # [B,N,1,L,P,feat_dim]

            weights_d, _ = self._get_weights_single_depth(
                instance_feature, anchor_embed
            )  # [B,N,1,L,P,G]

            feats_d = self._get_level_single_depth(
                feats_d, weights_d
            )  # [B,N,num_pts,feat_dim]

            # multi-point fuse: [B,N,feat_dim]
            feats_d = feats_d.sum(dim=2)
            depth_feats.append(feats_d)

        # [B,N,D,feat_dim]
        depth_feats = torch.stack(depth_feats, dim=2)
        depth_feats = self.proj_drop(depth_feats)
        return depth_feats


@MODELS.register_module()
class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

    def forward(self, query, key, value):
        batch, n_views, seq_len, c = query.shape
        
        q = query.view(batch * n_views, seq_len, c)
        k = key.view(batch * n_views, seq_len, c)
        v = value.view(batch * n_views, seq_len, c)

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        head_dim = c // self.num_heads
        
        q = q.view(batch * n_views, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch * n_views, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch * n_views, seq_len, self.num_heads, head_dim).transpose(1, 2)

        attn_output = self.scaled_dot_product_attention(q, k, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch * n_views, seq_len, c)
        
        attn_output = self.out_proj(attn_output)
        
        return attn_output.view(batch, n_views, seq_len, c)
    
    
@MODELS.register_module()
class DepthInteractionAggregator(BaseModule):
    def __init__(
        self,
        feat_dim: int,
        num_depth: int,
        num_heads: int = 4,
        instance_dim: int = 128,
        proj_drop: float = 0.0,
        num_layers: int = 2,
        dropout: float = 0.1,
        splits: List[int] = [4,3,3]
    ):
        super().__init__()
        
        self.num_depth = num_depth
        self.feat_dim = feat_dim
        self.splits = splits
        
        self.chunk1_embed = nn.Linear(instance_dim, instance_dim//2)
        self.chunk2_embed = nn.Linear(instance_dim, instance_dim//2)
        
        self.fuse_gate = nn.Linear(2 * instance_dim, 1)

        self.output_proj = nn.Linear(feat_dim, instance_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weight(self):
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(self, depth_feats: torch.Tensor) -> torch.Tensor:
        """
        depth_feats: [B, N, D, feat_dim]
        """
        B, N, D, C = depth_feats.shape
        assert D == self.num_depth
        assert C == self.feat_dim
        assert sum(self.splits) == D

        # (B,N,D,C) -> (D, B*N, C)
        x = depth_feats.view(B * N, D, C).permute(1, 0, 2)
        
        perm = torch.randperm(D)
        idx_chunks = perm.split(self.splits)
        chunks = [x[idx] for idx in idx_chunks]
        chunks = [x[idx].mean(dim=0) for idx in idx_chunks]

        chunk1 = self.chunk1_embed(chunks[0])  # [B*N, instance_dim//2]
        chunk2 = self.chunk2_embed(chunks[1])  # [B*N, instance_dim//2]
        chunk = torch.cat([chunk1, chunk2], dim=-1)  # [B*N, instance_dim]

        att = chunk*chunks[2]        
        
        feat = x.mean(dim=0) 
        
        g_in = torch.cat([att, feat], dim=-1)          
        gate = torch.sigmoid(self.fuse_gate(g_in))     

        fused = gate * att + (1.0 - gate) * feat
        fused = fused.unsqueeze(0)   
                
        fused = self.proj_drop(self.output_proj(fused))  # [B,N,embed_dims]
        
        return fused  # [B,N,feat_dim]


@MODELS.register_module()
class DepthFeatureAggregation(BaseModule):
    def __init__(
        self,
        embed_dims: int,            
        num_groups: int,
        num_levels: int,
        pc_range: List[float],
        num_depth: int = 10,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        residual_mode: str = "add",
        kps_generator: dict = None,
        feat_dim: Optional[int] = None,   
        depth_num_heads: int = 4,
        depth_num_layers: int = 2,
        splits = List[int]
    ):
        super().__init__()
        self.residual_mode = residual_mode
        self.instance_dim = embed_dims
        self.feat_dim = feat_dim if feat_dim is not None else embed_dims
        self.splits = splits

        self.depth_sampler = DepthWiseDeformableSampler(
            instance_dim=self.instance_dim,
            feat_dim=self.feat_dim,
            num_groups=num_groups,
            num_levels=num_levels,
            num_depth=num_depth,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            kps_generator=kps_generator,
            pc_range=torch.tensor(pc_range, dtype=torch.float32),
        )

        self.depth_agg = DepthInteractionAggregator(
            feat_dim=self.feat_dim,
            num_depth=num_depth,
            proj_drop=proj_drop,
            num_heads=depth_num_heads,
            instance_dim=self.instance_dim,
            num_layers=depth_num_layers,
            splits = splits
        )

    def forward(
        self,
        instance_feature: torch.Tensor,   # [B, N, embed_dims]
        anchor: torch.Tensor,             # [B, N, ...]
        anchor_embed: torch.Tensor,       # [B, N, embed_dims]
        feature_maps: List[torch.Tensor], # [B, D, feat_dim, H, W]
        metas: dict = None,
        **kwargs,
    ):
        depth_feats = self.depth_sampler(
            instance_feature, anchor, anchor_embed, feature_maps, metas
        )

        out = self.depth_agg(depth_feats)
        
        if self.residual_mode == "add":
            out = out + instance_feature
        elif self.residual_mode == "cat":
            out = torch.cat([out, instance_feature], dim=-1)

        return out