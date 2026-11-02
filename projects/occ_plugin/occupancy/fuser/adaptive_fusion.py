
from torch import nn
import torch.nn.functional as F
from mmengine.registry import MODELS
import torch


class DualStreamCrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj_L = nn.Linear(dim, dim) # Projects x
        self.k_proj_C = nn.Linear(dim, dim) # Projects y
        self.v_proj_C = nn.Linear(dim, dim) # Projects y
        
        self.q_proj_C = nn.Linear(dim, dim) # Projects y
        self.k_proj_L = nn.Linear(dim, dim) # Projects x
        self.v_proj_L = nn.Linear(dim, dim) # Projects x
        
        self.fusion_gate_mlp = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.Sigmoid()
            )
                
        self.consistency_scale = nn.Sequential(
                    nn.Linear(1, dim),
                    nn.Sigmoid()
                )

    def forward(self, f_lidar, f_camera):
        """
        x: LiDAR features [B, N, D]
        y: Camera features [B, N, D]
        """
        
        Q_x = self.q_proj_L(f_lidar)
        K_y = self.k_proj_C(f_camera)
        V_y = self.v_proj_C(f_camera)
        
        attention_score_L = (Q_x * K_y).sum(dim=-1, keepdim=True) / (f_lidar.shape[-1] ** 0.5)
        attention_weight_L = torch.sigmoid(attention_score_L) 
        H_L = f_lidar + (attention_weight_L * V_y)
        
        Q_y = self.q_proj_C(f_camera)
        K_x = self.k_proj_L(f_lidar)
        V_x = self.v_proj_L(f_lidar)
        
        attention_score_C = (Q_y * K_x).sum(dim=-1, keepdim=True) / (f_lidar.shape[-1] ** 0.5)
        attention_weight_C = torch.sigmoid(attention_score_C) 
        
        H_C = f_camera + (attention_weight_C * V_x)
        
        combined_streams = torch.cat([H_L, H_C], dim=-1)
        gate = self.fusion_gate_mlp(combined_streams)
        
        H_fused = (gate * H_L) + ((1 - gate) * H_C)
        
        raw_l_aligned = torch.nn.functional.normalize(Q_x, dim=-1)
        raw_c_aligned = torch.nn.functional.normalize(Q_y, dim=-1)
        sim_score = (raw_l_aligned * raw_c_aligned).sum(dim=-1, keepdim=True)
        consistency_weight = self.consistency_scale(sim_score)
        
        return H_fused * consistency_weight
    
    
@MODELS.register_module()
class AdaptiveFusion(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        
        self.cross_att_instance = DualStreamCrossAttention(dim=dim)
        
        self.cross_att_embed = DualStreamCrossAttention(dim=dim)
        
        anchor_dim = 28
        anchor_hidden_dim = 112
        
        self.anchor_fusion_mlp = nn.Sequential(
            nn.Linear(anchor_dim * 2, anchor_hidden_dim),
            nn.LayerNorm(anchor_hidden_dim),
            nn.GELU(),
            nn.Linear(anchor_hidden_dim, anchor_dim)
        )
        
    def instance_fusion(self, instance_img, instance_pts):
        fusion = self.cross_att_instance(instance_pts, instance_img)
        return fusion
    
    def embed_fusion(self, embed_img, embed_pts):
        fusion = self.cross_att_embed(embed_pts, embed_img)
        return fusion
        
    def anchor_fusion(self, anchor_img, anchor_pts):
        combined = torch.cat([anchor_img, anchor_pts], dim=-1)
        fused_anchor = self.anchor_fusion_mlp(combined)
        return fused_anchor
        
    def forward(self, instance_img, instance_pts,
                     embed_img, embed_pts,
                     anchor_img, anchor_pts):
              
        instance_fusion = self.instance_fusion(instance_img, instance_pts)
        embed_fusion = self.embed_fusion(embed_img, embed_pts)
        anchor_fusion = self.anchor_fusion(anchor_img, anchor_pts)
                
        return instance_fusion, embed_fusion, anchor_fusion
