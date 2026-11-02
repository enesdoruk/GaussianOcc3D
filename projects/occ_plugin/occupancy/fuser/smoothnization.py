
import torch
from torch import nn
import torch.nn.functional as F
from mmengine.registry import MODELS


@MODELS.register_module()
class Smoothnization(nn.Module):
    def __init__(self, dim, eps, temperature) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.temperature = temperature


    def cross_entropy(self, cam_feats: torch.Tensor, lidar_feats: torch.Tensor):        
        ce_cam_lidar = F.softmax(cam_feats/self.temperature, dim=self.dim) * \
                                F.log_softmax(lidar_feats/self.temperature+1e-8, dim=self.dim)
        
        ce_lidar_cam = F.softmax(lidar_feats/self.temperature, dim=self.dim) * \
                                F.log_softmax(cam_feats/self.temperature+1e-8, dim=self.dim)

        return ce_cam_lidar, ce_lidar_cam
    
    
    def entropy_manipulation(
        self,
        cam_feats: torch.Tensor,       # [N, C]
        lidar_feats: torch.Tensor,     # [N, C]
        cam_dom_entr: torch.Tensor,          # [N, 1]
        lidar_dom_entr: torch.Tensor,          # [N, 1]
    ):
        
        conf_cam = torch.exp(-cam_dom_entr)
        conf_lidar = torch.exp(-lidar_dom_entr)
        
        conf_sum = conf_cam + conf_lidar + 1e-8
        w_cam = conf_cam / conf_sum
        w_lidar = conf_lidar / conf_sum

        cam_out = cam_feats + self.eps * w_cam
        lidar_out = lidar_feats + self.eps * w_lidar
        
        return cam_out, lidar_out
    
    
    def forward(self, img, pts,
                     ):
        
        B, N, C = img.shape

        img_flat = img.reshape(B * N, C)   # [B*N, C]
        pts_flat = pts.reshape(B * N, C)
                
        cam_dom_entr, lidar_dom_entr = self.cross_entropy(img_flat, pts_flat)
        
        smooth_img, smooth_lidar = self.entropy_manipulation(
                    img_flat, pts_flat, 
                    cam_dom_entr,
                    lidar_dom_entr,
                )
        
        smooth_img = smooth_img.view(B, N, C)
        smooth_lidar = smooth_lidar.view(B, N, C)
                
        return smooth_img, smooth_lidar