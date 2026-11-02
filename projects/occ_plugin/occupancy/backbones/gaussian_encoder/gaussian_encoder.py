from typing import List, Optional
from xml.etree.ElementPath import ops
import torch, torch.nn as nn

from mmengine.registry import MODELS
from mmengine import build_from_cfg
from mmengine.model import BaseModule

import random

import importlib
importlib.import_module('projects.occ_plugin.occupancy.fuser.adaptive_fusion')
importlib.import_module('projects.occ_plugin.occupancy.fuser.smoothnization')


@MODELS.register_module()
class BaseEncoder(BaseModule):
    """Further encode 3D representations.
    image backbone -> neck -> lifter -> encoder -> segmentor
    """

    def __init__(self, init_cfg=None, **kwargs):
        super().__init__(init_cfg)
    
    def forward(
        self, 
        representation,
        ms_img_feats=None,
        metas=None,
        **kwargs
    ):
        pass
    
@MODELS.register_module()
class GaussianOccEncoder(BaseEncoder):
    def __init__(
        self,
        anchor_encoder: dict,
        norm_layer: dict,
        ffn: dict,
        refine_layer: dict,
        deformable_model_img: dict = None,
        deformable_model_depth: dict = None,
        mid_refine_layer: dict = None,
        spconv_layer: dict = None,
        fusion_layer: dict = None,
        smoothnization: dict = None,
        num_decoder: int = 6,
        version: str = 'RGB',
        operation_order: Optional[List[str]] = None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.num_decoder = num_decoder
        self.version = version
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.anchor_encoder = build(anchor_encoder, MODELS)
        
        if self.version == 'RGB':
            self.op_config_map = {
                "norm": [norm_layer, MODELS],
                "ffn": [ffn, MODELS],
                "deformable_img": [deformable_model_img, MODELS],
                "refine": [refine_layer, MODELS],
                "mid_refine":[mid_refine_layer, MODELS],
                "spconv": [spconv_layer, MODELS],
            }
        elif self.version == 'LiDAR':
            self.op_config_map = {
                "norm": [norm_layer, MODELS],
                "ffn": [ffn, MODELS],
                "deformable_depth": [deformable_model_depth, MODELS],
                "refine": [refine_layer, MODELS],
                "mid_refine":[mid_refine_layer, MODELS],
                "spconv": [spconv_layer, MODELS],
            }
        elif self.version == 'Multimodal':
            self.op_config_map = {
                "norm": [norm_layer, MODELS],
                "ffn": [ffn, MODELS],
                "deformable_img": [deformable_model_img, MODELS],
                "deformable_depth": [deformable_model_depth, MODELS],
                "refine": [refine_layer, MODELS],
                "mid_refine":[mid_refine_layer, MODELS],
                "spconv": [spconv_layer, MODELS],
                "fusion": [fusion_layer, MODELS],
            }
            
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )    
        
        if self.version == 'Multimodal':
            fusion_idx = self.operation_order.index('fusion')          
            before_fusion = self.operation_order[:fusion_idx]          
            self.selected_smoothnization = random.choice(before_fusion) 
        else:
            self.selected_smoothnization = None
        
        if self.version == 'Multimodal':  
            self.smoothnization = MODELS.build(smoothnization) if smoothnization is not None else None
   
    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(
        self,
        representation=None,
        rep_features=None,
        ms_img_feats=None,
        metas=None,
        pts_representation=None,
        pts_rep_features=None,
        ms_pts_feats=None,
        **kwargs
    ):
        self.fusion_flag = False
        
        if self.version == 'RGB' or self.version == 'Multimodal':
            feature_maps_img = ms_img_feats
            if isinstance(feature_maps_img, torch.Tensor):
                feature_maps_img = [feature_maps_img]
            instance_feature_img = rep_features
            anchor_img = representation

        if self.version == 'LiDAR' or self.version == 'Multimodal':
            feature_maps_pts = ms_pts_feats
            if isinstance(feature_maps_pts, torch.Tensor):
                feature_maps_pts = [feature_maps_pts]
            instance_feature_pts = pts_rep_features
            anchor_pts = pts_representation
        
        if self.version == 'Multimodal' or self.version == "RGB":
            anchor_embed_img = self.anchor_encoder(anchor_img)
        
        if self.version == 'LiDAR' or self.version == 'Multimodal':
            anchor_embed_pts = self.anchor_encoder(anchor_pts)
                        
        prediction = []
        for i, op in enumerate(self.operation_order):
            if op == "spconv":
                if op == self.selected_smoothnization and self.smoothnization is not None and self.version == 'Multimodal' and self.fusion_flag==False:
                    instance_feature_img, instance_feature_pts = self.smoothnization(instance_feature_img, instance_feature_pts)           
        
                if self.fusion_flag:  
                    instance_feature_fusion = self.layers[i](
                        instance_feature_fusion,
                        anchor_fusion  )
                else:
                    if self.version == 'RGB' or self.version == 'Multimodal':
                        instance_feature_img = self.layers[i](
                            instance_feature_img,
                            anchor_img  )
                    if self.version == 'Multimodal' or self.version == 'LiDAR':
                        instance_feature_pts = self.layers[i](
                            instance_feature_pts,
                            anchor_pts  )  
                                        
            elif op == "norm" or op == "ffn": 
                if op == self.selected_smoothnization and self.smoothnization is not None and self.version == 'Multimodal' and self.fusion_flag==False:
                    instance_feature_img, instance_feature_pts = self.smoothnization(instance_feature_img, instance_feature_pts)           
                    
                if self.fusion_flag: 
                    instance_feature_fusion = self.layers[i](instance_feature_fusion)
                else:
                    if self.version == 'RGB' or self.version == 'Multimodal':
                        instance_feature_img = self.layers[i](instance_feature_img)
                    if self.version == 'Multimodal' or self.version == 'LiDAR':
                        instance_feature_pts = self.layers[i](instance_feature_pts)
                
            elif op == "identity":
                if op == self.selected_smoothnization and self.smoothnization is not None and self.version == 'Multimodal' and self.fusion_flag==False:
                    instance_feature_img, instance_feature_pts = self.smoothnization(instance_feature_img, instance_feature_pts)           
      
                if self.fusion_flag:  
                    identity_fusion = instance_feature_fusion
                else:
                    if self.version == 'RGB' or self.version == 'Multimodal':
                        identity_img = instance_feature_img
                    if self.version == 'Multimodal' or self.version == 'LiDAR':
                        identity_pts = instance_feature_pts
                    
            elif op == "add":
                if op == self.selected_smoothnization and self.smoothnization is not None and self.version == 'Multimodal' and self.fusion_flag==False:
                    instance_feature_img, instance_feature_pts = self.smoothnization(instance_feature_img, instance_feature_pts)           
                    
                if self.fusion_flag:
                    instance_feature_fusion = instance_feature_fusion + identity_fusion
                else:
                    if self.version == 'RGB' or self.version == 'Multimodal':
                        instance_feature_img = instance_feature_img + identity_img
                    if self.version == 'Multimodal' or self.version == 'LiDAR':
                        instance_feature_pts = instance_feature_pts + identity_pts
                
            elif op == "deformable_img":
                if op == self.selected_smoothnization and self.smoothnization is not None and self.version == 'Multimodal' and self.fusion_flag==False:
                    instance_feature_img, instance_feature_pts = self.smoothnization(instance_feature_img, instance_feature_pts)           
                    
                if self.fusion_flag:
                    continue
                else:
                    instance_feature_img = self.layers[i](
                        instance_feature_img,
                        anchor_img,
                        anchor_embed_img,
                        feature_maps_img,
                    metas,
                    )
                   
            elif op == "deformable_depth":
                if op == self.selected_smoothnization and self.smoothnization is not None and self.version == 'Multimodal' and self.fusion_flag==False:
                    instance_feature_img, instance_feature_pts = self.smoothnization(instance_feature_img, instance_feature_pts)           
                    
                if self.fusion_flag:
                    continue
                else:
                    instance_feature_pts = self.layers[i](
                        instance_feature_pts,
                        anchor_pts,
                        anchor_embed_pts,
                        feature_maps_pts,
                        metas,
                    )                             
                
            elif op == "fusion":
                instance_feature_fusion, anchor_embed_fusion, anchor_fusion = self.layers[i](
                    instance_feature_img,
                    instance_feature_pts,
                    anchor_embed_img, 
                    anchor_embed_pts,
                    anchor_img, 
                    anchor_pts
                )
                self.fusion_flag = True                
                
            elif "refine" in op:
                if self.fusion_flag:  
                    anchor_fusion, gaussian_fusion = self.layers[i](
                    instance_feature_fusion,
                    anchor_fusion,
                    anchor_embed_fusion
                    )
                    
                    prediction.append({'gaussian': gaussian_fusion})
                    if i != len(self.operation_order) - 1:
                        anchor_embed = self.anchor_encoder(anchor_fusion)
                else:
                    if self.version == 'RGB' or self.version == 'Multimodal':
                        anchor_img, gaussian_img = self.layers[i](
                        instance_feature_img,
                        anchor_img,
                        anchor_embed_img,
                        )
                        prediction.append({'gaussian': gaussian_img})
                        if i != len(self.operation_order) - 1:
                            anchor_embed = self.anchor_encoder(anchor_img) 
                
                    if self.version == "Multimodal" or self.version == 'LiDAR':
                        anchor_pts, gaussian_pts = self.layers[i](
                        instance_feature_pts,
                        anchor_pts,
                        anchor_embed_pts,
                        )
                        prediction.append({'gaussian': gaussian_pts})
                        if i != len(self.operation_order) - 1:
                            anchor_embed = self.anchor_encoder(anchor_pts)
         
            else:
                raise NotImplementedError(f"{op} is not supported.")
            
        return {"representation": prediction}