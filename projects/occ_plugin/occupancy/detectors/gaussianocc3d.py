import torch
import collections 
import torch.nn.functional as F

from mmdet.models import DETECTORS
from mmcv.runner import auto_fp16, force_fp32
from .bevdepth import BEVDepth
from mmdet3d.models import builder
from mmengine.registry import MODELS

from tools.vis import save_occ, save_gaussian, save_gaussian_topdown
from PIL import Image
import os

import numpy as np
import time
import copy

from projects.occ_plugin.utils import SPConvVoxelization

import importlib
importlib.import_module('projects.occ_plugin.occupancy.lifter.gaussian_lifter')
importlib.import_module('projects.occ_plugin.occupancy.lifter.gaussian_lifter_v2')
importlib.import_module('projects.occ_plugin.occupancy.backbones.gaussian_encoder.anchor_encoder_module')
importlib.import_module('projects.occ_plugin.occupancy.backbones.gaussian_encoder.deformable_module')


@DETECTORS.register_module()
class GaussianOcc3D(BEVDepth):
    def __init__(self, 
            loss_cfg=None,
            version='RGB',
            disable_loss_depth=False,
            empty_idx=0,
            occ_encoder_backbone=None,
            occ_encoder_neck=None,
            gauss_lifter_img=None,
            gauss_lifter_pts=None,
            gauss_encoder=None,
            head=None,
            dataset='nuscenes',
            loss_norm=False,
            **kwargs):     
        
        pts_voxel_cfg = kwargs.get('pts_voxel_layer', None)
        kwargs['pts_voxel_layer'] = None
        super().__init__(**kwargs)
                
        self.loss_cfg = loss_cfg
        self.disable_loss_depth = disable_loss_depth
        self.dataset = dataset
        self.loss_norm = loss_norm
        
        self.version = version            
        
        self.record_time = False
        self.time_stats = collections.defaultdict(list)
        self.empty_idx = empty_idx
        
        if self.version == 'Multimodal' or self.version == 'LiDAR':
            self.pts_voxel_layer = SPConvVoxelization(**pts_voxel_cfg) if pts_voxel_cfg is not None else None
        
        self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone) if occ_encoder_backbone is not None else None
        self.occ_encoder_neck = builder.build_neck(occ_encoder_neck) if occ_encoder_neck is not None else None

        self.gauss_lifter_img = MODELS.build(gauss_lifter_img) if gauss_lifter_img is not None else None
        
        self.gauss_lifter_pts = MODELS.build(gauss_lifter_pts) if gauss_lifter_pts is not None else None

        self.gauss_encoder = MODELS.build(gauss_encoder) if gauss_encoder is not None else None
        
        self.head = MODELS.build(head) if head is not None else None
        

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        
        backbone_feats = self.img_backbone(imgs)

        if self.with_img_neck:
            x = self.img_neck(backbone_feats)
    
        x = list(x)
        for i in range(len(x)):            
            _, output_dim, ouput_H, output_W = x[i].shape
            x[i] = x[i].view(B, N, output_dim, ouput_H, output_W)
        x = tuple(x)
        
        return {'img_feats': x}
    
    @force_fp32()
    def occ_encoder(self, x):
        x = self.occ_encoder_backbone(x)
        x = self.occ_encoder_neck(x)
        return x
    
    def extract_img_feat(self, img, img_metas):        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        
        img_enc_feats = self.image_encoder(img[0])
        img_feats = img_enc_feats['img_feats']
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['img_encoder'].append(t1 - t0)
        
        return img_feats

    def extract_pts_feat(self, pts):
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()

        import pdb; pdb.set_trace()
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)

        xyz = coors[:, [3, 2, 1]].unsqueeze(0).float().contiguous()
        batch_size = coors[-1, 0] + 1
        
        pts_enc_feats = self.pts_middle_encoder(voxel_features, coors, pts=xyz, batch_size=batch_size)
        pts_enc_feats['pts_feats'] = pts_enc_feats['pts_feats'][0].permute(0,2,1,3,4)

        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['pts_encoder'].append(t1 - t0)
                    
        return pts_enc_feats['pts_feats']

    def extract_feat(self, points=None, img=None, img_metas=None):
        pts_feats = None
        img_feats =  None
        if self.version == 'Multimodal' or self.version == 'RGB':
            img_feats = self.extract_img_feat(img, img_metas)
            
        if self.version == 'Multimodal' or self.version == 'LiDAR':
            pts_feats = self.extract_pts_feat(points)
        
        if points is not None:
            device = points[0].device
        else:
            device = img[0].device
            
        cam_positions= torch.stack([torch.tensor(t['cam_positions']).to(device) for t in img_metas], dim=0)
        focal_positions= torch.stack([torch.tensor(t['focal_positions']).to(device) for t in img_metas], dim=0)
        projection_mat= torch.stack([torch.tensor(t['projection_mat']).to(device) for t in img_metas], dim=0)
        occ_cam_mask= torch.stack([torch.tensor(t['occ_cam_mask']).to(device) for t in img_metas], dim=0)
        occ_label= torch.stack([torch.tensor(t['occ_label']).to(device) for t in img_metas], dim=0)
        occ_xyz= torch.stack([torch.tensor(t['occ_xyz']).to(device) for t in img_metas], dim=0)

        if self.version == 'RGB' or self.version == 'Multimodal':
            img_shapes = [tuple(img_feats[0].shape[-2:]) for _ in range(img[0].shape[1])]
            image_wh = torch.tensor(np.ascontiguousarray(np.array(img_shapes, dtype=np.float32)[:, :2][:, ::-1])).to(device).unsqueeze(0)
                            
        if self.version == 'LiDAR':
            img_shapes = [(108, 200), (108, 200), (108, 200), (108, 200), (108, 200), (108, 200)]
            image_wh = torch.tensor([[[200., 108.],
                            [200., 108.],
                            [200., 108.],
                            [200., 108.],
                            [200., 108.],
                            [200., 108.]]]).to(device)
            
        
        gauss_metas = dict(cam_positions= cam_positions,
                          focal_positions= focal_positions,
                          projection_mat= projection_mat,
                          occ_cam_mask= occ_cam_mask,
                          occ_label= occ_label,
                          occ_xyz= occ_xyz,
                          image_wh=image_wh)


        if self.version == 'RGB' or self.version == 'Multimodal':
            lifter_img = self.gauss_lifter_img(img_feats)
        
        if self.version == 'Multimodal' or self.version == 'LiDAR':
            lifter_pts = self.gauss_lifter_pts(pts_feats)


        if self.version == 'Multimodal':
            gauss_feats = self.gauss_encoder(representation=lifter_img['representation'], 
                                        rep_features=lifter_img['rep_features'], 
                                        ms_img_feats=img_feats, 
                                        metas=gauss_metas,
                                        
                                        pts_representation=lifter_pts['representation'],
                                        pts_rep_features=lifter_pts['rep_features'],
                                        ms_pts_feats=pts_feats,
                                        )
            
            return gauss_metas, gauss_feats, img_feats, pts_feats, lifter_img, lifter_pts
        
        elif self.version == 'LiDAR':
            gauss_feats = self.gauss_encoder(pts_representation=lifter_pts['representation'],
                                        pts_rep_features=lifter_pts['rep_features'],
                                        ms_pts_feats=pts_feats,
                                        metas=gauss_metas,
                                        )
            return gauss_metas, gauss_feats, pts_feats, lifter_pts
        
        elif self.version == 'RGB':
            gauss_feats = self.gauss_encoder(representation=lifter_img['representation'], 
                                        rep_features=lifter_img['rep_features'], 
                                        ms_img_feats=img_feats, 
                                        metas=gauss_metas,
                                        )
            
            return gauss_metas, gauss_feats, img_feats, lifter_img
        
        else:
            raise NotImplementedError(f"{self.version} is not supported.")
            
    
    def forward_train(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            points_occ=None,
            visible_mask=None,
            **kwargs,
        ):
        
    
        if self.version == 'LiDAR':
            gauss_metas, gauss_feats, pts_feats, lifter_pts = self.extract_feat(points=points, img_metas=img_metas)
        elif self.version == 'RGB':
            gauss_metas, gauss_feats, img_feaats, lifter_img = self.extract_feat(img=img_inputs, img_metas=img_metas) 
        elif self.version == 'Multimodal':       
            gauss_metas, gauss_feats, img_feaats, pts_feats, lifter_img, lifter_pts = self.extract_feat(points=points, img=img_inputs, img_metas=img_metas)
        else:
            raise NotImplementedError(f"{self.version} is not supported.")
        
        output = self.head(gauss_feats['representation'], gauss_metas)
                
        loss = self.head.loss_gauss(output)['loss']
        losses = dict(loss = loss)
        
        if self.loss_norm:
            for loss_key in losses.keys():
                if loss_key.startswith('loss'):
                    losses[loss_key] = losses[loss_key] / (losses[loss_key].detach() + 1e-9)

        def logging_latencies():
            avg_time = {key: sum(val) / len(val) for key, val in self.time_stats.items()}
            sum_time = sum(list(avg_time.values()))
            out_res = ''
            for key, val in avg_time.items():
                out_res += '{}: {:.4f}, {:.1f}, '.format(key, val, val / sum_time)
        
        if self.record_time:
            logging_latencies()
             
        return losses
        
    def forward_test(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            visible_mask=None,
            **kwargs,
        ):
        
        return self.simple_test(img_metas, img_inputs, points, gt_occ=gt_occ, visible_mask=visible_mask, **kwargs)
    
    def simple_test(self, img_metas, img=None, points=None, rescale=False, points_occ=None, 
            gt_occ=None, visible_mask=None):
        
        start_time = time.time()

        if self.version == 'LiDAR':
            gauss_metas, gauss_feats, pts_feats, lifter = self.extract_feat(points=points, img_metas=img_metas)
        elif self.version == 'RGB':
            gauss_metas, gauss_feats, img_feaats, lifter = self.extract_feat(img=img, img_metas=img_metas) 
        elif self.version == 'Multimodal':       
            gauss_metas, gauss_feats, img_feaats, pts_feats, lifter, lifter_pts = self.extract_feat(points=points, img=img, img_metas=img_metas)
        else:
            raise NotImplementedError(f"{self.version} is not supported.")

        output = self.head(gauss_feats['representation'], gauss_metas)

        end_time = time.time()
        
        result_dict = dict()
        result_dict.update(gauss_metas)
        result_dict.update(gauss_feats)  
        result_dict.update(rep_features=lifter['rep_features'])
        result_dict.update(anchor_init=lifter['anchor_init'])  
        result_dict.update(output)
          
        ious,occ_iou, miou, sc_metric, ssc_metric, ssc_occ_metric = self.evaluation_semantic(result_dict)
       
        # self.visualize(
        #           'base', 
        #           img, 
        #           output, 
        #           lifter)
        
        test_output = {
            'IOU': ious,
            'occ_IOU': occ_iou,
            'mIOU': miou,
            'SC': sc_metric,
            'SSC': ssc_metric,
            'SSC_occ': ssc_occ_metric,
            'time_use': end_time - start_time,
        }

        return test_output


    def evaluation_semantic(self, result_dict):
        class_indices = list(range(1, 17))
        num_classes = len(class_indices)
        empty_label = 17
            
        # compute iou, precision, recall
        for idx, pred in enumerate(result_dict['final_occ']):
            output = pred
            target = result_dict['sampled_label'][idx]
            mask = result_dict['occ_mask'][idx].flatten()
            
            if mask is not None:
                output = output[mask]
                target = target[mask]

            total_seen = torch.zeros(num_classes+1)
            total_correct = torch.zeros(num_classes+1)
            total_positive = torch.zeros(num_classes+1)
            
            for i, c in enumerate(class_indices):
                total_seen[i] += torch.sum(target == c).item()
                total_correct[i] += torch.sum((target == c)
                                                & (output == c)).item()
                total_positive[i] += torch.sum(output == c).item()
            
            total_seen[-1] += torch.sum(target != empty_label).item()
            total_correct[-1] += torch.sum((target != empty_label)
                                                & (output != empty_label)).item()
            total_positive[-1] += torch.sum(output != empty_label).item()
        
        ious = []
        precs = []
        recas = [] 
        for i in range(num_classes):
            if total_positive[i] == 0:
                precs.append(0.)
            else:
                cur_prec = total_correct[i] / total_positive[i]
                precs.append(cur_prec.item())
            if total_seen[i] == 0:
                ious.append(1)
                recas.append(1)
            else:
                cur_iou = total_correct[i] / (total_seen[i]
                                                   + total_positive[i]
                                                   - total_correct[i])
                cur_reca = total_correct[i] / total_seen[i]
                ious.append(cur_iou.item())
                recas.append(cur_reca)
        miou = np.mean(ious)
                
        occ_iou = total_correct[-1] / (total_seen[-1]
                                            + total_positive[-1]
                                            - total_correct[-1])
                
        #SC metric
        pred = result_dict['final_occ'][0]
        gt = result_dict['sampled_label'][0]
        mask = result_dict['occ_mask'][0].flatten()
        valid_pred = pred[mask]
        valid_gt = gt[mask]
        pred_binary = torch.zeros_like(valid_pred)
        gt_binary = torch.zeros_like(valid_gt)
        pred_binary[valid_pred != empty_label] = 1
        gt_binary[valid_gt != empty_label] = 1
        
        gt_binary = gt_binary.cpu().numpy().astype(np.int)
        pred_binary = pred_binary.cpu().numpy()
        noise_mask = gt_binary != 255
        
        sc_metric = fast_hist(pred_binary[noise_mask], gt_binary[noise_mask], max_label=2)
        
        #SSC metric
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy().astype(np.int)
        if mask is not None:
            mask = mask.cpu().numpy()
            noise_mask = gt != 255
            mask = noise_mask & (mask!=0)
            ssc_occ_metric = fast_hist(pred[mask], gt[mask], max_label=num_classes+1)

        ssc_metric = fast_hist(pred[noise_mask], gt[noise_mask], max_label=num_classes+1)
        
        return ious, occ_iou, miou, sc_metric, ssc_metric, ssc_occ_metric
        
    
    def forward_dummy(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            points_occ=None,
            **kwargs,
        ):

        if self.version == 'LiDAR':
            gauss_metas, gauss_feats, pts_feats, lifter_pts = self.extract_feat(points=points, img_metas=img_metas)
        elif self.version == 'RGB':
            gauss_metas, gauss_feats, img_feaats, lifter_img = self.extract_feat(img=img_inputs, img_metas=img_metas) 
        elif self.version == 'Multimodal':       
            gauss_metas, gauss_feats, img_feaats, pts_feats, lifter_img, lifter_pts = self.extract_feat(points=points, img=img_inputs, img_metas=img_metas)
        else:
            raise NotImplementedError(f"{self.version} is not supported.")

        output = self.head(gauss_feats['representation'], gauss_metas)

        return output
    
    def visualize(self,
                  model_type, 
                  img_inputs, 
                  output, 
                  lifter,
                  **kwargs,
                  ):
        
        if model_type == "base":
            draw_gaussian_params = dict(
                scalar = 1.5,
                ignore_opa = False,
                filter_zsize = False
            )
        elif model_type == "prob":
            draw_gaussian_params = dict(
                scalar = 2.0,
                ignore_opa = True,
                filter_zsize = True
            )
            
        i_iter_val=4
        dataset='nusc'
        save_dir="/home/edoruk/GMOcc/vis_output/"
        for i in range(img_inputs[0].shape[1]):
            input_img = img_inputs[0][0][i].cpu().numpy()
            input_img = np.transpose(input_img, (1,2,0))
            input_img = Image.fromarray(input_img.astype(np.uint8))
            input_img.save(os.path.join(save_dir, f'{i_iter_val}_image_{i}.png'))
                

        for idx, pred in enumerate(output['final_occ']):
            pred_occ = pred
            gt_occ = output['sampled_label'][idx]
            occ_shape = [200, 200, 16]
            
            save_gaussian_topdown(
                save_dir,
                lifter['anchor_init'],
                output['gaussians'],
                f'val_{i_iter_val}_topdown'
            )            

            save_occ(
                save_dir,
                pred_occ.reshape(1, *occ_shape),
                f'val_{i_iter_val}_pred',
                True, 0, dataset=dataset)

            
            save_occ(
                save_dir,
                gt_occ.reshape(1, *occ_shape),
                f'val_{i_iter_val}_gt',
                True, 0, dataset=dataset)
            
            save_gaussian(
                save_dir,
                output['gaussian'],
                f'val_{i_iter_val}_gaussian',
                **draw_gaussian_params)

    
def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2)
    return bin_count[:max_label ** 2].reshape(max_label, max_label)
