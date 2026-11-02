import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import os
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from projects.occ_plugin.utils.formating import cm_to_ious, format_SC_results, format_SSC_results
import torch

@DATASETS.register_module()
class NuscOCCDataset(NuScenesDataset):
    def __init__(self, occ_size, pc_range, occ_root, **kwargs):
        super().__init__(**kwargs)
        # self.data_infos = list(sorted(self.data_infos, key=lambda e: e['timestamp']))
        # self.data_infos = self.data_infos[::self.load_interval]
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.occ_root = occ_root
        self._set_group_flag()

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
            
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            
            return data

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None

        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)

        if input_dict is None:
            return None

        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def get_data_info(self, index):
        info = self.data_infos[index]
        
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            # frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
            lidar_token=info['lidar_token'],
            lidarseg=info['lidarseg'],
            curr=info,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            lidar2cam_dic = {}
            
            cam_positions = []       
            focal_positions = []   
            img_shapes = []   
            
            f = 0.0055
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)
                
                lidar2cam_dic[cam_type] = lidar2cam_rt.T
                
                R_c2l = np.asarray(cam_info['sensor2lidar_rotation'], dtype=float)     # camera->lidar
                t_c2l = np.asarray(cam_info['sensor2lidar_translation'], dtype=float)  # camera->lidar
                cam2lidar = np.eye(4, dtype=np.float32)
                cam2lidar[:3, :3] = R_c2l
                cam2lidar[:3, 3] = t_c2l
                C_h = np.array([0.0, 0.0, 0.0, 1.0]).reshape(4, 1)
                F_h = np.array([0.0, 0.0, f, 1.0]).reshape(4, 1)
                C_lidar = cam2lidar @ C_h
                F_lidar = cam2lidar @ F_h
                cam_positions.append(C_lidar.flatten()[:3])
                focal_positions.append(F_lidar.flatten()[:3])  
                
                if 'img_shape' in cam_info and cam_info['img_shape'] is not None:
                    H, W = int(cam_info['img_shape'][0]), int(cam_info['img_shape'][1])
                elif 'height' in cam_info and 'width' in cam_info:
                    H, W = int(cam_info['height']), int(cam_info['width'])
                else:
                    H, W = 256, 704
                img_shapes.append((H, W, 3))              

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                    lidar2cam_dic=lidar2cam_dic,
                    cam_positions=np.asarray(cam_positions),         
                    focal_positions=np.asarray(focal_positions),
                    img_shape=np.asarray(img_shapes, dtype=np.int32),
                    projection_mat=np.float32(np.stack(lidar2img_rts)),
                ))
            
            
        if self.modality['use_lidar']:
            # FIXME alter lidar path
            input_dict['pts_filename'] = input_dict['pts_filename'].replace('./data/nuscenes/', self.data_root)
            for sw in input_dict['sweeps']:
                sw['data_path'] = sw['data_path'].replace('./data/nuscenes/', self.data_root)
        
        return input_dict


    def evaluate(self, results, logger=None, **kawrgs):
        eval_results = {}
        
        ''' evaluate mIOU '''
        res_table, res_dic = format_SC_results([results['mIOU']], return_dic=True)
        for key, val in res_dic.items():
            eval_results['mIOU_{}'.format(key)] = val
        if logger is not None:
            logger.info('mIOU Evaluation')
            logger.info(res_table)
        
        ''' evaluate IOU '''
        res_table, res_dic = format_SSC_results(results['IOU'], return_dic=True)
        for key, val in res_dic.items():
            eval_results['IOU_{}'.format(key)] = val
        if logger is not None:
            logger.info('IOU Evaluation')
            logger.info(res_table)
        
        return eval_results
