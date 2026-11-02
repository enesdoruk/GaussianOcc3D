# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmdet.utils import get_root_logger
from mmdet.core import encode_mask_results


import mmcv
import numpy as np
import pycocotools.mask as mask_util
import mcubes
#import open3d as o3d
import pdb
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from projects.occ_plugin.utils.formating import cm_to_ious, format_SC_results, format_SSC_results
from projects.occ_plugin.core import save_occ

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]

def custom_single_gpu_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.3):
    model.eval()
    
    SC_metric = 0
    SSC_metric = 0
    IOU_metric = 0
    occ_IOU_metric = 0
    SSC_occ_metric = 0
    mIOU_metric = 0
    
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    logger = get_root_logger()
    
    logger.info(parameter_count_table(model))
        
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

            if show:
                pred_c = result['pred_c'] if 'pred_c' in result.keys() else None
                pred_f = result['pred_f'] if 'pred_f' in result.keys() else None
                gt_occ = data['gt_occ'] if 'gt_occ' in result.keys() else None
                points = data['points'] if 'points' in data.keys() else None
                visual_mask = data['visible_mask'] if 'visible_mask' in data.keys() else None
                img_metas = data['img_metas'] if 'img_metas' in data.keys() else None
                save_occ(pred_c, pred_f, points, img_metas, out_dir, visual_mask,gt_occ)
            
            # only support semantic voxel segmentation
            if 'SC' in result.keys():
                SC_metric += result['SC']
            if 'SSC' in result.keys():
                SSC_metric += result['SSC']
            if 'IOU' in result.keys():
                IOU_metric += result['IOU']
            if 'mIOU' in result.keys():
                mIOU_metric += result['mIOU']
            if 'occ_IOU' in result.keys():
                occ_IOU_metric += result['occ_IOU']
            if 'SSC_occ' in result.keys():
                SSC_occ_metric += result['SSC_occ']
            batch_size = 1

        
        if 'SC' in result.keys():
            print(format_SC_results(result['mIOU']))
        if 'SSC' in result.keys():
            print(format_SSC_results(result['IOU']))
        
        prog_bar.update()


    res = {
        'SC': SC_metric,
        'SSC': SSC_metric,
        'IOU': IOU_metric,
        'occ_IOU': occ_IOU_metric,
        'SSC_occ': SSC_occ_metric,
        'mIOU': mIOU_metric
    }

    return res

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, show=False, out_dir=None):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """

    model.eval()
    
    # init predictions
    SC_metric = []
    SSC_metric = []
    IOU_metric = []
    occ_IOU_metric = []
    SSC_occ_metric = []
    mIOU_metric = []


    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    logger = get_root_logger()
    logger.info(parameter_count_table(model))
    

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            
            if show:
                pred_c = result['pred_c'] if 'pred_c' in result.keys() else None
                pred_f = result['pred_f'] if 'pred_f' in result.keys() else None
                gt_occ = data['gt_occ'] if 'gt_occ' in data.keys() else None
                points = data['points'] if 'points' in data.keys() else None
                visual_mask = data['visible_mask'] if 'visible_mask' in data.keys() else None
                img_metas = data['img_metas'] if 'img_metas' in data.keys() else None
                save_occ(pred_c, pred_f, points, img_metas, out_dir, visual_mask,gt_occ)
        
            if 'SC' in result.keys():
                SC_metric.append(result['SC'])
            if 'SSC' in result.keys():
                SSC_metric.append(result['SSC'])
            if 'IOU' in result.keys():
                IOU_metric.append(np.array(result['IOU']))
            if 'mIOU' in result.keys():
                mIOU_metric.append(np.array(result['mIOU']))
            if 'occ_IOU' in result.keys():
                occ_IOU_metric.append(np.array(result['occ_IOU'].item()))
            if 'SSC_occ' in result.keys():
                SSC_occ_metric.append(result['SSC_occ'])
            batch_size = 1

                
        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect lists from multi-GPUs
    res = {}
    
    if 'SC' in result.keys():
        SC_metric = [sum(SC_metric)]
        SC_metric = collect_results_cpu(SC_metric, len(dataset), tmpdir)
        res['SC'] = SC_metric

    if 'SSC' in result.keys():
        SSC_metric = [sum(SSC_metric)]
        SSC_metric = collect_results_cpu(SSC_metric, len(dataset), tmpdir)
        res['SSC'] = SSC_metric
        
    if 'IOU' in result.keys():
        IOU_metric = [sum(IOU_metric)]
        IOU_metric = collect_results_cpu(IOU_metric, len(dataset), tmpdir)
        res['IOU'] = IOU_metric
    
    if 'occ_IOU' in result.keys():
        occ_IOU_metric = [sum(occ_IOU_metric)]
        occ_IOU_metric = collect_results_cpu(occ_IOU_metric, len(dataset), tmpdir)
        res['occ_IOU'] = occ_IOU_metric

    if 'SSC_occ' in result.keys():
        SSC_occ_metric = [sum(SSC_occ_metric)]
        SSC_occ_metric = collect_results_cpu(SSC_occ_metric, len(dataset), tmpdir)
        res['SSC_occ'] = SSC_occ_metric

    if 'mIOU' in result.keys():
        mIOU_metric = [sum(mIOU_metric)]
        mIOU_metric = collect_results_cpu(mIOU_metric, len(dataset), tmpdir)
        res['mIOU'] = mIOU_metric
    
    return res


def collect_results_cpu(result_part, size, tmpdir=None, type='list'):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()

    # collect all parts
    if rank == 0:
    
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))

        # sort the results
        if type == 'list':
            ordered_results = []
            for res in part_list:  
                ordered_results.extend(list(res))
            # the dataloader may pad some samples
            ordered_results = ordered_results[:size]
        
        else:
            raise NotImplementedError
        
        # remove tmp dir
        shutil.rmtree(tmpdir)
    
    dist.barrier()

    if rank != 0:
        return None
    
    return ordered_results

