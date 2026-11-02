import torch.nn as nn
from mmdet.models import LOSSES

from torch.utils.tensorboard import SummaryWriter
from mmengine.utils import ManagerMixin

class WrappedTBWriter(SummaryWriter, ManagerMixin):

    def __init__(self, name, **kwargs):
        SummaryWriter.__init__(self, **kwargs)
        ManagerMixin.__init__(self, name)


if 'selfocc' in WrappedTBWriter._instance_dict:
    writer = WrappedTBWriter.get_instance('selfocc')
else:
    writer = None

@LOSSES.register_module()
class MultiLoss(nn.Module):

    def __init__(self, loss_cfgs):
        super().__init__()
        
        assert isinstance(loss_cfgs, list)
        self.num_losses = len(loss_cfgs)
        
        losses = []
        for loss_cfg in loss_cfgs:
            losses.append(LOSSES.build(loss_cfg))
        self.losses = nn.ModuleList(losses)
        self.iter_counter = 0

    def forward(self, inputs):
        
        loss_dict = {}
        tot_loss = 0.
        for loss_func in self.losses:
            loss = loss_func(inputs)
            tot_loss += loss
            loss_dict.update({
                loss_func.__class__.__name__: \
                loss.detach().item()
            })
            if writer and self.iter_counter % 10 == 0:
                writer.add_scalar(
                    f'loss/{loss_func.__class__.__name__}', 
                    loss.detach().item(), self.iter_counter)
        if writer and self.iter_counter % 10 == 0:
            writer.add_scalar(
                'loss/total', tot_loss.detach().item(), self.iter_counter)
        self.iter_counter += 1
        
        return tot_loss, loss_dict