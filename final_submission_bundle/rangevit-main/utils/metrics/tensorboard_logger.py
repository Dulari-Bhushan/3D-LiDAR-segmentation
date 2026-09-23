'''
From Z. Zhuang et al.
https://github.com/ICEORY/PMF
'''

import torch
import tensorboardX


def _to_scalar(value, default=0.0):
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        if value.numel() > 1:
            value = value.float().mean()
        if not torch.isfinite(value):
            return default
        return float(value.item())
    if isinstance(value, (float, int)):
        return float(value)
    return default

def tensorboard_logger(epoch,
                       mode,
                       recorder,
                       metrics_dict,
                       loss_dict,
                       lr,
                       mapped_cls_name):
    
    # Metrics
    mean_acc, class_acc = metrics_dict['mean_acc'], metrics_dict['class_acc']
    mean_recall, class_recall = metrics_dict['mean_recall'], metrics_dict['class_recall']
    mean_iou, class_iou = metrics_dict['mean_iou'], metrics_dict['class_iou']
    
    # Losses
    loss_meter_avg = loss_dict['loss_meter_avg']
    loss_focal = loss_dict['loss_focal']
    loss_lovasz = loss_dict['loss_lovasz']

    recorder.tensorboard.add_scalar(
        tag='{}_Loss'.format(mode), scalar_value=_to_scalar(loss_meter_avg), global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_LossSoftmax'.format(mode), scalar_value=_to_scalar(loss_focal), global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_LossLovasz'.format(mode), scalar_value=_to_scalar(loss_lovasz), global_step=epoch)

    if 'loss_prototype' in loss_dict:
        recorder.tensorboard.add_scalar(
            tag='{}_LossPrototype'.format(mode),
            scalar_value=_to_scalar(loss_dict['loss_prototype']),
            global_step=epoch)
    if 'loss_geometry' in loss_dict:
        recorder.tensorboard.add_scalar(
            tag='{}_LossGeometry'.format(mode),
            scalar_value=_to_scalar(loss_dict['loss_geometry']),
            global_step=epoch)
    if 'loss_attention_reg' in loss_dict:
        recorder.tensorboard.add_scalar(
            tag='{}_LossAttentionReg'.format(mode),
            scalar_value=_to_scalar(loss_dict['loss_attention_reg']),
            global_step=epoch)
    
    recorder.tensorboard.add_scalar(
        tag='{}_meanAcc'.format(mode), scalar_value=_to_scalar(mean_acc), global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_meanIOU'.format(mode), scalar_value=_to_scalar(mean_iou), global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_meanRecall'.format(mode), scalar_value=_to_scalar(mean_recall), global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_lr'.format(mode), scalar_value=lr, global_step=epoch)

    if 'proto_purity' in metrics_dict:
        recorder.tensorboard.add_scalar(
            tag='{}_ProtoPurity'.format(mode),
            scalar_value=_to_scalar(metrics_dict['proto_purity']),
            global_step=epoch)
    if 'attention_entropy' in metrics_dict:
        recorder.tensorboard.add_scalar(
            tag='{}_AttentionEntropy'.format(mode),
            scalar_value=_to_scalar(metrics_dict['attention_entropy']),
            global_step=epoch)

    for i, (_, v) in enumerate(mapped_cls_name.items()):
        recorder.tensorboard.add_scalar(
            tag='{}_{:02d}_{}_Acc'.format(mode, i, v), scalar_value=_to_scalar(class_acc[i]), global_step=epoch)
        recorder.tensorboard.add_scalar(
            tag='{}_{:02d}_{}_Recall'.format(mode, i, v), scalar_value=_to_scalar(class_recall[i]),
            global_step=epoch)
        recorder.tensorboard.add_scalar(
            tag='{}_{:02d}_{}_IOU'.format(mode, i, v), scalar_value=_to_scalar(class_iou[i]), global_step=epoch)
