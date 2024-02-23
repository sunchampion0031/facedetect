#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import sys
import time
from typing import Iterable
import random
import torch
from Util.create_labels import gt_creator
from Util.visualize import vis_data, vis_targets, vis_prediction
import Util.misc as misc
import Util.lr_sched as lr_sched
from Util.postprocess import postprocess
from Util.get_mAP import get_mAP
import numpy as np

# In[4]:




# In[ ]:


iou_thresh = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


# In[5]:


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,  max_norm: float = 0,
                    log_writer=None, args=None, criterion = None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch + 1, args.epochs)
    print_freq = args.accum_iter * 10 * 4

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('\nlog_dir: {}'.format(log_writer.log_dir))
    start = time.time()
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        # multi-scale 多尺度训练，改造samples，targets
        if data_iter_step % args.multi_scale_change == 0 and data_iter_step > 0 and args.multi_scale:
            size_range = args.multi_scale_range
            train_size = random.randint(size_range[0], size_range[1]) * 32
            model.set_grid(train_size)
        if args.multi_scale and model.img_size != args.input_size:
            samples = torch.nn.functional.interpolate(input=samples, size=model.img_size, mode='bilinear', align_corners=False)
            
        targets = [label.tolist() for label in targets]
        
        # 检查一下ground truth，带bbox和label
        if args.vis_data:
            vis_data(samples, targets)

        
        # 创建根据targets 创建真正的labels
        targets = gt_creator(model.img_size, model.stride, targets, args.anchor_size, args.multi_anchor, args.center_sample)

        # 检查创建的labels是否正确，画出bbox、label以及匹配的anchor
        if args.vis_targets:
            vis_targets(samples, targets, args.anchor_size, model.stride)

        targets = targets.to(device, non_blocking=True)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, (data_iter_step + 1) / len(data_loader) + epoch, args)

        pred_obj, pred_cls, pred_iou, targets = model(samples, targets)


        loss_confidence, loss_class, loss_iou, loss = criterion(pred_obj, pred_cls, pred_iou, targets)

        loss_value = loss.item()
        # print(loss_value, loss_confidence, loss_class, loss_iou)

        # check loss
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)


        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            
        if args.distributed:
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_dict = dict(
            loss_confidence=loss_confidence,
            loss_class=loss_class,
            loss_iou=loss_iou,
            total_loss=loss * accum_iter
        )
        loss_dict_reduced = misc.reduce_loss_dict(loss_dict)

        
        # 加入tensorboard便于显示
        if log_writer is not None and (data_iter_step + 1) % (accum_iter*20) == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_100x = int(
                ((data_iter_step + 1) / len(data_loader) + epoch) * 100)
            log_writer.add_scalar('loss_confidence', loss_dict_reduced['loss_confidence'].item(), epoch_100x)
            log_writer.add_scalar('loss_class', loss_dict_reduced['loss_class'].item(), epoch_100x)
            log_writer.add_scalar('loss_iou', loss_dict_reduced['loss_iou'].item(), epoch_100x)
            log_writer.add_scalar('total_loss', loss_dict_reduced['total_loss'].item(), epoch_100x)
            log_writer.add_scalar('lr', max_lr, epoch_100x)
            
            
            
            print('[Epoch %3d/%3d][Iter %2d/%2d][lr %.6f][Loss: total %.2f || confidence %.2f || class %.2f || iou %.2f || size %d || time %.2f]'
                        % (epoch+1, 
                        args.epochs, 
                        (data_iter_step+1)//accum_iter,
                        len(data_loader)//accum_iter,
                        max_lr,
                        loss_dict_reduced['total_loss'].item(),
                        loss_dict_reduced['loss_confidence'].item(),
                        loss_dict_reduced['loss_class'].item(),
                        loss_dict_reduced['loss_iou'].item(),
                        model.img_size,
                        time.time() - start),
                        flush=True)
            start = time.time()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# In[3]:


@torch.no_grad()
def evaluate(args, data_loader, model, device):
    if model.img_size != args.input_size:
        model.set_grid(args.input_size)

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    # all_target = []
    # all_preds_info = []
    for batch in metric_logger.log_every(data_loader, 1, header):
        images = batch[0]
        target = batch[1]
#         size = batch[2]
#         scale = batch[3]
#         offset = batch[4]
#         images_name = batch[5]
        
        images = images.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)
        
#         size = size.to(device, non_blocking=True)
#         scale = scale.to(device, non_blocking=True)
#         offset = offset.to(device, non_blocking=True)
#         images_name = images_name.to(device, non_blocking=True)
        
        # compute output

        preds_info = model(images)
    
        if args.vis_predict:
            vis_prediction(images.detach().cpu(), preds_info, target)

        # all_target = all_target + target
        # all_preds_info = all_preds_info + preds_info
        #===========计算mAP==========
        iou_mAP, n_target = get_mAP(preds_info, target, iou_thresh, args.nb_classes, device)


        # batch_size = args.batch_size
        batch_size = images.shape[0]
        for i in range(len(iou_thresh)):
            metric_logger.meters[f'{iou_thresh[i]}'].update(iou_mAP[i].item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    p_message = [f'* mAP_{k}: {meter.global_avg:.4f}' for k, meter in metric_logger.meters.items()]

    print("||".join(p_message))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# In[ ]:




