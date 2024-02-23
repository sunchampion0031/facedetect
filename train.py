#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from Config.yolo import yolo_config
from Util import lr_decay as lrd
from Util import misc as misc
from Dataset.build_dataset import Dataset
from Dataset.transforms import TrainTransforms, ColorTransforms, ValTransforms
from engine import train_one_epoch, evaluate
from Model.yolo import build_model
from Util.criterion import build_criterion
from Util.data_collate import train_collate, val_collate
from Util.misc import background_step


# In[2]:


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='YOLOv4 for face detection', add_help=False)

    # 训练的基本超参数
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--best_status', default=0.0, type=float, metavar='N',
                        help='best status')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    # 通用训练策略 xc
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--weight_decay', type=float, default=0.005,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--eval_epoch', type=int, default=5,
                        help='interval between evaluations')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # 专用训练策略
    parser.add_argument('--multi_scale', action='store_false',
                        help='use multi-scale trick')
    parser.add_argument('--multi_scale_range', nargs='+', default=[10, 18], type=int,
                        help='multi_scale_range')
    parser.add_argument('--multi_scale_change', default=10, type=int,
                        help='multi_scale_change')
    parser.add_argument('--multi_anchor', action='store_false', default=True,
                        help='use multiple anchor boxes as the positive samples')
    parser.add_argument('--center_sample', action='store_true',
                        help='use center sample for labels')

    # 数据集参数
    # parser.add_argument('--data_path', default='/root/autodl-tmp/data/', type=str,
    #                     help='dataset path')
    parser.add_argument('--data_path', default='YOLO/Data', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default='Face_Mask_Detection', type=str,
                        help='dataset set')
    parser.add_argument('--data_set_train', default='train', type=str,
                        help='dataset set train')
    parser.add_argument('--data_set_val', default='val', type=str,
                        help='dataset set val')
    parser.add_argument('--data_set_test', default='test', type=str,
                        help='dataset set test')
    parser.add_argument('--nb_classes', default=3, type=int,
                        help='number of the classification types')

    # 模型参数
    parser.add_argument('--model', default='yolov4', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=608, type=int,
                        help='images input size')
    parser.add_argument('--num_anchor', default=9, type=int,
                        help='all number of initial anchor')

    # 损失计算
    parser.add_argument('--loss_confidence_weight', default=1, type=float,
                        help='weight of obj loss')
    parser.add_argument('--loss_class_weight', default=1, type=float,
                        help='weight of cls loss')
    parser.add_argument('--loss_iou_weight', default=1, type=float,
                        help='weight of reg loss')
    parser.add_argument('--scale_loss', default='batch', type=str,
                        help='scale loss: batch or positive samples')
    parser.add_argument('--loss_confidence_f_s', default='mse', type=str,
                        help='confidence loss: mse or bce')
    parser.add_argument('--loss_iou_s', default='ciou', type=str,
                        help='Loss iou')

    # 数据增强
    parser.add_argument('--mosaic', action='store_false', default=True,
                        help='use Mosaic Augmentation trick')
    parser.add_argument('--mixup', action='store_false', default=True,
                        help='use MixUp Augmentation trick')

    # 预训练模型导入
    # 迁移学习训练时把MAE模型的参数导入finetune模型之中，encoder参数名都是一样的
    # parser.add_argument('--finetune', default='/root/autodl-tmp/yolov4_1/models/backbone/weights/cspdarknet53/cspdarknet53.pth',
    #                     help='start finetune from checkpoint')
    # parser.add_argument('--resume', default='/root/autodl-tmp/yolov4_1/trained_model/yolov4-150-(0.5103).pth',
    #                     help='resume from checkpoint')
    parser.add_argument('--finetune', default='',
                        help='start finetune from checkpoint')
    # YOLO/TempOutput/Weight/yolov4-0.5951.pth
    # YOLO/Weight/yolo.pth
    parser.add_argument('--resume', default='YOLO/TempOutput/Weight/yolov4-0.5951.pth',
                        help='resume from checkpoint')

    # 模型结果保存
    # parser.add_argument('--output_dir', default='/root/autodl-tmp/yolov4_1/trained_model',
    #                      help='path where to save, empty for no saving')
    # parser.add_argument('--log_dir', default='/root/tf-logs/yolov4_1',
    #                      help='path where to tensorboard log')
    parser.add_argument('--output_dir', default='YOLO/TempOutput/Weight',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='YOLO/TempOutput/Log',
                        help='path where to tensorboard log')

    # 训练初始化参数
    # parser.add_argument('--device', default='cuda',
    #                     help='device to use for training / testing')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    # parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # 分布式训练参数
    """
    world_size：指进程组中的进程数量
        若使用单台机器多GPU，world_size表示使用的GPU数量
        若使用多台机器单GPU，world_size表示使用的机器数量
        DDP的最佳使用方式是每一个GPU使用一个进程
    """
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')  #环境变量初始化方式，需要在环境变量中配置4个参数 MASTER_PORT，MASTER_ADDR，WORLD_SIZE，RANK
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')

    # 测试
    parser.add_argument('--eval', action='store_false',
                        help='Perform evaluation only')
    parser.add_argument('--conf_thresh', default=0.5, type=float,
                        help='NMS threshold')
    parser.add_argument('--soft_nms_thresh', default=0.6, type=float,
                        help='NMS threshold')
    parser.add_argument('--soft_nms_sigma', default=0.5, type=float,
                        help='NMS sigma')
    parser.add_argument('--soft_nms_iou', default='diou', type=str,
                        help='NMS iou')

    # 结果可视化显示
    parser.add_argument('--vis_data', action='store_true', default=False,
                        help='visualize images and labels.')
    parser.add_argument('--vis_targets', action='store_true', default=False,
                        help='visualize assignment.')
    parser.add_argument('--vis_predict', action='store_false',
                        help='visualize images and labels of the prediction.')

    return parser


# In[3]:


def main(args):
    # =================分布式多或单gpu/单CPU训练配置================
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.getcwd())))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    print(device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # ====================创建dataset、sampler、dataloader============
    dataset_train = Dataset(
        data_dir=args.data_path,
        img_size=args.input_size,
        image_sets=[
                    # ('MAFA', args.data_set_train),
                    ('Face_Mask_Detection', args.data_set_train),
                    # ('WiderFace', args.data_set_train)
                    ],
        transform=TrainTransforms,
        color_augment=ColorTransforms,
        mosaic=args.mosaic,
        mixup=args.mixup,
    )

    dataset_val = Dataset(
        data_dir=args.data_path,
        img_size=args.input_size,
        image_sets=[
                    ("MAFA", args.data_set_val),
                    ("Face_Mask_Detection", args.data_set_val),
                    ("WiderFace", args.data_set_val),
                    ],
        transform=ValTransforms,
        color_augment=ColorTransforms,
        mosaic=False,
        mixup=False,
    )

    if args.distributed:  # True:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=train_collate
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=train_collate
    )

    # =============创建loss_scaler（反向传播）、criterion（计算loss）==============
    loss_scaler = background_step()

    criterion = build_criterion(scale_loss=args.scale_loss, loss_confidence_f_s=args.loss_confidence_f_s,
                                label_smoothing=args.smoothing, loss_confidence_weight=args.loss_confidence_weight,
                                loss_class_weight=args.loss_class_weight, loss_iou_weight=args.loss_iou_weight
                                )

    print("criterion = %s" % str(criterion))

    # ==========创建model，并导入迁移学习的参数， 将model放入相应设备gpu/cpu=======================================
    cfg = yolo_config[args.model]
    #     args.anchor_size = kmeans_anchor(args.num_anchor, args.input_size, args.data_path, args.data_set)
    args.anchor_size = [[  5.21,   6.97],[ 10.75,  14.21],[ 20.01,  25.78],[ 32.47,  44.75],[ 51.65,  69.79],[ 77.75, 103.28],[113.56, 150.48],[158.43, 211.24],[223.33, 287.21]]

    model = build_model(args=args,
                    cfg=cfg,
                    device=device,
                    num_classes=args.nb_classes)

    # 训练阶段导入迁移学习模型训练好的checkpoint，不用从头训练。如果是测试阶段，应该从resume导入训练好的模型，而不是导入没有训练过的参数。
    # 如果有训练好的模型，就不需要从MAE的预训练模型finetune中导入数据了,浪费时间
    if args.finetune and not args.eval and not args.resume:
        load_model = model.backbone
        print("Load pre-trained backbone checkpoint from: %s" % args.finetune)
        checkpoint_state = torch.load(args.finetune, map_location='cpu')["model"]
        model_state = load_model.state_dict()
        # check
        for k in list(checkpoint_state.keys()):
            if k in model_state:
                shape_model = tuple(model_state[k].shape)
                shape_checkpoint = tuple(checkpoint_state[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state.pop(k)
            else:
                checkpoint_state.pop(k)
        """
        strict=False, 如果参数的名称对的上但是值的形状对不上就会有问题，无法加载
        所以前面接一个check
        """
        msg = load_model.load_state_dict(checkpoint_state, strict=False)
        print(msg)

    # =============冻结前面的backbone,训练head=====================
    # for n, p in model.named_parameters():
    #     if n.startswith("head"):
    #         p.requires_grad = True
    #     else:
    #         p.requires_grad = False

    model.to(device)
    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    # ==========如果是多卡环境，配置模型分布式训练=============
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # =======计算真实的batch_size, 学习率======================
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)


    # =======创建optimizer,配置每层的学习率衰减和权重衰减==============
    # build optimizer with layer-wise lr decay (lrd)
    """
    逐层配置model的parameters（），只有requires_grad=True的param才需要放入optimizer
    学习率：所有层的参数都配置，前面几层（cls_token,pos_embed,patch_embed)的学习率最小，后面几层（norm,head)的学习率大
    权重衰减：除了指定的几个层以及数据维度为一（bias）的层不权重衰减外，其他的都按指定值权重衰减
    """
    """
    因为每一层都是单独配置的（不是通过Adma(model.parameters()统一配置的)，
    这样optimizer.param_groups()列表的长度就不是一了
    [{'params': [tensor([[-1.5916, -1.6110, -0.5739],
        [ 0.0589, -0.5848, -0.9199],
        [-0.4206, -2.3198, -0.2062]], requires_grad=True)], 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False, 'maximize': False}, 
    {'params': [tensor([[-0.5546, -1.2646,  1.6420],
        [ 0.0730, -0.0460, -0.0865],
        [ 0.3043,  0.4203, -0.3607]], requires_grad=True)], 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False, 'maximize': False}]

    """

    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=[],
                                        layer_decay=args.layer_decay
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    # =======计算模型的参数数量，=================
    model_without_ddp.eval()
    with torch.no_grad():
        params = sum([param.nelement() for param in model_without_ddp.parameters()])

    # ====创建tensorboard，记录训练时的loss、学习率等参数，以便可视化显示========
    if misc.get_rank() == 0 and args.log_dir is not None and not args.eval:
        c_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        log_path = os.path.join(args.log_dir, c_time)
        os.makedirs(log_path, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_path)
    else:
        log_writer = None

    # ==========如果有训练好的成品模型参数，则导入模型,前面配置的model\optimizer\loss_scale全部替换===============
    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    # =============再次冻结前面的backbone,只训练head，保险点=====================
    # for n, p in model_without_ddp.named_parameters():
    #     if n.startswith("head"):
    #         p.requires_grad = True
    #     else:
    #         p.requires_grad = False


    # ===============训练或者测试评估=====================
    if args.eval:
        test_stats = evaluate(args, data_loader_val, model, device)
        mAP = torch.mean(torch.tensor(list(test_stats.values()), dtype=torch.float32))
        print(
            f"mAP of the network on the {len(dataset_val)} test images: {mAP:.4f}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = args.best_status
    print("max_accuracy: ", max_accuracy)
    save_interval = 0
    args.start_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            args=args,
            criterion=criterion
        )
        save_interval = save_interval + 1
        # =====每eval_epoch个或最后一个epoch都跑一次验证集，若效果变好就存储一次模型==========
        if (epoch + 1) % args.eval_epoch == 0 or (epoch + 1) == args.epochs:
            print()
            test_stats = evaluate(args, data_loader_val, model, device)
            mAP = torch.mean(torch.tensor(list(test_stats.values()), dtype=torch.float32))
            print(
                f"mAP of the network on the {len(dataset_val)} test images: {mAP:.4f}")
            if mAP > max_accuracy or (epoch + 1) == args.epochs or save_interval >= 5:
                if mAP > max_accuracy:
                    max_accuracy = mAP
                if args.output_dir:
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch + 1, mAP=mAP)
                    print("save model at %d epoch" % (epoch + 1))
                    save_interval = 0
            print(f'Max mAP: {max_accuracy:.4f}')

            if log_writer is not None:
                for k, v in test_stats.items():
                    log_writer.add_scalar(f'val/mAP_{k}', v, epoch)

            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            log_stats = {"time": str(now), "model": args.model,
                         **{f'train_{k}': round(v, 4) for k, v in train_stats.items()},
                         **{f'test_{k}': round(v, 4) for k, v in test_stats.items()},
                        'epoch': epoch + 1,
                        'n_parameters': params}

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        # close mosaic augmentation
        if args.mosaic and args.epochs - epoch == 15:
            print('close Mosaic Augmentation ...')
            data_loader_train.dataset.mosaic = False
        # close mixup augmentation
        if args.mixup and args.epochs - epoch == 15:
            print('close Mixup Augmentation ...')
            data_loader_train.dataset.mixup = False
    if log_writer is not None:
        log_writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))





if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_known_args()[0]
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

