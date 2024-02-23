#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math


# In[ ]:


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    # 在训练开始的时候先选择使用一个较小的学习率，训练了一些steps（15000steps）或者epoches(5epoches),再修改为预先设置的学习来进行训练
    # 模型趋于稳定后，再加速训练
    if epoch <= args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        # 预热后以半周期余弦衰减学习率
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        # lr_scale 学习率按网络层的衰减或递增
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

