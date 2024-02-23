#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# In[2]:


class MSEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x, label, label_obj):
        inputs = x.sigmoid()
        # mse loss
        # reduction="none":不进行sum/mean。函数返回的是一个batch中每个样本的损失，结果为向量
        loss = F.mse_loss(input=inputs, 
                          target=label,
                          reduction="none")
        positive_loss = loss * label_obj * 5.0
        negative_loss = loss * (1.0 - label_obj) * 1.0
        loss = positive_loss + negative_loss
        if self.reduction == 'mean':
            loss = loss.mean()

        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


# In[3]:


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, positive_weight=1.0, negative_weight=0.25, reduction='mean'):
        super().__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.reduction = reduction

    def forward(self, x, label, label_obj):
        # bce loss
        # BCEWithLogitsLoss函数是将Sigmoid层和BCELoss合并在一个类中
        loss = F.binary_cross_entropy_with_logits(input=x, target=label, reduction="none")
        positive_loss = loss * label_obj * self.positive_weight
        negative_loss = loss * (1.0 - label_obj) * self.negative_weight
        loss = positive_loss + negative_loss

        if self.reduction == 'mean':
            loss = loss.mean()

        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


# In[4]:


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

             
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ''' 
        Args:
            pred: prediction of model output       [B, C, HW]
            target: ground truth of sampler     [B, HW]
        '''        
        # [B, HW, C]
        pred = pred.permute(0, 2, 1)
        self.class_num = pred.shape[-1]
        
        # cross entropy loss with label smoothing
        logprobs = F.log_softmax(pred, dim=-1)  # softmax + log
        
        # [B, HW, C]
        target = F.one_hot(target, self.class_num)  # 转换成one-hot

        # label smoothing
        # 实现 1
        # target = (1.0-self.label_smooth)*target + self.label_smooth/self.class_num 	
        # 实现 2
        # implement 2
        target = torch.clamp(target.float(), min=self.smoothing/(self.class_num-1), max=self.confidence)
        
        # [B, HW]
        loss = -1*torch.sum(target*logprobs, dim=-1)
        
        return loss

    


# In[5]:


class Criterion(nn.Module):
    def __init__(self,
                 scale_loss = 'batch',                   # batch/positive
                 loss_confidence_f_s = 'mse',            # mse/bce
                 label_smoothing = 0,                     # 0.1
                 loss_confidence_weight=1.0, 
                 loss_class_weight=1.0, 
                 loss_iou_weight=1.0, 
                ):
        super().__init__()
        self.scale_loss = scale_loss
        self.loss_confidence_f_s = loss_confidence_f_s
        self.loss_confidence_weight = loss_confidence_weight
        self.loss_class_weight = loss_class_weight
        self.loss_iou_weight = loss_iou_weight

        # objectness loss
        try:
            if self.loss_confidence_f_s == 'mse':
                self.confidence_loss_f = MSEWithLogitsLoss(reduction='none')
            elif self.loss_confidence_f_s == 'bce':
                self.confidence_loss_f = BCEWithLogitsLoss(reduction='none')
        except:
            self.confidence_loss_f = MSEWithLogitsLoss(reduction='none')
        # class loss
        if label_smoothing > 0 and label_smoothing < 1:
            self.class_loss_f = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            self.class_loss_f = nn.CrossEntropyLoss(reduction='none')


    def loss_confidence(self, pred_confidence, label_confidence, label_obj):
        """
            pred_confidence: (FloatTensor) [B, HW, 1]
            label_confidence: (FloatTensor) [B, HW,]
            label_obj: (FloatTensor) [B, HW,]
        """
        # obj loss: [B, HW,]
        loss_confidence = self.confidence_loss_f(pred_confidence[..., 0], label_confidence, label_obj)

        if self.scale_loss == 'batch':
            # scale loss by batch size
            batch_size = pred_confidence.size(0)
            loss_confidence = loss_confidence.sum() / batch_size
        elif self.scale_loss == 'positive':
            # scale loss by number of positive samples
            num_pos = label_obj.sum().clamp(1.0)
            loss_confidence = loss_confidence.sum() / num_pos

        return loss_confidence


    def loss_class(self, pred_class, label_class, label_obj):
        """
            pred_class: (FloatTensor) [B, HW, C]
            label_class: (LongTensor) [B, HW,]
            label_obj: (FloatTensor) [B, HW,]
        """
        # [B, HW, C] -> [B, C, HW]
        pred_class = pred_class.permute(0, 2, 1)
#         print(pred_class.shape)
#         print(label_class.shape)
        # reg loss: [B, HW, ]
        loss_class = self.class_loss_f(pred_class, label_class)
        # valid loss. Here we only compute the loss of positive samples
        loss_class = loss_class * label_obj

        if self.scale_loss == 'batch':
            # scale loss by batch size
            batch_size = pred_class.size(0)
            loss_class = loss_class.sum() / batch_size
        elif self.scale_loss == 'positive':
            # scale loss by number of positive samples
            num_pos = label_obj.sum().clamp(1.0)
            loss_class = loss_class.sum() / num_pos

        return loss_class


    def loss_iou(self, pred_iou, label_obj, label_scale):
        """
            pred_iou: (FloatTensor) [B, HW, ]
            label_obj: (FloatTensor) [B, HW,]
            label_scale: (FloatTensor) [B, HW,]
        """
#         print(pred_iou.shape, label_obj.shape, label_scale.shape, sep="\n")

        # bbox loss: [B, HW,]
        loss_iou = 1. - pred_iou
        loss_iou = loss_iou * label_scale
        # valid loss. Here we only compute the loss of positive samples
        loss_iou = loss_iou * label_obj

        if self.scale_loss == 'batch':
            # scale loss by batch size
            batch_size = pred_iou.size(0)
            loss_iou = loss_iou.sum() / batch_size
        elif self.scale_loss == 'positive':
            # scale loss by number of positive samples
            num_pos = label_obj.sum().clamp(1.0)
            loss_iou = loss_iou.sum() / num_pos

        return loss_iou


    def forward(self, pred_confidence, pred_class, pred_iou, labels):
        """
            pred_confidence: (Tensor) [B,  (H3*W3 + H4*W4 + H5*W5)*num_A, 1]
            pred_class: (Tensor) [B,  (H3*W3 + H4*W4 + H5*W5)*num_A, C]
            pred_iou: (Tensor) [B,  (H3*W3 + H4*W4 + H5*W5)*num_A, 1]
            labels: (Tensor) [B, (H3*W3 + H4*W4 + H5*W5)*num_A, 1+1+1+4+1] [iou_objectness, confidence, class, x1y1x2y2, weight]
        """
#         print(pred_confidence.shape)
#         print(pred_class.shape)
#         print(pred_iou.shape)
#         print(labels.shape)
        # groundtruth
        label_confidence = labels[..., 0].float()     # [B, HW]

        label_obj = labels[..., 1].float()            # [B, HW]

        label_class = labels[..., 2].long()           # [B, HW]

        label_scale = labels[..., -1].float()         # [B, HW]


        # confidence loss
        loss_confidence = self.loss_confidence(pred_confidence, label_confidence, label_obj)
#         print(loss_confidence)
        # class loss
        loss_class = self.loss_class(pred_class, label_class, label_obj)
#         print(loss_class)
        # iou loss(x1y1x2y2) regression
        loss_iou = self.loss_iou(pred_iou.squeeze(-1), label_obj, label_scale)
#         print(loss_iou)
        # total loss
        losses = self.loss_confidence_weight * loss_confidence + \
                 self.loss_class_weight * loss_class + \
                 self.loss_iou_weight * loss_iou

        return loss_confidence, loss_class, loss_iou, losses




# In[6]:


def build_criterion(scale_loss = 'batch', loss_confidence_f_s = 'mse', label_smoothing = 0,
                   loss_confidence_weight=1, loss_class_weight=1, loss_iou_weight=1):
    criterion = Criterion(
                 scale_loss = scale_loss,               # batch/positive
                 loss_confidence_f_s = loss_confidence_f_s,            # mse/bce
                 label_smoothing = label_smoothing,
                 loss_confidence_weight=loss_confidence_weight, 
                 loss_class_weight=loss_class_weight, 
                 loss_iou_weight=loss_iou_weight, 
       )
    return criterion




