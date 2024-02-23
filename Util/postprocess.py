#!/usr/bin/env python
# coding: utf-8

# In[2]:

import sys
import torch
from .get_iou import bbox_iou


# ![%E6%88%AA%E5%B1%8F2023-03-24%2014.25.54.png](attachment:%E6%88%AA%E5%B1%8F2023-03-24%2014.25.54.png)

# In[6]:


def soft_nms(bboxes, scores, sigma=0.5, iou_s='diou', nms_thresh=0.3):
    # bboxes:[N, 4]
    # scores:[N, 1]

    # order [N], 从大到小排序 的索引值列表
    order = torch.argsort(scores, dim=0, descending=True).squeeze(-1)

    keep = []
    while order.shape[0] > 0:
        i = order[0]
        keep.append(i)
        # [4]
        main_bbox = bboxes[i]
        # [len(order)-1, 4]
        other_bboxex = bboxes[order[1:]]
        # 如果other_bboxex为空，表就示完成了nms，退出即可
        if other_bboxex.shape[0] == 0:
            break

        if iou_s == 'giou':
            iou = bbox_iou(main_bbox, other_bboxex,
                           GIoU=True, DIoU=False, CIoU=False)
            iou = 0.5 * (iou + 1)
        elif iou_s == 'diou':
            iou = bbox_iou(main_bbox, other_bboxex,
                           GIoU=False, DIoU=True, CIoU=False)
            iou = 0.5 * (iou + 1)
        elif iou_s == 'ciou':
            iou = bbox_iou(main_bbox, other_bboxex,
                           GIoU=False, DIoU=False, CIoU=True)
            iou = iou.clamp(0.)
        else:
            iou = bbox_iou(main_bbox, other_bboxex,
                           GIoU=False, DIoU=False, CIoU=False)

        # print("iou.shape", iou.shape)
        # [len(order)-1, 1]
        iou = iou.unsqueeze(-1)
        try:
            # 根据计算的iou更新score
            scores[order[1:]] = scores[order[1:]] * (torch.exp(-iou**2/sigma))
        except:
            print("main_bbox",main_bbox)
            print("other_bboxex", other_bboxex)
            print("main_bbox.shape", main_bbox.shape)
            print("other_bboxex.shape", other_bboxex.shape)
            print("iou", iou.flatten())
            print("iou.shape", iou.shape)
            print("score.shape",scores[order[1:]].shape)
            sys.exit(1)
        # 首先对[1:的score进行降序排序
        # 根据nms_thresh,将score<=nms_thresh的bbox，剔除掉。
        # 更新order
        score_sort = torch.argsort(
            scores[order[1:]], dim=0, descending=True).squeeze(-1) + 1
        order = order[score_sort]

        keep_ids = torch.where(scores[order] > nms_thresh)[0]

        order = order[keep_ids]

    return keep


# In[4]:


def postprocess(confidence_pred, class_pred, x1y1x2y2_pred, num_classes, iou_c='ciou',nms_thresh=0.3, conf_thresh=0.001, sigma=0.5):
       # conf_thresh: score过小的就抛弃掉，因为很可能识别不到物体
       # [B,  (H3*W3 + H4*W4 + H5*W5)*num_A, num_C]
       scores = torch.sigmoid(confidence_pred) * torch.softmax(class_pred, dim=-1)
       # [B,  (H3*W3 + H4*W4 + H5*W5)*num_A, 4]
       bboxes = torch.clamp(x1y1x2y2_pred, 0. , 1.)
       
       batches_info = []
       
       for batch_index in range(bboxes.shape[0]):
           # [(H3*W3 + H4*W4 + H5*W5)*num_A, num_C]
           score = scores[batch_index]
           # [(H3*W3 + H4*W4 + H5*W5)*num_A, 4]
           bbox = bboxes[batch_index]
           
           # 根据每个bbox最大的score，确定其的cls，并记录该score
           # [(H3*W3 + H4*W4 + H5*W5)*num_A, 1]
           cls_ind = torch.argmax(score, dim=-1, keepdim = True)
           # [(H3*W3 + H4*W4 + H5*W5)*num_A, 1]
           score = torch.gather(score, dim = -1, index=cls_ind)
           # for i in range(cls_ind.shape[0]):
           #     print(score[i], ": ", cls_ind[i], end='||')
           # print()
           # 仅保留大于conf_thresh地bbox
           # threshold
           keep = torch.where(score >= conf_thresh)[:-1]
           # [N, 4]
           bbox = bbox[keep]
           # [N, 1]
           score = score[keep]
           # [N, 1]
           cls_ind = cls_ind[keep]

           # NMS
           # 一个类别进行一次NMS
           # keep，记录保留的bbox的第一维索引，保留为1，不保留为0
           keep = torch.zeros(len(bbox), dtype=torch.int)
           for i in range(num_classes):
               # 类别为第i类的bbox的第一维索引（相对于cls_ind）
               inds = torch.where(cls_ind == i)[0]
               if len(inds) == 0:
                   continue
               # 提取出第i类bbox
               i_bbox = bbox[inds]
               i_score = score[inds]
               # i_keep:提取出的所有第i类bbox中保留的bbox的索引（相对于inds）
               i_keep = soft_nms(i_bbox, i_score, sigma, iou_c, nms_thresh)
               # i_keep = torch.tensor(i_keep, dtype=torch.int)
               i_keep = torch.tensor(i_keep, dtype=torch.long)
               try:
                    keep[inds[i_keep]] = 1
               except:
                   print("i_keep", i_keep)
                   print("inds.shape",inds.shape)
                   print("inds", inds)
                   print("inds_keep", inds[i_keep])
                   print("keep.shape",keep.shape)
                   print()
                   sys.exit(1)

           keep = torch.where(keep > 0)
           bbox = bbox[keep]
           score = score[keep]
           cls_ind = cls_ind[keep]
           
           # [N, 4+1+1]
           batch_info = torch.cat([bbox, score, cls_ind], dim=-1)

           batches_info.append(batch_info)
       
       return batches_info


# In[ ]:




