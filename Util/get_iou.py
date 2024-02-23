#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import torch
import numpy as np


# In[2]:


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    
    """
    多对多
    :params box1: [B*(H3*W3 + H4*W4 + H5*W5)*num_A, 4] [x1, y1, x2, y2]
    :params box2: [B*(H3*W3 + H4*W4 + H5*W5)*num_A, 4] [x1, y1, x2, y2]
    :return box1和box2的IoU/GIoU/DIoU/CIoU
      
    一对一
    :params box1: [1, 4]/[4] [x1, y1, x2, y2]
    :params box2: [1, 4]/[4] [x1, y1, x2, y2]
    :return box1和box2的IoU/GIoU/DIoU/CIoU
    
    一对多/多对一
    :params box1/2: [1, 4]/[4] [x1, y1, x2, y2]
    :params box2/1: [N, 4] [x1, y1, x2, y2]
    :return box1和box2的IoU/GIoU/DIoU/CIoU
    """
    
    multiple = box2

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[...,0], box1[...,1], box1[...,2], box1[...,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[...,0], box2[...,1], box2[...,2], box2[...,3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[...,0] - box1[...,2] / 2, box1[...,0] + box1[...,2] / 2
        b1_y1, b1_y2 = box1[...,1] - box1[...,3] / 2, box1[...,1] + box1[...,3] / 2
        b2_x1, b2_x2 = box2[...,0] - box2[...,2] / 2, box2[...,0] + box2[...,2] / 2
        b2_y1, b2_y2 = box2[...,1] - box2[...,3] / 2, box2[...,1] + box2[...,3] / 2

    # Intersection area   tensor.clamp(0): 将矩阵中小于0的元数变成0
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    # print(np.any(np.isnan(inter.cpu().detach().numpy())))
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    # print(np.any(np.isnan(union.cpu().detach().numpy())))
    iou = inter / union
    # print(np.any(np.isnan(iou.cpu().detach().numpy())))
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # 两个框的最小闭包区域的width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # 两个框的最小闭包区域的height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


# In[14]:


# if __name__ == '__main__':
# #     bboxes_a = torch.tensor([[10, 10, 20, 20],[10, 10, 20, 20],[10, 10, 20, 20],[10, 10, 20, 20]])
# #     bboxes_b = torch.tensor([[13, 15, 27, 25],[13, 15, 27, 25],[13, 15, 27, 25],[13, 15, 27, 25]])
# #     iou = bbox_iou(bboxes_a, bboxes_b, CIoU=True)
# #     print(iou)
#     n = [10, 12,12]
#     bboxes_a = torch.tensor([[13, 15, 27, 25], [13, 15, 27, 25], [13, 15, 27, 25]])
#     bboxes_b = torch.tensor([10, 10, 20, 20])
#     iou = bbox_iou(bboxes_a, bboxes_b, CIoU=True)
#     print(iou)
#     print(torch.max(iou, dim=0))
#     print(n[torch.max(iou, dim=0)[1]])


# In[4]:


# tensor([[[0.0733],
#          [0.0733]],

#         [[0.0733],
#          [0.0733]]])
# tensor([[0.0733, 0.0733],
#         [0.0733, 0.0733]])
# tensor([[0.0733, 0.0733, 0.0733]])
# tensor([[0.1707]])
# tensor([[-0.0253]])
# tensor([[0.0733]])

