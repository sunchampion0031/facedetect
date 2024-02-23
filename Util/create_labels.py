#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import torch


# In[8]:


def compute_iou(anchor_boxes, gt_box):
    """
    都是像素值
    anchoe_boxes[[0,0,anchor_w, anchor_h], ...]
    gt_box:[0, 0, anchor_w, anchor_h]
    多对一，计算的IOU是同一个中心点，所以IOU不可能为零，看哪一个anchor和该box比较匹配
    Input:
        anchor_boxes : ndarray -> [[xc_s, yc_s, anchor_w, anchor_h], ..., [xc_s, yc_s, anchor_w, anchor_h]].
        gt_box : ndarray -> [xc_s, yc_s, anchor_w, anchor_h].
    Output:
        iou : ndarray -> [iou_1, iou_2, ..., iou_m], and m is equal to the number of anchor boxes.
    """
    # compute the iou between anchor box and gt box
    # First, change [xc_s, yc_s, anchor_w, anchor_h] ->  [x1, y1, x2, y2]
    # anchor box :
    ab_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    ab_x1y1_x2y2[:, 0] = anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2  # x1
    ab_x1y1_x2y2[:, 1] = anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2  # y1
    ab_x1y1_x2y2[:, 2] = anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2  # x2
    ab_x1y1_x2y2[:, 3] = anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2  # y2
    w_ab, h_ab = anchor_boxes[:, 2], anchor_boxes[:, 3]
    
    # gt_box : 
    # We need to expand gt_box(ndarray) to the shape of anchor_boxes(ndarray), in order to compute IoU easily. 
    gt_box_expand = np.repeat(gt_box, len(anchor_boxes), axis=0)

    gb_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    gb_x1y1_x2y2[:, 0] = gt_box_expand[:, 0] - gt_box_expand[:, 2] / 2 # x1
    gb_x1y1_x2y2[:, 1] = gt_box_expand[:, 1] - gt_box_expand[:, 3] / 2 # y1
    gb_x1y1_x2y2[:, 2] = gt_box_expand[:, 0] + gt_box_expand[:, 2] / 2 # x2
    gb_x1y1_x2y2[:, 3] = gt_box_expand[:, 1] + gt_box_expand[:, 3] / 2 # y1
    w_gt, h_gt = gt_box_expand[:, 2], gt_box_expand[:, 3]

    # Then we compute IoU between anchor_box and gt_box
    S_gt = w_gt * h_gt
    S_ab = w_ab * h_ab
    
    I_w = np.minimum(gb_x1y1_x2y2[:, 2], ab_x1y1_x2y2[:, 2]) - np.maximum(gb_x1y1_x2y2[:, 0], ab_x1y1_x2y2[:, 0])
    I_h = np.minimum(gb_x1y1_x2y2[:, 3], ab_x1y1_x2y2[:, 3]) - np.maximum(gb_x1y1_x2y2[:, 1], ab_x1y1_x2y2[:, 1])
    
    S_I = I_h * I_w
    U = S_gt + S_ab - S_I + 1e-20
    IoU = S_I / U
    
    return IoU


# In[9]:


def set_anchors(anchor_size):
    """
    Input:
        anchor_size : list -> [[w_1, h_1], [w_2, h_2], ..., [w_n, h_n]].
    Output:
        anchor_boxes : ndarray -> [[0, 0, anchor_w, anchor_h],
                                   [0, 0, anchor_w, anchor_h],
                                   ...
                                   [0, 0, anchor_w, anchor_h]].
    """
    num_anchors = len(anchor_size)
    anchor_boxes = np.zeros([num_anchors, 4])
    for index, size in enumerate(anchor_size): 
        anchor_w, anchor_h = size
        anchor_boxes[index] = np.array([0, 0, anchor_w, anchor_h])
    
    return anchor_boxes


# In[10]:


def label_assignment_with_anchorbox(anchor_size, target_boxes, num_anchors, strides, multi_anchor=False):
    """
    都是像素值
    anchor_size : list -> [[w_1, h_1], [w_2, h_2], ..., [w_n, h_n]].
    target_boxes: list -> [center_x, center_y, W, H]
    num_anchors: 每一个多级检测头，有几个锚框 3
    strides：[stride1, stride2, stride3], 为了确定分配到哪一个grid
    multi_anchor：multi_anchor策略，True：当有IOU大于阈值时，全部作为正样本，不忽略，否则选最大的
                    False: 正样本就选择IoU最大的，其他都为负样本，不忽略。
    """
    # prepare
    anchor_boxes = set_anchors(anchor_size)
    gt_box = np.array([[0, 0, target_boxes[2], target_boxes[3]]])

    # compute IoU
    iou = compute_iou(anchor_boxes, gt_box)

    label_assignment_results = []
    if multi_anchor:
        # We consider those anchor boxes whose IoU is more than 0.5,
        iou_mask = (iou > 0.5)
        if iou_mask.sum() == 0:
            # We assign the anchor box with highest IoU score.
            iou_ind = np.argmax(iou)

            # scale_ind, anchor_ind = index // num_scale, index % num_scale
            scale_ind = iou_ind // num_anchors
            anchor_ind = iou_ind - scale_ind * num_anchors

            # get the corresponding stride
            stride = strides[scale_ind]

            # compute the grid cell
            xc_s = target_boxes[0] / stride
            yc_s = target_boxes[1] / stride
            grid_x = int(xc_s)
            grid_y = int(yc_s)

            label_assignment_results.append([grid_x, grid_y, scale_ind, anchor_ind])
        else:            
            for iou_ind, iou_m in enumerate(iou_mask):
                if iou_m:
                    # scale_ind, anchor_ind = index // num_scale, index % num_scale
                    scale_ind = iou_ind // num_anchors
                    anchor_ind = iou_ind - scale_ind * num_anchors

                    # get the corresponding stride
                    stride = strides[scale_ind]

                    # compute the gride cell
                    xc_s = target_boxes[0] / stride
                    yc_s = target_boxes[1] / stride
                    grid_x = int(xc_s)
                    grid_y = int(yc_s)

                    label_assignment_results.append([grid_x, grid_y, scale_ind, anchor_ind])

    else:
        # We assign the anchor box with highest IoU score.
        iou_ind = np.argmax(iou)

        # scale_ind, anchor_ind = index // num_scale, index % num_scale
        scale_ind = iou_ind // num_anchors
        anchor_ind = iou_ind - scale_ind * num_anchors

        # get the corresponding stride
        stride = strides[scale_ind]

        # compute the grid cell
        xc_s = target_boxes[0] / stride
        yc_s = target_boxes[1] / stride
        grid_x = int(xc_s)
        grid_y = int(yc_s)

        label_assignment_results.append([grid_x, grid_y, scale_ind, anchor_ind])

    return label_assignment_results


# In[11]:


def label_assignment_without_anchorbox(target_boxes, strides):
    # no anchor box
    scale_ind = 0
    anchor_ind = 0

    label_assignment_results = []
    # get the corresponding stride
    stride = strides[scale_ind]

    # compute the grid cell
    xc_s = target_boxes[0] / stride
    yc_s = target_boxes[1] / stride
    grid_x = int(xc_s)
    grid_y = int(yc_s)
    
    label_assignment_results.append([grid_x, grid_y, scale_ind, anchor_ind])
            
    return label_assignment_results


# In[12]:


def gt_creator(img_size, strides, label_lists, anchor_size=None, multi_anchor=False, center_sample=False):
    """
    创建target
    img_size：int 为了将像素的比例还原为像素值
    strides：[stride1, stride2, stride3]，确定targrt的大小
    label_lists：比例 [[[xmin, ymin, xmax, ymax, label_ind],[xmin, ymin, xmax, ymax, label_ind],...],... ]
    anchor_size: 3 每一个多级检测头，有几个锚框,若为None 则为不用锚框
    multi_anchor:
    center_sample: True:将它右、下、右下，都作为正样本。
    """
    # prepare
    batch_size = len(label_lists)
    img_h = img_w = img_size
    num_scale = len(strides)
    gt_tensor = []
    KA = len(anchor_size) // num_scale if anchor_size is not None else 1

    for s in strides:
        fmp_h, fmp_w = img_h // s, img_w // s
        # [B, H, W, KA, obj+cls+box+scale]
        gt_tensor.append(np.zeros([batch_size, fmp_h, fmp_w, KA, 1+1+4+1]))
        
    # generate gt datas  
    for bi in range(batch_size):
        label = label_lists[bi]
        for box_cls in label:
            # get a bbox coords
            cls_id = int(box_cls[-1])
            x1, y1, x2, y2 = box_cls[:-1]
            # [x1, y1, x2, y2] -> [xc, yc, bw, bh]
            xc = (x2 + x1) / 2 * img_w
            yc = (y2 + y1) / 2 * img_h
            bw = (x2 - x1) * img_w
            bh = (y2 - y1) * img_h
            target_boxes = [xc, yc, bw, bh]
            box_scale = 2.0 - (bw / img_w) * (bh / img_h)

            # check label 
            # 如果bounding box太小则舍弃
            if bw < 1. or bh < 1.:
                # print('A dirty data !!!')
                continue

            # label assignment
            if anchor_size is not None:
                # use anchor box
                label_assignment_results = label_assignment_with_anchorbox(
                                                anchor_size=anchor_size,
                                                target_boxes=target_boxes,
                                                num_anchors=KA,
                                                strides=strides,
                                                multi_anchor=multi_anchor)
            else:
                # no anchor box
                label_assignment_results = label_assignment_without_anchorbox(
                                                target_boxes=target_boxes,
                                                strides=strides)

            # make labels
            for result in label_assignment_results:
                grid_x, grid_y, scale_ind, anchor_ind = result
                if center_sample:
                    # We consider four grid points near the center point
                    for j in range(grid_y, grid_y+2):
                        for i in range(grid_x, grid_x+2):
                            if (j >= 0 and j < gt_tensor[scale_ind].shape[1]) and (i >= 0 and i < gt_tensor[scale_ind].shape[2]):
                                gt_tensor[scale_ind][bi, j, i, anchor_ind, 0] = 1.0
                                gt_tensor[scale_ind][bi, j, i, anchor_ind, 1] = cls_id
                                gt_tensor[scale_ind][bi, j, i, anchor_ind, 2:6] = np.array([x1, y1, x2, y2])
                                gt_tensor[scale_ind][bi, j, i, anchor_ind, 6] = box_scale
                else:
                    # We ongly consider top-left grid point near the center point
                    if (grid_y >= 0 and grid_y < gt_tensor[scale_ind].shape[1]) and (grid_x >= 0 and grid_x < gt_tensor[scale_ind].shape[2]):
                        gt_tensor[scale_ind][bi, grid_y, grid_x, anchor_ind, 0] = 1.0
                        gt_tensor[scale_ind][bi, grid_y, grid_x, anchor_ind, 1] = cls_id
                        gt_tensor[scale_ind][bi, grid_y, grid_x, anchor_ind, 2:6] = np.array([x1, y1, x2, y2])
                        gt_tensor[scale_ind][bi, grid_y, grid_x, anchor_ind, 6] = box_scale

                   

    gt_tensor = [gt.reshape(batch_size, -1, 1+1+4+1) for gt in gt_tensor]
    gt_tensor = np.concatenate(gt_tensor, axis=1)
    # [B,  (H3*W3 + H4*W4 + H5*W5)*num_A, 1+1+4+1]
    return torch.from_numpy(gt_tensor).float()


# In[13]:


if __name__ == "__main__":
    gt_box = np.array([[0.0, 0.0, 10, 10]])
    anchor_boxes = np.array([[0.0, 0.0, 10, 10], 
                             [0.0, 0.0, 4, 4], 
                             [0.0, 0.0, 8, 8], 
                             [0.0, 0.0, 16, 16]
                             ])
    iou = compute_iou(anchor_boxes, gt_box)
    print(iou)


# In[ ]:




