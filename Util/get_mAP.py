#!/usr/bin/env python
# coding: utf-8

# In[69]:


import torch
from .get_iou import bbox_iou
import numpy as np


# In[70]:


def get_mAP(pred, target, iou_thresh, num_class, device):
    """
    pred:是一个二维列表, 表示预测的边界框
         第一维表示是那张图片
         第二维表示是图片中的哪个预测的边界框[[bbox1, score, cls_id],[bbox2],[...],]
    target:是一个二维列表，表示ground truth
         第一维表示是那张图片
         第二维表示是图片中的哪个真实的边界框[[bbox1,cls_id],[bbox2],[...],]
    iou_thresh：是一个一维列表，代表一系列的iou阈值，用来计算TP、FP、FN
    output：一维列表，表示不同iou阈值下的mAP
    """
    
#     创建一个二维字典【GT_id】[iou_thresh],记录该 GT在该iou阈值下，是否已被匹配成功，全零初始化。
#     如果多个pred都和一个GT的iou>thresh，则保留confidence最大的一项
    
#     给target标上序号【GT_id，bbox，cls_id】，并按类别分类，将target变为三维列表[image, class, bbox_id]
#     【就不需要记录cls_id了】【GT_id，bbox】，记录每种类别总的ground truth数量
    
#     给pred标上序号【图片编号，bbox，score，cls_id】，将pred变成一个一维的列表
    
    
    pred_objs = []
    
    target_objs = [[[] for j in range(num_class)] for i in range(len(target))]
    num_classes = [[0] for i in range(num_class)]
    n_target = 0
    dec_dict = {}
    
    for image_id in range(len(target)):
        for pred_obj in pred[image_id].tolist():
            pred_obj.insert(0, image_id)
            pred_objs.append(pred_obj)
            
        for target_obj in target[image_id].tolist():
            target_obj.insert(0, n_target)
            
            num_classes[int(target_obj[-1])][0] += 1
            target_objs[image_id][int(target_obj[-1])].append(target_obj[:-1])
            
            dec_dict[n_target] = {}
            for thresh in iou_thresh:
                dec_dict[n_target][thresh] = 0
            n_target += 1    
            
    # class_aps[猫：[iou1，iou2，iou3, iou4], 狗：[ ... ]]
    class_aps = [[] for _ in range(num_class)]
    
    pred_objs = torch.tensor(pred_objs, dtype=torch.float32).to(device, non_blocking=True)
    #for循环分别计算每种类别的ap
    for cls_index in range(num_class):
        # 从pred中挑选该类别的预测框
        try:
            if pred_objs.shape[0] == 0:
                inds = torch.tensor([], dtype=torch.int64).to(device, non_blocking=True)
            else:
                inds = torch.where(pred_objs[..., -1] == cls_index)[0]
        except:
            print("pred_objs",pred_objs)
            print("pred_objs[..., -1]", pred_objs[..., -1])
            print("cls_index", cls_index)
            print("torch.where(pred_objs[..., -1] == cls_index)", torch.where(pred_objs[..., -1] == cls_index))
            print()
        #如果挑选出的预测框数量为零，但target不为零，那么TP为0，FP为零，FN为该类target的数量， ap为零
        if len(inds) == 0 and num_classes[cls_index][0] != 0:
            for i in range(len(iou_thresh)):
                class_aps[cls_index].append(0)
        #如果该类别的target数量为零，但pred不为零，那么全部的TP为零，FP为该类pred的数量，FN为0， ap为零
        elif len(inds) != 0 and num_classes[cls_index][0] == 0:
            for i in range(len(iou_thresh)):
                class_aps[cls_index].append(0)
        #如果都为零，ap为1
        elif len(inds) == 0 and num_classes[cls_index][0] == 0:
            for i in range(len(iou_thresh)):
                class_aps[cls_index].append(1)
        else:
            # 如果都不为零，按confidence降序排序（这样就省去了后期的confidence排序以及避免了iou>thresh 并且confidence更大,却不得不标为FP的情况）      
            pred_class = pred_objs[inds]
            sorte, indices = torch.sort(pred_class[...,-2], dim=0, descending=True)
            # print(sorte, indices)
            pred_class = pred_class[indices]
            
            #创建一个二维列表，为每一个pred 记录 “GT_id，confidence，iou，“（TP， FP，”precision，recall“）*len(iou_thresh) ，只需要记录TP FP即可
            record_class = [[[0, 0] for _ in range(len(iou_thresh))] for i in range(len(inds))]
            
            #for循环遍历该类别的pred,填充record_class
            n_pred = -1
            for p_class in pred_class:
                n_pred = n_pred + 1
                #若target【1】【猫】中的bbox数量为0，无法计算iou，iou为零，属于FP
                if len(target_objs[int(p_class[0])][cls_index]) == 0:
                    for i in range(len(iou_thresh)):
                        #FP
                        record_class[n_pred][i][1] = 1
                    continue
                # 计算于各个target的iou
                else:
                    iou = bbox_iou(p_class[1:5], torch.tensor(target_objs[int(p_class[0])][cls_index], dtype=torch.float32).to(device, non_blocking=True)[...,-4:])
                    max_iou, indice = torch.max(iou, dim=0)
                    # print(max_iou)
                    n_thresh = -1
                    for iou_th in iou_thresh:
                        n_thresh = n_thresh + 1
                        if max_iou > iou_th:
                            GT_id = target_objs[int(p_class[0])][cls_index][indice][0]
                            if dec_dict[GT_id][iou_th] == 0:
                                dec_dict[GT_id][iou_th] = 1
                                # TP
                                record_class[n_pred][n_thresh][0] = 1
                            else:
                                #FP
                                record_class[n_pred][n_thresh][1] = 1
                        else:
                            #FP
                            record_class[n_pred][n_thresh][1] = 1
                                
            # print(record_class)
            
            record_class = torch.tensor(record_class, dtype=torch.float32).to(device, non_blocking=True)
            #for 循环提取每个iou thresh的TP、FP，计算recall、precision
            for index_thresh in range(len(iou_thresh)):
                tp = record_class[:, index_thresh, 0]
                fp = record_class[:, index_thresh, 1]
                
                tp = torch.cumsum(tp, dim=0)
                fp = torch.cumsum(fp, dim=0)
                
                #在前面的判断条件 若target框数量为零 已处理，不会出现总target为零的情况
                rec = tp / float(num_classes[cls_index][0])
                
                #在前面的判断条件 若预测框数量为零 已处理，不会出现tp + fp=0
                #tp + fp为零，用很小的eps替代，结果还是零，只是让他不报错
                prec = tp / torch.maximum(tp + fp, torch.tensor(torch.finfo(torch.float64).eps).to(device, non_blocking=True))
                # print(prec, rec, sep="\n")
                ap = get_ap(rec.cpu().numpy(), prec.cpu().numpy(), use_07_metric=False)
                class_aps[cls_index].append(ap)
    # [猫：[iou1, iou2, ... iou10]， 狗[]]          
    class_aps = torch.tensor(class_aps, dtype=torch.float32).to(device, non_blocking=True)
    # [iou1, iou2, iou3, iou4....,iou10]    
    iou_mAP = torch.mean(class_aps, dim=0)
    # print(iou_mAP)
    return iou_mAP, n_target
    
    


# In[71]:


def get_ap(rec, prec, use_07_metric=False):
    """Compute AP given precision and recall. If use_07_metric is true, uses
    the  11-point method (default:False).
    """
    if use_07_metric:  # 使用07年方法
        # 11 个点
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])  # 插值
            ap = ap + p / 11.
    else:  # 新方式，计算所有点
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        
        # compute the precision 曲线值（也用了插值）
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


# In[ ]:




