#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from Dataset.build_dataset import FACE_CLASSES

# In[ ]:


# FACE_CLASSES = ('without_mask', 'with_mask', "mask_weared_incorrect")

# class_colors = [(np.random.randint(255),
#                  np.random.randint(255),
#                  np.random.randint(255)) for _ in range(len(FACE_CLASSES))]


class_colors = [(0, 0, 255), (0,255,0), (0, 255, 255)]


# In[2]:


# 制作正样本之前的可视化
# 分数越高字体越粗，颜色越红
def vis_prediction(images, targets, groundT):
    """
        images: (tensor) [B, 3, H, W]
        targets: (list) a list of targets
    """
    batch_size = images.size(0)
    # vis data
    bgr_mean=np.array((0.406, 0.456, 0.485), dtype=np.float32)
    bgr_std=np.array((0.225, 0.224, 0.229), dtype=np.float32)

    for bi in range(batch_size):
        # to numpy
        image = images[bi].permute(1, 2, 0).cpu().numpy()
        # to BGR
        image = image[..., (2, 1, 0)]
        # denormalize
        image = ((image * bgr_std + bgr_mean)*255).astype(np.uint8)
        image = image.copy()
        img_h, img_w = image.shape[:2]

        targets_i = targets[bi]
        for target in targets_i:
            x1, y1, x2, y2 = target[:4]
            s = target[4].item()
            score = round(s, 2)
            label = target[-1]
            x1 = int(x1 * img_w)
            y1 = int(y1 * img_h)
            x2 = int(x2 * img_w)
            y2 = int(y2 * img_h)
            color = class_colors[int(label)]
            label = FACE_CLASSES[int(label)]
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            message ="P:" + label+" "+str(score)

            t_size, bottom = cv2.getTextSize(message, 0, fontScale=1, thickness=1)
            # thickness 参数表示矩形边框的厚度，如果为负值，如 CV_FILLED，则表示填充整个矩形。
            image = cv2.rectangle(image, (x1, int(y1 - (t_size[1] + bottom)*((x2-x1)/t_size[0]))), (int(x2), y1), color, -1)

            cv2.putText(image, message, (x1, int(y1 - 2)), 0, (x2-x1)/t_size[0], (0, 0, 0), 1, lineType=cv2.LINE_AA)
        if groundT != []:
            targets_i = groundT[bi]
            for target in targets_i:
                x1, y1, x2, y2 = target[:4]
                label = target[-1]
                x1 = int(x1 * img_w)
                y1 = int(y1 * img_h)
                x2 = int(x2 * img_w)
                y2 = int(y2 * img_h)
                color = class_colors[int(label)]
                label = FACE_CLASSES[int(label)]
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                message = "G:" + label
                cv2.putText(image, message, (x1, int(y1 - 5)), 0, 0.5, (255, 0, 0),
                            1, lineType=cv2.LINE_AA)

        cv2.imshow('prediction', image)
        k = cv2.waitKey(0)  # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待
        if k == 27:  # 键盘上esc键的键值
            cv2.destroyAllWindows()
            break
        if bi == batch_size - 1:
            cv2.destroyAllWindows()
            break


# 制作正样本之前的可视化
def vis_data(images, targets):
    """
        images: (tensor) [B, 3, H, W]
        targets: (list) a list of targets
    """
    batch_size = images.size(0)
    # vis data
    bgr_mean = np.array((0.406, 0.456, 0.485), dtype=np.float32)
    bgr_std = np.array((0.225, 0.224, 0.229), dtype=np.float32)

    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(len(FACE_CLASSES))]

    for bi in range(batch_size):
        # to numpy
        image = images[bi].permute(1, 2, 0).cpu().numpy()
        # to BGR
        image = image[..., (2, 1, 0)]
        # denormalize
        image = ((image * bgr_std + bgr_mean) * 255).astype(np.uint8)
        image = image.copy()
        img_h, img_w = image.shape[:2]

        targets_i = targets[bi]
        for target in targets_i:
            x1, y1, x2, y2 = target[:4]
            label = target[-1]
            x1 = int(x1 * img_w)
            y1 = int(y1 * img_h)
            x2 = int(x2 * img_w)
            y2 = int(y2 * img_h)
            color = class_colors[int(label)]
            label = FACE_CLASSES[int(label)]
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image,label,(x1, y1 - 5), 0, 0.5, color, 2, lineType=cv2.LINE_AA)

        cv2.imshow('groundtruth', image)
        k = cv2.waitKey(0)  # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待
        if k == 27:  # 键盘上esc键的键值
            cv2.destroyAllWindows()
            break
        if bi == batch_size-1:
            cv2.destroyAllWindows()
            break


# In[ ]:


# 制作正样本之后的可视化
def vis_targets(images, targets, anchor_sizes=None, strides=[8, 16, 32]):
    """
        images: (tensor) [B, 3, H, W]
        targets: (tensor) [B, HW*KA, 1+1+4+1]
        anchor_sizes: (List) 
        strides: (List[Int]) output stride of network
    """
    batch_size = images.size(0)
    KA = len(anchor_sizes) // len(strides) if anchor_sizes is not None else 1
    # vis data
    rgb_mean=np.array((0.485, 0.456, 0.406), dtype=np.float32)
    rgb_std=np.array((0.229, 0.224, 0.225), dtype=np.float32)
    
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(len(FACE_CLASSES))]

    for bi in range(batch_size):
        # to numpy
        image = images[bi].permute(1, 2, 0).cpu().numpy()
        # denormalize
        image = ((image * rgb_std + rgb_mean)*255).astype(np.uint8)
        # to BGR
        image = image[..., (2, 1, 0)]
        image = image.copy()
        img_h, img_w = image.shape[:2]

        target_i = targets[bi] # [HW*KA, 1+1+4+1]
        N = 0
        for si, s in enumerate(strides):
            fmp_h, fmp_w = img_h // s, img_w // s
            HWKA = fmp_h * fmp_w * KA
            targets_i_s = target_i[N:N+HWKA]
            N += HWKA
            # [HW*KA, 1+1+4+1] -> [H, W, KA, 1+1+4+1]
            targets_i_s = targets_i_s.reshape(fmp_h, fmp_w, KA, -1)
            for j in range(fmp_h):
                for i in range(fmp_w):
                    for k in range(KA):
                        target = targets_i_s[j, i, k] # [1+1+4+1,]
                        # 画出bbox
                        if target[0] > 0.:
                            # gt box
                            box = target[1:6]
                            label, x1, y1, x2, y2 = box
                            # denormalize bbox
                            x1 = int(x1 * img_w)
                            y1 = int(y1 * img_h)
                            x2 = int(x2 * img_w)
                            y2 = int(y2 * img_h)
                            color = class_colors[int(label)]
                            label = FACE_CLASSES[int(label)]
                            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
                            # 画出匹配该bbox的锚框
                            if anchor_sizes is not None:
                                # anchor box
                                anchor_size = anchor_sizes[si*KA + k]
                                x_anchor = (i) * s
                                y_anchor = (j) * s
                                w_anchor, h_anchor = anchor_size
                                anchor_box = [x_anchor, y_anchor, w_anchor, h_anchor]
                                print('stride: {} - anchor box: ({}, {}, {}, {})'.format(s, *anchor_box))
                                x1_a = int(x_anchor - w_anchor * 0.5)
                                y1_a = int(y_anchor - h_anchor * 0.5)
                                x2_a = int(x_anchor + w_anchor * 0.5)
                                y2_a = int(y_anchor + h_anchor * 0.5)
                                cv2.rectangle(image, (x1_a, y1_a), (x2_a, y2_a), (255, 0, 0), 2)
                            # 画出中心点
                            else:
                                x_anchor = (i) * s
                                y_anchor = (j) * s
                                anchor_point = (x_anchor, y_anchor)
                                print('stride: {} - anchor point: ({}, {})'.format(s, *anchor_point))
                                cv2.circle(image, anchor_point, 10, (255, 0, 0), -1)

        cv2.imshow('assignment', image)
        k = cv2.waitKey(0)  # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待
        if k == 27:  # 键盘上esc键的键值
            cv2.destroyAllWindows()
            break
        if bi == batch_size-1:
            cv2.destroyAllWindows()
            break

