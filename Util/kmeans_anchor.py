#!/usr/bin/env python
# coding: utf-8

# In[1]:


import import_ipynb
import numpy as np
import random
import argparse
import os
import sys
sys.path.append('..')

from data_process.build_dataset import Dataset



# In[2]:


def parse_args():
    parser = argparse.ArgumentParser(description='kmeans for anchor box')
    parser.add_argument('--data_dir', default='../../../data',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='archive',
                        help='face')
    parser.add_argument('-na', '--num_anchorbox', default=6, type=int,
                        help='number of anchor box.')
    parser.add_argument('-size', '--img_size', default=448, type=int,
                        help='input size.')
    return parser.parse_known_args()[0]
                    


# In[3]:


class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h




# In[4]:


def iou(box1, box2):
    x1, y1, w1, h1 = box1.x, box1.y, box1.w, box1.h
    x2, y2, w2, h2 = box2.x, box2.y, box2.w, box2.h

    S_1 = w1 * h1
    S_2 = w2 * h2

    xmin_1, ymin_1 = x1 - w1 / 2, y1 - h1 / 2
    xmax_1, ymax_1 = x1 + w1 / 2, y1 + h1 / 2
    xmin_2, ymin_2 = x2 - w2 / 2, y2 - h2 / 2
    xmax_2, ymax_2 = x2 + w2 / 2, y2 + h2 / 2

    I_w = min(xmax_1, xmax_2) - max(xmin_1, xmin_2)
    I_h = min(ymax_1, ymax_2) - max(ymin_1, ymin_2)
    if I_w < 0 or I_h < 0:
        return 0
    I = I_w * I_h

    IoU = I / (S_1 + S_2 - I)

    return IoU




# In[5]:


def init_centroids(boxes, n_anchors):
    """
        We use kmeans++ to initialize centroids.
    """
    centroids = []
    boxes_num = len(boxes)

    # 从【0，boxes_num】选出一个数， 构建第一个centroids
    centroid_index = int(np.random.choice(boxes_num, 1)[0])
    centroids.append(boxes[centroid_index])
#     print(centroids[0].w,centroids[0].h)

    # 构建其余的n_anchors-1个centroids
    for centroid_index in range(0, n_anchors-1):
        # sum_distance 保存每个box到centroid distance中的最小的distance 之和
        sum_distance = 0
        # distance_thresh 距离阈值，第一个cur_sum大于该阈值的box作为新的centroid
        distance_thresh = 0
        # distance_list保存，box到所有的centroid distance中的最小的distance
        distance_list = []
        # cur_sum保存前n个box到centroid的最小distance之和
        cur_sum = 0

        for box in boxes:
            # 1 - iou为两框的距离，distance最大为1
            min_distance = 1
            for centroid_i, centroid in enumerate(centroids):
                distance = (1 - iou(box, centroid))
                if distance < min_distance:
                    min_distance = distance
            sum_distance += min_distance
            distance_list.append(min_distance)

        distance_thresh = sum_distance * np.random.random()

        # 遍历所有的box， 不会有重复的box作为centroid，因为distance为0
        for i in range(0, boxes_num):
            cur_sum += distance_list[i]
            if cur_sum > distance_thresh:
                centroids.append(boxes[i])
#                 print(boxes[i].w, boxes[i].h)
                break
    return centroids




# In[6]:


def do_kmeans(n_anchors, boxes, centroids):
    loss = 0
    groups = []
    new_centroids = []
    # for box in centroids:
    #     print('box: ', box.x, box.y, box.w, box.h)
    # exit()
    for i in range(n_anchors):
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))
    
    for box in boxes:
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):
            distance = (1 - iou(box, centroid))
            if distance < min_distance:
                min_distance = distance
                group_index = centroid_index
        groups[group_index].append(box)
        loss += min_distance
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h

    for i in range(n_anchors):
        new_centroids[i].w /= max(len(groups[i]), 1)
        new_centroids[i].h /= max(len(groups[i]), 1)

    return new_centroids, groups, loss# / len(boxes)




# In[7]:


def anchor_box_kmeans(total_gt_boxes, n_anchors, loss_convergence, iters, plus=True):
    """
        This function will use k-means to get appropriate anchor boxes for train dataset.
        Input:
            total_gt_boxes: 
            n_anchor : int -> the number of anchor boxes.
            loss_convergence : float -> threshold of iterating convergence.
            iters: int -> the number of iterations for training kmeans.
        Output: anchor_boxes : list -> [[w1, h1], [w2, h2], ..., [wn, hn]].
    """
    boxes = total_gt_boxes
    centroids = []
    if plus:
        centroids = init_centroids(boxes, n_anchors)
    else:
        total_indexs = range(len(boxes))
        sample_indexs = random.sample(total_indexs, n_anchors)
        for i in sample_indexs:
            centroids.append(boxes[i])

    # iterate k-means
    centroids, groups, old_loss = do_kmeans(n_anchors, boxes, centroids)
    iterations = 1
    while(True):
        centroids, groups, loss = do_kmeans(n_anchors, boxes, centroids)
        iterations += 1
        print("Loss = %f" % loss)
        print("Old_loss - Loss = %f" % (old_loss - loss))
        for centroid in centroids:
            print("[%.2f, %.2f]" %(centroid.w, centroid.h), end='&&')
        print("\n\n")
        
        if abs(old_loss - loss) < loss_convergence or iterations > iters:
            break
        old_loss = loss

        
    
    
#     print("k-means result : ") 
    areas = []
    for i in range(n_anchors):
        centroids[i] = [round(centroids[i].w, 2), round(centroids[i].h, 2)]
        areas.append(round(centroids[i][0] * centroids[i][1], 2))
    areas = np.array(areas)
    centroids = np.array(centroids)
    
    index_sort = np.argsort(areas)
    areas = areas[index_sort]
    centroids = centroids[index_sort]
    
    return centroids, areas




# In[10]:


def kmeans_anchor(num_anchorbox, img_size, data_path, data_set):    
    loss_convergence = 1e-6
    iters_n = 1000
    
    boxes = []

    dataset = Dataset(
        data_dir = data_path,
        img_size = img_size,
        image_sets=[(data_set, 'train'), (data_set, 'val')],
    )

    # FACE
    for i in range(len(dataset)):
        if i % 500 == 0:
            print('Loading FACE data [%d / %d]' % (i+1, len(dataset)))

        # For FACE
        img, _ = dataset.pull_image(i)
        w, h = img.shape[1], img.shape[0]
        _, annotation = dataset.pull_anno(i)

        # prepare bbox datas
        for box_and_label in annotation:
            box = box_and_label[:-1]
            xmin, ymin, xmax, ymax = box
            bw = (xmax - xmin) / max(w, h) * img_size
            bh = (ymax - ymin) / max(w, h) * img_size
            # check bbox
            if bw < 1.0 or bh < 1.0:
                continue
            boxes.append(Box(0, 0, bw, bh))


    print("Number of all bboxes: ", len(boxes))
    print("Start k-means !")
    centroids, areas = anchor_box_kmeans(boxes, num_anchorbox, loss_convergence, iters_n, plus=True)
    print("Result:")
    print(centroids, areas, sep="\n")
    return centroids


# In[ ]:

kmeans_anchor(6, 448, '../../../data', "archive")



# In[11]:


# centroids =  [[  5.21,   6.97],
#  [ 10.75,  14.21],
#  [ 20.01,  25.78],
#  [ 32.47,  44.75],
#  [ 51.65,  69.79],
#  [ 77.75, 103.28],
#  [113.56, 150.48],
#  [158.43, 211.24],
#  [223.33, 287.21]]

# areas = [3.63137000e+01, 1.52757500e+02, 5.15857800e+02, 1.45303250e+03,
#  3.60465350e+03, 8.03002000e+03, 1.70885088e+04, 3.34667532e+04,
#  6.41426093e+04]


# In[12]:


# kmeans_anchor(9, 448, '../../../data', "face")


# In[ ]:




