#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


def train_collate(batch):
    """
    Custom collate fn for dealing with batches of images that have a different
    number of asssociated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets




# In[ ]:


def val_collate(batch):
    """
    Custom collate fn for dealing with batches of images that have a different
    number of asssociated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    imgs = []
    targets = []
    size = []
    scale = []
    offset = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        high = torch.FloatTensor(sample[2])
        width = torch.FloatTensor(sample[3])
        size.append([width, high, width, high])
        scale.append(torch.FloatTensor(sample[4]))
        offset.append(torch.FloatTensor(sample[5]))
        img_name.append(sample[6])
    return torch.stack(imgs, 0), targets, size, scale, offset, img_name



