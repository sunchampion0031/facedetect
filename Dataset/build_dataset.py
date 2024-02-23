#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import xml.etree.ElementTree as ET
import os.path as path
import cv2
import numpy as np
import torch.utils.data as data
from .transforms import ColorTransforms, ValTransforms

# In[2]:


FACE_CLASSES = ('face', 'face_mask', "mask_weared_incorrect")

# In[3]:


# 从注解中提取class， bounding box
class AnnotationTransform(object):
    """Transforms a  annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of 's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(FACE_CLASSES, range(len(FACE_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        # .iter('tag')方法 寻找所有符合要求的Element
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
                # bndbox：[xmin, ymin, xmax, ymax] 相对于原图像的比例，不是像素值
            label_idx = self.class_to_ind[name]
            # [xmin, ymin, xmax, ymax, label_ind]
            bndbox.append(label_idx)
            
            res += [bndbox]  # [[xmin, ymin, xmax, ymax, label_ind]]
            # img_id = target.find('filename').text[:-4]
        # 该张图片中有一个对象  res:[[xmin, ymin, xmax, ymax, label_ind]]
        # 该张图片中有多个对象  res:[[xmin, ymin, xmax, ymax, label_ind], ... ]
        # 位置元素都是相对于原图的比例，这样原图放缩都不影响位置。
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]




# In[4]:


# 重建自己的Dataset  继承torch.utils.data.Dataset
# 构建一个自己的dataset，需要重写魔法方法__getitem__()来指定索引访问数据的方法，同时需要重写__len__()来获取数据集的长度（数量）。
class Dataset(data.Dataset):
    """Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self,
                data_dir=None,
                img_size=640,
                image_sets=[('MAFA', 'train'), ('MAFA', 'val')],
                transform=None,
                color_augment=None,
                target_transform=AnnotationTransform(),
                mosaic=False,
                mixup=False):
        self.root = data_dir
        self.img_size = img_size
        self.image_set = image_sets
        self.target_transform = target_transform
        self._annopath = path.join('%s','%s.xml')
        self._imgpath = path.join('%s','%s.jpg')
        self.ids = list()
        for (dataset, data) in image_sets:
            rootpath = path.join(self.root, dataset, data)
            for line in open(path.join(rootpath, 'filename.txt')):
                if line.strip() != "":
                    self.ids.append((rootpath, line.strip()))
        # augmentation
        self.transform = transform(size=img_size)
        self.mosaic = mosaic
        self.mixup = mixup
        self.color_augment = color_augment(size=img_size)
        if self.mosaic:
            print('use Mosaic Augmentation ...')
        if self.mixup:
            print('use MixUp Augmentation ...')


    def __getitem__(self, index):
        im, gt, h, w, scale, offset = self.pull_item(index)
        return im, gt


    def __len__(self):
        return len(self.ids)


    def load_img_targets(self, img_id):
        # load an image

        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        # laod a target
        target = ET.parse(self._annopath % img_id).getroot()
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)


        return img, target, height, width


    def load_mosaic(self, index):
        ids_list_ = self.ids[:index] + self.ids[index+1:]
        # random sample other indexs
        id1 = self.ids[index]
        id2, id3, id4 = random.sample(ids_list_, 3)
        ids = [id1, id2, id3, id4]

        img_lists = []
        tg_lists = []
        # load image and target
        for id_ in ids:
            img_i, target_i, _, _ = self.load_img_targets(id_)
            img_lists.append(img_i)
            tg_lists.append(target_i)

        mean = np.array([v*255 for v in self.transform.mean])
        mosaic_img = np.ones([self.img_size*2, self.img_size*2, img_i.shape[2]], dtype=np.uint8) * mean
        # mosaic center
        yc, xc = [int(random.uniform(-x, 2*self.img_size + x)) for x in [-self.img_size // 2, -self.img_size // 2]]
        # yc = xc = self.img_size

        mosaic_tg = []
        for i in range(4):
            img_i, target_i = img_lists[i], tg_lists[i]
            target_i = np.array(target_i)
            h0, w0, _ = img_i.shape

            # resize
            scale_range = np.arange(50, 210, 10)
            s = np.random.choice(scale_range) / 100.

            if np.random.randint(2):
                # keep aspect ratio
                r = self.img_size / max(h0, w0)
                if r != 1:
                    img_i = cv2.resize(img_i, (int(w0 * r * s), int(h0 * r * s)))
            else:
                img_i = cv2.resize(img_i, (int(self.img_size * s), int(self.img_size * s)))
            h, w, _ = img_i.shape

            # place img in img4
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            # labels
            target_i_ = target_i.copy()
            if len(target_i) > 0:
                # a valid target, and modify it.
                target_i_[:, 0] = (w * (target_i[:, 0]) + padw)
                target_i_[:, 1] = (h * (target_i[:, 1]) + padh)
                target_i_[:, 2] = (w * (target_i[:, 2]) + padw)
                target_i_[:, 3] = (h * (target_i[:, 3]) + padh)
                # check boxes
                valid_tgt = []
                for tgt in target_i_:
                    x1, y1, x2, y2, label = tgt
                    bw, bh = x2 - x1, y2 - y1
                    if bw > 5. and bh > 5.:
                        valid_tgt.append([x1, y1, x2, y2, label])
                if len(valid_tgt) == 0:
                    valid_tgt.append([0., 0., 0., 0., 0.])

                mosaic_tg.append(target_i_)
        # check target
        if len(mosaic_tg) == 0:
            mosaic_tg = np.zeros([1, 5])
        else:
            mosaic_tg = np.concatenate(mosaic_tg, axis=0)
            # Cutout/Clip targets
            np.clip(mosaic_tg[:, :4], 0, 2 * self.img_size, out=mosaic_tg[:, :4])
            # normalize
            mosaic_tg[:, :4] /= (self.img_size * 2)

        return mosaic_img, mosaic_tg, self.img_size, self.img_size


    def pull_item(self, index):
        # load a mosaic image
        if self.mosaic and np.random.randint(2):
            # mosaic
            img, target, height, width = self.load_mosaic(index)

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if self.mixup and np.random.randint(2):
                img2, target2, height, width = self.load_mosaic(np.random.randint(0, len(self.ids)))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                target = np.concatenate((target, target2), 0)

            # augment
            img, boxes, labels, scale, offset = self.color_augment(img, target[:, :4], target[:, 4])

        # load an image and target
        else:
            img_id = self.ids[index]
            img, target, height, width = self.load_img_targets(img_id)
            if len(target) == 0:
                target = np.zeros([1, 5])
            else:
                target = np.array(target)
            # augment

            img, boxes, labels, scale, offset = self.transform(img, target[:, :4], target[:, 4])


        target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, target, height, width, scale, offset


    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR), img_id


    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt



# In[ ]:


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    pixel_mean = np.array((0.406, 0.456, 0.485), dtype=np.float32)  # BGR
    pixel_std = np.array((0.225, 0.224, 0.229),dtype=np.float32)   # BGR
    
    img_size = 448

    # dataset
    # WiderFace：整张脸
    # MAFA Face_Mask_Detection 其他：眉毛以下
    dataset = Dataset(
        data_dir="YOLO/Data",
        img_size = img_size,
        image_sets=[('Face_Mask_Detection', 'train')],
        transform=ValTransforms,
        color_augment=ValTransforms,
        mosaic=False,
        mixup=False,
        )
    
    np.random.seed(1)
    class_colors = [(np.random.randint(255),
                    np.random.randint(255),
                    np.random.randint(255)) for _ in range(len(FACE_CLASSES))]

    for i in range(len(dataset)):
        # im:[3, img_size_H img_size_W] RGB
        im, gt, h, w, scale, offset = dataset.pull_item(i)

#         print(scale, offset,sep="\n", end="\n\n")
        pixel_offsets = offset * img_size
        if h > w:
            pixel_offset = int(pixel_offsets[0][0])
            # to numpy
            image = cv2.resize(im.permute(1, 2, 0).numpy()[:,pixel_offset:img_size-pixel_offset], (w, h)).astype(np.float32)
        else:
            pixel_offset = int(pixel_offsets[0][1])
            # to numpy
            image = cv2.resize(im.permute(1, 2, 0).numpy()[pixel_offset:img_size-pixel_offset], (w, h)).astype(np.float32)

        # to BGR
        image = image[..., (2, 1, 0)]
        # denormalize
        image = (image * pixel_std + pixel_mean) * 255
        # to 
        image = image.astype(np.uint8).copy()
        
        gt[...,:4] = (gt[...,:4] - offset) / scale

        # draw bbox
        for box in gt:
            xmin, ymin, xmax, ymax, label = box
            label_id = int(label)
            color = class_colors[label_id]
            label = FACE_CLASSES[label_id]
            xmin *= w
            ymin *= h
            xmax *= w
            ymax *= h
            # 绘制方框,在原有图像基础上不断绘制方框
            image = cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            cv2.putText(image, label, (int(xmin), int(ymin - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        # 在gt窗口显示图片，窗口名称为 gt
        #image:[img_size_H img_size_W, 3] BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()
        print("输出")
        time.sleep(10)
        # cv2.imshow('target', image)
        # 参数none和0是代表无限延迟，而整数数字代表延迟多少ms，返回值是你按键的asii值
        # cv2.waitKey(0)


# In[ ]:




