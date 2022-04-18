from random import sample, shuffle

import cv2
import os
import numpy as np
import random
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

#from utils.utils import cvtColor, preprocess_input
from utils.common import cvtColor, preprocess_input, letterbox, mixup, random_affine
from utils.utils_bbox import xyxy2cxcywhab
from DOTA_devkit import dota_utils

dotav10_classes = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
                'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
                'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

dotav15_classes = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
                'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
                'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter',
                'container-crane']

def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord

def _mirror(image, boxes, prob=0.5): #the format of boexes must be xyxy
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 0::2]
    return image, boxes

def drawOneImg(img, label, save_path=False):
    pts = label[:, : -1]
    for i, poly in enumerate(pts):
        poly = poly.reshape([4,2]).astype(np.int32)
        cv2.polylines(img, [poly], isClosed=True, color=(0,0,255), thickness=2)

    if save_path:
        cv2.imwrite(save_path, img)
    else:
        cv2.namedWindow("image", 0)
        cv2.imshow("image", img)
        cv2.waitKey()

def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)
    # no return needed
    return cv2.cvtColor(img_hsv.astype(img.dtype), code=cv2.COLOR_HSV2BGR)

class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length, mosaic, train, mosaic_ratio = 0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.epoch_length       = epoch_length
        self.mosaic             = mosaic
        self.train              = train
        self.mosaic_ratio       = mosaic_ratio

        self.epoch_now          = -1
        self.length             = len(self.annotation_lines)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        if self.mosaic:
            if self.rand() < 0.5 and self.epoch_now < self.epoch_length * self.mosaic_ratio:
                lines = sample(self.annotation_lines, 3)
                lines.append(self.annotation_lines[index])
                shuffle(lines)
                image, box  = self.get_random_data_with_Mosaic(lines, self.input_shape)
            else:
                image, box  = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train)
        else:
            image, box      = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train)
        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box         = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return image, box

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line    = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(line[0])
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        return image_data, box
    
    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = [] 
        box_datas   = []
        index       = 0
        for line in annotation_line:
            #---------------------------------#
            #   每一行进行分割
            #---------------------------------#
            line_content = line.split()
            #---------------------------------#
            #   打开图片
            #---------------------------------#
            image = Image.open(line_content[0])
            image = cvtColor(image)
            
            #---------------------------------#
            #   图片的大小
            #---------------------------------#
            iw, ih = image.size
            #---------------------------------#
            #   保存框的位置
            #---------------------------------#
            box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])
            
            #---------------------------------#
            #   是否翻转图片
            #---------------------------------#
            flip = self.rand()<.5
            if flip and len(box)>0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0,2]] = iw - box[:, [2,0]]

            #------------------------------------------#
            #   对图像进行缩放并且进行长和宽的扭曲
            #------------------------------------------#
            new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale*h)
                nw = int(nh*new_ar)
            else:
                nw = int(scale*w)
                nh = int(nw/new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            #-----------------------------------------------#
            #   将图片进行放置，分别对应四张分割图片的位置
            #-----------------------------------------------#
            if index == 0:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y) - nh
            elif index == 1:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y)
            elif index == 2:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y)
            elif index == 3:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y) - nh
            
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            #---------------------------------#
            #   对box进行重新处理
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box
            
            image_datas.append(image_data)
            box_datas.append(box_data)

        #---------------------------------#
        #   将图片分割，放在一起
        #---------------------------------#
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image       = np.array(new_image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype           = new_image.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对框进行进一步的处理
        #---------------------------------#
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes

# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for (img, box) in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes


class DotaDataset(Dataset):
    def __init__(self, name, data_dir, img_size, num_classes=16, mosaic=True, mosaic_ratio = 0.7, mixup_prob=0.5):
        super(DotaDataset, self).__init__()
        self.name = name
        self.data_dir = data_dir
        self.img_size = img_size
        self.labels_dir = os.path.join(data_dir, name, 'labelTxt')
        self.imgs_dir = os.path.join(data_dir, name, 'images')
        self.num_classes = num_classes
        self.mosaic = mosaic
        self.labels_file = [files for root, dirs, files in os.walk(self.labels_dir)]
        self.labels_file = [os.path.join(self.labels_dir, file) for file in self.labels_file[0]]
        self.imgs_file = [file.replace('labelTxt', 'images').replace('.txt', '.png') for file in self.labels_file]
        assert len(self.labels_file) == len(self.imgs_file)
        self.imgs_num = len(self.imgs_file)
        self.class_id = {}
        for i, cls in enumerate(dotav15_classes):
            self.class_id[cls] = i

        self.ids = [i for i in range(len(self.labels_file))]
        random.shuffle(self.ids)
        self.mosaic_ratio = mosaic_ratio
        self.mixup_prob = mixup_prob
        self.epoch_now = -1
        self.degrees=0.0
        self.scale=(0.5, 0.5)
        self.shear = 2.0
        self.translate = 0.1

    def __len__(self):
        return self.imgs_num

    def load_image(self, index):
        return cv2.imread(self.imgs_file[index])

    def load_anno(self, index):
        ann_file = self.labels_file[index]
        objects = dota_utils.parse_dota_poly2(ann_file)
        targets = []
        for obj in objects:
            class_id = self.class_id[obj['name']]
            poly = obj['poly']
            targets.append(poly + [class_id])
        return np.array([targets])

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def pull_item(self, index):
        img = self.load_image(index)

        ann_file = self.labels_file[index]
        objects = dota_utils.parse_dota_poly2(ann_file)
        targets = []
        for obj in objects:
            class_id = self.class_id[obj['name']]
            poly = obj['poly']
            targets.append(poly + [class_id])
        res = np.array(targets)

        # self.draw(img, res)
        return img, res.copy()

    def mixup(self, origin_img, origin_labels):
        r = np.random.beta(32.0, 32.0)
        HFLIP = random.uniform(0, 1) > 0.5
        VFLIP = random.uniform(0, 1) > 0.5
        cp_index = random.randint(0, self.__len__() - 1)
        cp_img, cp_labels = self.pull_item(cp_index)
        if cp_img.shape != origin_img.shape:
            cp_img, scale, (padw, padh) = letterbox(cp_img)
            cp_labels[:, 0:-1:2] = scale[1] * cp_labels[:, 0:-1:2]
            cp_labels[:, 1:-1:2] = scale[0] * cp_labels[:, 1:-1:2]
            cp_labels[:, 0:-1:2] += int(padw)
            cp_labels[:, 1:-1:2] += int(padh)
            # drawOneImg(cp_img, cp_labels)
        width, height = cp_img.shape[0], cp_img.shape[1]

        if HFLIP:  # horizontal flip
            cp_img = cp_img[:, ::-1, :].copy()
            cp_labels[:, 1:-1:2] = width - cp_labels[:, 1:-1:2]
        if VFLIP:  # vertical flip
            cp_img = cp_img[::-1, :, :].copy()
            cp_labels[:, 2:-1:2] = height - cp_labels[:, 2:-1:2]

        img = (origin_img * r + (1 - r) * cp_img).astype(np.uint8)
        labels = np.concatenate((origin_labels, cp_labels), 0)
        # draw(img, labels, origin_img, origin_labels,cp_img, cp_labels)
        return img, labels

    def __getitem__(self, index):
        if random.random() < self.mosaic_ratio:
            mosaic_labels = []
            input_h, input_w = self.img_size[0], self.img_size[1]

            yc = int(random.uniform(0.75 * input_h, 1.25 * input_h))
            xc = int(random.uniform(0.75 * input_w, 1.25 * input_w))

            indices = [index] + [random.randint(0, self.imgs_num - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                img, _labels = self.pull_item(index)
                h0, w0 = img.shape[:2]
                scale = min(1. * input_h / h0, 1. * input_w / w0)

                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )

                (h,w,c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )
                mosaic_img[l_y1:l_y2, l_x1: l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1
                labels = _labels.copy()
                if _labels.size > 0:
                    labels[:, [0,2,4,6]] = scale * labels[:, [0,2,4,6]] + padw
                    labels[:, [1,3,5,7]] = scale * labels[:, [1,3,5,7]] + padh
                mosaic_labels.append(labels)
            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)

            mosaic_img, mosaic_labels = random_affine(
                mosaic_img,
                mosaic_labels,
                target_size=(input_w, input_h),
                degrees=self.degrees,
                translate=self.translate,
                scales=self.scale,
                shear=self.shear,
                oriented=True
            )
            if random.random() < self.mixup_prob:
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels)
            drawOneImg(mosaic_img, mosaic_labels, save_path=f"../draw/{index}_mosaic.png")
            return mosaic_img, mosaic_labels

        else:
            img, labels = self.pull_item(index)
            img, scale, (padw, padh) = letterbox(img)
            if len(labels) > 0:
                labels[:, 0:-1:2] = scale[1] * labels[:, 0:-1:2]
                labels[:, 1:-1:2] = scale[0] * labels[:, 1:-1:2]
                labels[:, 0:-1:2] += int(padw)
                labels[:, 1:-1:2] += int(padh)
            return img, labels

if __name__ == '__main__':
    train_dataset = DotaDataset(name='train', data_dir='/home/yanggang/data/DOTA_SPLIT', img_size=(1024,1024))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=yolo_dataset_collate, drop_last=True)
    for i, (imgs, targets) in enumerate(train_dataloader):
        print(f"batch {i} : {imgs.shape}")

