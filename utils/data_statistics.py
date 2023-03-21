import os
import cv2
import numpy as np
from pymic.io.image_read_write import *
from os import makedirs
import shutil

def data_intensity(dir_img):
    filenames = os.listdir(dir_img)
    for filename in filenames:
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = cv2.imread(os.path.join(dir_img, filename))
            print(filename, image.min(), image.max())

def data_intensity_norm(dir_img, dir_gt, dir_img_save=None, dir_gt_save=None):
    if (dir_img_save is not None) and (dir_gt_save is not None):
        makedirs(dir_img_save)
        makedirs(dir_gt_save)

    filenames = os.listdir(dir_img)
    vals_mean = []
    for filename in filenames:
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = cv2.imread(os.path.join(dir_img, filename))
            # image = (image - image.min()) / (image.max() - image.min())
            # cv2.imwrite(os.path.join(dir_img_save, filename), image)
            
            # gt = cv2.imread(os.path.join(dir_gt, filename))
            # gt = gt / 255
            # gt = gt.astype('uint8')
            # cv2.imwrite(os.path.join(dir_img_save, filename), gt)
            # shutil.copy(os.path.join(dir_gt, filename), os.path.join(dir_gt_save, filename))
            vals_mean.append(image.mean())
    
    vals_mean = np.array(vals_mean)
    print(stage, vals_mean.mean(), vals_mean.std())


# if __name__ == '__main__':
#     stages = ['train', 'valid', 'test']
#     root_dir = "/home/data/FJ/Code/WSL4fetal/raw_data/cls/slice_resample_jpg"
#     save_dir = "/home/data/FJ/Code/WSL4fetal/raw_data/cls/slice_resample_jpg_mean"
    
#     for stage in stages:
#         dir_img = root_dir + "/image/" + stage
#         dir_gt = root_dir + "/brain/" + stage
#         dir_img_save = save_dir + "/image/" + stage
#         dir_gt_save = save_dir + "/brain/" + stage

#         data_intensity_norm(dir_img, dir_gt, dir_img_save, dir_gt_save)

if __name__ == '__main__':
    stage = 'train'
    root_dir = "/home/data/FJ/Dataset/Fetal_brain/slice_png"       # slice_resample_png
    dir_img = root_dir + "/image/" + stage
    dir_gt = root_dir + "/brain/" + stage

    # data_intensity(dir_img)
    data_intensity_norm(dir_img, dir_gt)

