"""
create_crops.py: script to crops for training and validation.
                It will save the crop images and mask in the format:
                <image_name>_<x>_<y>.<suffix>
                where x = [0, (Image_Height - crop_size) / stride]
                      y = [0, (Image_Width - crop_size) / stride]

It will create following directory structure:
    base_dir
        |   train_crops.txt   # created by script
        |   val_crops.txt     # created by script
        |
        └───crops       # created by script
        │   └───gt
        │   └───images
"""

from __future__ import print_function

import argparse
import os
import mmap
import cv2
import time
import numpy as np
from skimage import io
from tqdm import tqdm
tqdm.monitor_interval = 0



def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return False
    return True

def CreatCrops(base_dir, dataset, crop_type, size, stride, image_suffix, gt_suffix):

    crops = os.path.join(base_dir, 'crops')
    if not os.path.exists(crops):
        os.mkdir(crops)
        os.mkdir(crops+'/images')
        os.mkdir(crops+'/gt')
    crops_file = open(os.path.join(base_dir,'{}_crops.txt'.format(crop_type)),'w')

    full_file_path = os.path.join(base_dir,'{}.txt'.format(crop_type))
    full_file = open(full_file_path,'r')

    def get_num_lines(file_path):
        fp = open(file_path, "r+")
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
        return lines

    failure_images = []
    for name in tqdm(full_file, ncols=100, desc="{}_crops".format(crop_type),
                            total=get_num_lines(full_file_path)):

        name = name.strip("\n")
        if dataset == 'SpaceNet':
            image_file = os.path.join(base_dir, 'images', name)
            gt_file = os.path.join(base_dir, 'gt', name.split('n_')[1])
        elif dataset == 'DeepGlobe':
            image_file = os.path.join(base_dir, 'images', name + '_sat.jpg')
            gt_file = os.path.join(base_dir, 'gt', name + '_mask.png')

        if not verify_image(image_file):
            failure_images.append(image_file)
            continue

        image = cv2.imread(image_file)
        gt = cv2.imread(gt_file,0)

        if image is None:
            failure_images.append(image_file)
            continue

        if gt is None:
            failure_images.append(image_file)
            continue

        H,W,C = image.shape
        maxx = int((H-size)/stride)
        maxy = int((W-size)/stride)

        if dataset == 'SpaceNet':
            name = name.split('.')[0]
            name_a = name.split('n_')[1]
        elif dataset == 'DeepGlobe':
            name_a = name

        for x in range(maxx+1):
            for y in range(maxy+1):
                im_ = image[x*stride:x*stride + size,y*stride:y*stride + size,:]
                gt_ = gt[x*stride:x*stride + size,y*stride:y*stride + size]
                crops_file.write('{}_{}_{}.png\n'.format(name,x,y))
                cv2.imwrite(crops+'/images/{}_{}_{}.png'.format(name,x,y),  im_)
                cv2.imwrite(crops+'/gt/{}_{}_{}.png'.format(name_a,x,y), gt_)

    crops_file.close()
    full_file.close()
    if len(failure_images) > 0:
        print("Unable to process {} images : {}".format(len(failure_images), failure_images))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--base_dir', type=str, default='../../data/SpaceNet/spacenet/result_3m',
        help='Base directory for Spacenent Dataset.')
    parser.add_argument('--dataset', type=str, default='DeepGlobe',
                        choices=['SpaceNet', 'DeepGlobe'], help='dataset name')
    parser.add_argument('--crop_size', type=int, default=650,
        help='Crop Size of Image')
    parser.add_argument('--crop_overlap', type=int, default=0,
        help='Crop overlap Size of Image')
    parser.add_argument('--im_suffix', type=str, default='.png',
        help='Dataset specific image suffix.')
    parser.add_argument('--gt_suffix', type=str, default='.png',
        help='Dataset specific gt suffix.')

    args = parser.parse_args()

    start = time.clock()
    # Create crops for training
    CreatCrops(args.base_dir,
                args.dataset,
                crop_type='train',
                size=args.crop_size,
                stride=args.crop_size,
                image_suffix=args.im_suffix,
                gt_suffix=args.gt_suffix)

    ## Create crops for validation
    CreatCrops(args.base_dir,
                args.dataset,
                crop_type='val',
                size=args.crop_size,
                stride=args.crop_size,
                image_suffix=args.im_suffix,
                gt_suffix=args.gt_suffix)

    end = time.clock()
    print('Finished Creating crops, time {0}s'.format(end - start))

if __name__ == "__main__":
    main()
