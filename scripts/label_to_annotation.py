import pandas as pd
from tqdm import tqdm
import sys
import os
import copy
import random
import math
import itertools
import pickle
import ast, csv
import datetime
# from datetime import datetime
from dateutil.relativedelta import relativedelta
from functools import partial
import warnings
import numpy as np
import random
from shapely.geometry import Polygon
from skimage import measure
from shapely.validation import make_valid
from skimage.io import imread, imsave
from skimage.io import imread
from glob import glob
from time import gmtime, strftime
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import skimage
import torch
import mmcv
import cv2
import json
import shutil
from skimage import measure
from shapely import geometry
from shapely.validation import make_valid
# sorted(namer.keys())
import matplotlib
from scipy import stats # to compute pearsons correaltion coefficeint 
import xml.etree.ElementTree as ET
import seaborn as sns
# matplotlib.use('Agg')



def systematic_split(root_dir,split_ratio= [0.7, 0.2], val_test_split=True):
    imgs = sorted(glob(root_dir + '/images' + '/*.tif'))
    N = len(imgs)
    
    tr = round(N*split_ratio[0])
    tr_ts = round(N*split_ratio[1])
    vest_ind = list(range(0, N, round(N/tr_ts)))

    
    val_ts_im = [imgs[i] for i in vest_ind]
    tr_im = list(set(imgs).symmetric_difference(set(val_ts_im)))
    
    if val_test_split:
        val = [val_ts_im[i] for i in range(0, len(val_ts_im), 2)]
        ts = [val_ts_im[i] for i in range(1, len(val_ts_im), 2)]
        
        return tr_im, val, ts
    else:
        return tr_im, val_ts_im
    
    
def scale_8bit(data, cmin=None, cmax=None, high=255, low=0):
    """
    Converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255). If the input image already has 
    dtype uint8, no scaling is done.
    :param data: 16-bit image data array
    :param cmin: bias scaling of small values (def: data.min())
    :param cmax: bias scaling of large values (def: data.max())
    :param high: scale max value to high. (def: 255)
    :param low: scale min value to low. (def: 0)
    :return: 8-bit image data array
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        high = 255
    if low < 0:
        low = 0
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)

def std_stretch_data(data, n=2.5):
    """Applies an n-standard deviation stretch to data."""

    mean, d = data.mean(), data.std() * n
    new_min = math.floor(max(mean - d, data.min()))
    new_max = math.ceil(min(mean + d, data.max()))
    
    data = np.clip(data, new_min, new_max)
    data = (data - data.min()) / (new_max - new_min)
    data = data*255
    return data.astype(np.uint8)

def std_stretch_all(img, std = 2.5, chanell_order = "last"):
    if chanell_order == "first":
        stacked = np.dstack((std_stretch_data(img[2,:,:], std), std_stretch_data(img[1,:,:], std),std_stretch_data(img[0, :,:], std)))
    else:
        stacked =  np.dstack((std_stretch_data(img[:,:,2], std), std_stretch_data(img[:,:,1], std),std_stretch_data(img[:,:,0], std)))
    return stacked

def read_images(im, scale=True, stretch=True):
    img = imread(im)
    if scale:
        img = scale_8bit(img)
    if stretch:
        img = std_stretch_all(img)
    return img
    
def read_labels(IM):
    Im = imread(IM)
    Im = Im.astype(np.uint8)
    return Im

def t_diff(a, b):
    t_diff = relativedelta(b, a) 
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)

def find_area(contr):
    '''computes the areaa of each countour'''
    c = np.expand_dims(contr.astype(np.float32), 1)
    c = cv2.UMat(c)
    area = cv2.contourArea(c)
    return area

def file_to_dict(label):
    '''
    files: image and label path as tuple
    treshold: number of pixels to be considered as object
    '''
 
    MASK = read_labels(label)

    assert len(MASK.shape)==2, 'shape of the mask should be two dimensional. Please check it'

    contours = measure.find_contours(MASK, 0.5)
    
    data = []
   
    if len(contours)>=1:
        for cont in contours:
            for i in range(len(cont)):
                row, col = cont[i]
                cont[i] = (col, row)
            if len(cont)<4: # invalid geometries
                continue
            poly = Polygon(cont)
            poly = poly.simplify(1.0, preserve_topology=False)
            if poly.is_empty:
                continue
            if poly.geom_type == 'MultiPolygon':
                for spoly in poly:
                    if spoly.is_empty:
                        continue
                    if not spoly.is_valid:
                        continue
                    min_x, min_y, max_x, max_y = spoly.bounds
                    width = max_x - min_x
                    height = max_y - min_y
                    # bbox = (min_x, min_y, width, height)
                    area = spoly.area
                    seg = np.array(spoly.exterior.coords).ravel().tolist()
                  
                    data_anno = dict(image_id = None,
                             id = None,
                             category_id = 0,
                             bbox = [min_x, min_y, width, height],
                             area = area,
                             segmentation = [seg],
                             iscrowd = 0)
                    data.append(data_anno)
            else:
                if not poly.is_valid:
                    continue
                min_x, min_y, max_x, max_y = poly.bounds
                width = max_x - min_x
                height = max_y - min_y
                bbox = (min_x, min_y, width, height)
                area = poly.area
                seg = np.array(poly.exterior.coords).ravel().tolist()

                data_anno = dict(image_id = None,
                                 id = None,
                                 category_id = 1,
                                 bbox = [min_x, min_y, width, height],
                                 area = area,
                                 segmentation = [seg],
                                 iscrowd = 0)
                data.append(data_anno)
    else:
        data = []
        
    return data


def dict2coco(files, out_file_name, return_valid=True, w=300, h=300):
    annotations = []
    images = []
    idx = 0
    count = 0
    valid_images = []
    for pairs in files:
        image = pairs[0]
        mask = pairs[1]
        
        annots = file_to_dict(mask)
        num_obj = len(annots)
        
        if num_obj == 0 or None in annots:
            pass
        else:
            # read and save image s jpg
            # save image ddict to images fold
            img = read_images(image)
            name = os.path.split(image)[1].replace('.tif', '.jpg')
            width = img.shape[0]
            height = img.shape[1]
            
            if return_valid:
                valid_images.append(image)
            
            image_dict = dict(license=0,
                              url = None,
                              file_name = name,
                              height = height,
                              width = width,
                              date_captured = None,
                              id = idx)
            
            images.append(image_dict)

            for obj in annots:
                obj['image_id'] = idx
                obj['id'] = count
                annotations.append(obj)
                count+=1
            idx+=1
    
    now = datetime.datetime.now()
    
    information=dict(
            description='refugee camp data segmentation',
            url='University of salzburg',
            version='1.0',
            year=now.year,
            contributor='GeoHUm',
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"))
        
    right = dict(url=None, id=0, name=None)
        
    coco_json = dict(info = information,
                     licenses = right,
                     images = images,
                     annotations=annotations,
                     categories =[{'supercategory':'camp', 'id':0, 'name': 'background'},{'supercategory':'camp', 'id':1, 'name': 'dwelling'}]) # {'supercategory':'camp', 'id':0, 'name': 'background'},
    
    mmcv.dump(coco_json, out_file_name)
    
    if return_valid:
        return valid_images
    
    

    
def file_to_dict_fromPaskal(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    f_name = root[0].text # file name
    
    root_ = root[3:]
    data = []
    if len(root_)>=1:
        for i in range(len(root_)):
            xmin = float(root_[i][1][0].text) # xmin
            ymin = float(root_[i][1][1].text) # ymin
            xmax = float(root_[i][1][2].text) # xmax
            ymax = float(root_[i][1][3].text) # ymax
            width = xmax - xmin
            height = ymax - ymin
            area = 0.5*height*width  # just a note
        
            data_anno = dict(image_id = None,
                         id = None,
                         category_id = 1,
                         bbox = [xmin, ymin, width, height],
                         area = area,
                         segmentation = [None],
                         iscrowd = 0)
            data.append(data_anno)
    return data, f_name.replace('.tif', '.jpg')

def dict2coco_fromPascal(files, out_file_name, w=300, h=300):
    annotations = []
    images = []
    idx = 0
    count = 0
    for file_ in files:        
        annots, name = file_to_dict_fromPaskal(file_)
        num_obj = len(annots)
        image_dict = dict(license=0,
                              url = None,
                              file_name = name,
                              height = h,
                              width = w,
                              date_captured = None,
                              id = idx)
        images.append(image_dict)
        for obj in annots:
            obj['image_id'] = idx
            obj['id'] = count
            annotations.append(obj)
            count+=1
        idx+=1
    
    now = datetime.datetime.now()
    
    information=dict(
            description='refugee camp data segmentation',
            url='University of salzburg',
            version='1.0',
            year=now.year,
            contributor='GeoHUm',
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"))
        
    right = dict(url=None, id=0, name=None)
        
    coco_json = dict(info = information,
                     licenses = right,
                     images = images,
                     annotations=annotations,
                     categories =[{'supercategory':'camp', 'id':0, 'name': 'background'},{'supercategory':'camp', 'id':1, 'name': 'dwelling'}]) # {'supercategory':'camp', 'id':0, 'name': 'background'},
    
    mmcv.dump(coco_json, out_file_name)
    print('Processing done!')
    
    

def geotif2jpg(file, dst_dir=None):
    if dst_dir is None:
        raise ValueError('Destination folder can not be "None"')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)

    if len(file) == 1:
        ext = os.path.splitext(file)[1]
    else:
        ext = os.path.splitext(file[0])[1]
        
    if ext in ['jpg', 'jpeg', 'JPEG', 'png']:
        print(f'image extension {ext} is incompressed format')
    
    if len(file) == 1:
        name = os.path.split(file)[1].replace(ext, '.jpg')
        path = os.path.join(dst_dir, name)
        arr = read_images(file)
        imsave(path, arr)
    elif len(file)>1:
        for i in range(len(file)):
            name = os.path.split(file[i])[1].replace(ext, '.jpg')
            path = os.path.join(dst_dir, name)
            arr = read_images(file[i])
            imsave(path, arr)
        print(f'Done! All files saved at {dst_dir}')
        
        
class RandomParser:
    '''Samples and prepares test data for evaluation'''
    def __init__(self, root_dir, out_dir=None, ratio=[0.7, 0.15, 0.15]):
        self.root_dir = root_dir
        self.out_dir = out_dir
        self.ratio=ratio
        
        imfiles = self.root_dir + '/images' # shuffle
        imfiles = glob(imfiles + '/*.tif')
        
        N = len(imfiles)
        trn = int(round(N*ratio[0]))
        vrn = int(round(N*ratio[1]))
        tsrn = int(round(N*ratio[2]))

        tr_im_files = imfiles[:trn]
        tr_lb_files = [im.replace('images', 'labels') for im in tr_im_files]
        self.train_data = list(zip(tr_im_files, tr_lb_files))

        if vrn >0:
            v_im_files = imfiles[trn:trn+vrn]
            v_lb_files = [im.replace('images', 'labels') for im in v_im_files]
            self.valid_data = list(zip(v_im_files, v_lb_files))
        
        if tsrn>0:
            ts_im_files = imfiles[trn+vrn:]
            ts_lb_files = [im.replace('images', 'labels') for im in ts_im_files]
            self.test_data = list(zip(ts_im_files, ts_lb_files))
            
    def report(self):
        print(f'Train data length: {len(self.train_data)}')
        print(f'Valid data length: {len(self.valid_data)}')
        print(f'Test data length: {len(self.test_data)}')

    def paus(self, part='train'):
        assert part in ['train', 'valid', 'test'], 'cat must be either of "train", "valid","test"'
        if part == 'train':
            np.random.shuffle(self.train_data)
        elif part == 'valid':
            np.random.shuffle(self.test_data)
        elif part == 'test':
            np.random.shuffle(self.valid_data)
        else:
            raise ValueError(f'part type {part} not known')
            
    def prepare2coco(self,partition='train', shuffle=True):
        assert partition in ['train', 'valid', 'test'], 'partiton not known'
        
        if self.out_dir is None:
            raise ValueError('Path to save the imges is not known')
        save_dir = os.path.join(self.out_dir, partition)
        
        jason_file = os.path.join(save_dir, partition +'_annotations.json')
        
        if shuffle:
            self.paus(part=partition) # randomize the samples
            
        if partition=='train':
            files = self.train_data
        elif partition == 'valid':
            files = self.valid_data
        elif partition == 'test':
            files = self.test_data
        
        values = dict2coco(files=files, out_file_name=jason_file, return_valid=True)
        
        geotif2jpg(file=values, dst_dir=save_dir)
        
class SystematicParser:
    '''Samples and prepares training data for evaluation'''
    def __init__(self, root_dir, out_dir=None, ratio=[0.7, 0.2]):
        self.root_dir = root_dir
        self.out_dir = out_dir
        self.ratio=ratio
        
        trim, val, tst = systematic_split(root_dir=self.root,
                                          split_ratio= self.ratio, 
                                          val_test_split=True)
        trlb = [im.replace('images', 'labels') for im in trim]
        val_lb = [im.replace('images', 'labels') for im in val]
        tst_lb = [im.replace('images', 'labels') for im in tst]
        
        imfiles = os.path.join(self.root_dir, 'images') # shuffle
        imfiles = glob(imfiles + '\*.tif')
        
        self.train_data = list(zip(trim, trlb))
        self.valid_data = list(zip(val, val_lb))
        self.test_data = list(zip(tst, tst_lb))

    def report(self):
        print(f'Train data length: {len(self.train_data)}')
        print(f'Valid data length: {len(self.valid_data)}')
        print(f'Test data length: {len(self.test_data)}')

    def paus(self, part='train'):
        assert part in ['train', 'valid', 'test'], 'cat must be either of "train", "valid", "test"'
        if part == 'train':
            np.random.shuffle(self.train_data)
        elif part == 'valid':
            np.random.shuffle(self.test_data)
        elif part == 'test':
            np.random.shuffle(self.valid_data)
        else:
            raise ValueError(f'part type {part} not known')
            
    def prepare2coco(self,partition='train'):
        assert partition in ['train', 'valid', 'test'], 'partiton not known'
        
        if self.out_dir is None:
            raise ValueError('Path to save the imges is not known')
        save_dir = os.path.join(self.out_dir, partition)
        
        jason_file = os.path.join(save_dir, partition +'_annotations.json')
        
        if partition=='train':
            files = self.train_data
        elif partition == 'valid':
            files = self.valid_data
        elif partition == 'test':
            files = self.test_data
        
        values = dict2coco(files=files, out_file_name=jason_file, return_valid=True)
        
        geotif2jpg(file=values, dst_dir=save_dir)
        
class FolderParser:
    '''Samples and prepares test data for evaluation'''
    def __init__(self, root_dir, out_dir=None):
        self.root_dir = root_dir
        self.out_dir = out_dir
        
        images = sorted(glob(root_dir+'/images'+'/*.tif'))
        labels = sorted(glob(root_dir+'/labels'+'/*.tif'))
        
        assert len(images) == len(labels), 'images and labels have no equal lengths'
        
        self.data = list(zip(images, labels))

    def report(self):
        print(f'The data has a pair of : {len(self.data)}')

    def paus(self, part='train'):
        np.random.shuffle(self.data)
        
    def prepare2coco(self,partition='train'):
        if self.out_dir is None:
            raise ValueError('Path to save the imges is not known')
        save_dir = os.path.join(self.out_dir, partition)
        
        jason_file = os.path.join(save_dir, partition +'_annotations.json')
        
        files = self.data
        values = dict2coco(files=files, out_file_name=jason_file, return_valid=True)
        
        geotif2jpg(file=values, dst_dir=save_dir)
