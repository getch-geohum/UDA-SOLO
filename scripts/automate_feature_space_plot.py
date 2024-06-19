# Import necessary packages to the environment
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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
from functools import partial
import warnings
import numpy as np
import skimage
from skimage.io import imread, imsave
from skimage import measure
from skimage.io import imread
from glob import glob
from time import gmtime, strftime
from scipy.spatial import distance
import random
import numpy as np
from skimage.io import imsave
import string  
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm # new
from sklearn.manifold import TSNE
#import learn2learn as l2l
import seaborn as sns
import torchvision.models as models

from ipdb import set_trace as st
from sklearn import svm
import torch
import numpy as np
import argparse
import json
import os
import copy
from skimage.io import imread
from skimage import measure
from skimage import draw
from torchvision.ops import masks_to_boxes
#import mmcv

## This is mainly for SOLO model
# import mmcv
# from mmcv.runner import load_checkpoint
# from mmdet.apis import inference_detector, show_result_pyplot
#from mmdet.models import build_detector
# from mmdet.datasets import build_dataset, build_dataloader
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from itertools import cycle
# import torch
# import numpy as np
# import argparse
# import json
# import os
# import copy

### Common for all input output images ###

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


def shiftaxis(img):
    if len(img.shape) == 2:
        pass
    elif len(img.shape) == 3:
        imreshap = np.swapaxes(np.swapaxes(img, 1,2), 0,1)
    elif len(img.shape) == 4:
        imreshap = np.swapaxes(np.swapaxes(img, 2,3), 1,2)
    else:
        imreshap = None
        print('image with unknown shape is provided, returen None value')
    
    return imreshap

def read_images(im, scale=False, stretch=False, shiftaaxis=True):
    img = imread(im)
    if scale:
        img = scale_8bit(img)
    if stretch:
        img = std_stretch_all(img)
    if shiftaaxis:
        return shiftaxis(img)
    else:
        return img
    
def read_labels(im, shiftaaxis=True):
    img = imread(im)
    if shiftaaxis:
        img = np.expand_dims(img, axis = -1)
        return shiftaxis(img)
    else:
        return img

def auto_read(fold, types='im'):
    if types == 'im':
        fs = [read_images(im) for im in fold]
        fs = np.array(fs)
        return fs
    elif types == 'lb':
        fs = [read_labels(im) for im in fold]
        fs = np.array(fs)
        return fs
    else:
        raise ValueError('Unknown image type')


def run_batch(tensor, b_size, M, model_type='seg'):
    '''
    Tensor: image tensor read from batch of file paths
    
    '''
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert model_type in ['seg', 'det']
    if model_type == 'det':
        lists = []
        inds = list(range(0, tensor.shape[0],b_size))
        for i in inds:
            with torch.no_grad():
                im = tensor[i:i+b_size,:3,:,:].to(device, dtype=torch.float)
                feature, _ , _ =  M.forward_train(im)
                lists.append(feature[-1].cpu().reshape(feature[-1].shape[0],-1))
        lists = torch.cat(lists, dim=0)
        return lists
    elif model_type == 'seg': ## work on any segmentation model
        lists = []
        inds = list(range(0, tensor.shape[0],b_size))
        for i in inds:
            with torch.no_grad():
                im = tensor[i:i+b_size,:3,:,:].float().cuda() # .to(device, dtype=torch.float)  # check
                print(f'image shape: {im.shape}')
                feature =  torch.nn.AvgPool2d(8)(M(im)[-1])
                print(f'feature shape: {feature.shape}')
                lists.append(feature.cpu().reshape(feature.shape[0],-1))
                # lists.append(feature[-1].cpu().reshape(feature[-1].shape[0],-1))
        lists = torch.cat(lists, dim=0)
        return lists

# TSNE related features 
def compute_iou(mask1, mask2):
    intersection = np.sum(mask1*mask2)
    union = np.sum(mask1) + np.sum(mask2) - intersection
    return intersection/union

def union_mask(masks):
    sum_masks = np.zeros_like(masks[0])
    for mask in masks:
        sum_masks += mask
    return np.where(sum_masks != 0., 1, 0.)

def compute_iot(mask1, mask2):
    intersection = np.sum(mask1*mask2)
    return intersection / np.sum(mask2)

class ConvexHull: # computes the convex hull of two masks
    def __init__(self, array = True, return_mask=False):
        self.array = array
        self.return_mask = return_mask
    def find_bbox(self, im):
        if not self.array:
            im = np.load(open(self.im, 'rb'), allow_pickle=True)
        img_msk = im.astype(np.uint8) #imread(mask_path).astype(np.uint8)
        shp = img_msk.shape
        assert len(shp)==2, 'shape of the mask should be two dimensional. Please check it'
        contours = measure.find_contours(img_msk, 0.01)
        mask = np.zeros((len(contours), shp[0], shp[1])).astype(np.uint8)
        
        for j in range(len(contours)):
            poly = contours[j]
            xy = [poly[i] for i in range(poly.shape[0])]
            x = [m[0] for m in xy]
            y = [m[1] for m in xy]

            rr, cc = draw.polygon(x,y)
            mask[j, rr, cc] = 1
        masks = torch.from_numpy(mask==1)
        boxes = masks_to_boxes(masks)
        xmin, xmax, ymin, ymax = boxes[:,0].min().item(), boxes[:,2].max().item(), boxes[:,1].min().item(), boxes[:,3].max().item()
        return xmin, xmax, ymin, ymax
        
    def compute(self,x1, x2):
        assert x1.shape == x2.shape, 'input masks shape is not the same'
        a, b, c, d = self.find_bbox(x1)
        e, f, g, h = self.find_bbox(x2)
        
        xx, XX = int(min(a,e)), int(max(b, f)) # xmin and xmax of two masks
        yy, YY = int(min(c,g)), int(max(d,h))  # ymin, ymax
        out = np.zeros(x1.shape, dtype=np.uint8)
        
        
        out[xx:XX,yy:YY] = 1
        count = out.sum()
        
        if self.return_mask:
            return count, out
        else:
            return count


class GeneratEmbedFeatures:
    def __init__(self, data_root, out_root, model):
        self.data_root = data_root
        self.out_root = out_root
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.format = '.tif'
        
    def generate_tsne_features(self):
        if not os.path.exists(self.out_root):
            os.makedirs(out_root, exist_ok=True)
        folds = os.listdir(self.data_root)

        sit_ind = []
        fname = []
        length = []
        array = []
        print(f'Model device: {self.device}')
        self.model.to(self.device)
        
        for i, fold in enumerate(folds):
            print(f'processing for folder {fold}')
            path = os.path.join(self.data_root, fold, 'images')
            paths = glob(path + '/*{}'.format(self.format))
            if len(paths) == 0:
                pass
            else:
                sit_ind+=[i]*len(paths)
                imgs = auto_read(paths)
                imgs = torch.from_numpy(imgs/255)
                outs = run_batch(imgs, 10, self.model) #model(imgs)
                print(f'Output shape for folder: {fold}, input shape: :{imgs.shape}, output shape: {outs.shape}')
                array.append(outs)
                fname.append(fold)
                length.append(outs.shape[0])
        big_array = torch.cat(array, dim=0)

        npyy = big_array.numpy()
        assert len(sit_ind) == npyy.shape[0], 'Site index length and array length not matching'

        np.save(self.out_root + '/deep_features', npyy)
        np.save(self.out_root + '/site_index', sit_ind)
        print(f'deep features written in {self.out_root}')
        print(f'Site index written in {self.out_root}')

        inds = list(np.unique(sit_ind))

        sampleDict = {}
        for i in range(len(inds)):
            sampleDict[inds[i]] = fname[i]

        with open(self.out_root + '/sit_ind_sit_name.json', "w") as write_file:
            json.dump(str(sampleDict), write_file, indent=4)
            print(f'Site names written in {self.out_root}')

        print('Computing TSNE')
        tsne = TSNE(n_components=2,
                    init='pca',
                    perplexity=50,
                    n_iter=5000,
                    n_jobs=-1).fit_transform(big_array)

        np.save(self.out_root + '/tsne_out', tsne)
        print(f'TSNE features written in {self.out_root}')
        print(f'Feature space comutation done!')
        if len(sit_ind) !=tsne.shape[0]:
            print(f'Warning, TSNE FEATURE AND INDEX DID NOT MATCH')
    
    def plot_joint_feature_space(self):
        index_site_name = self.out_root + '/sit_ind_sit_name.json'
        with open(index_site_name) as jopen_file:
            sites = ast.literal_eval(json.load(jopen_file))
        sites = list(sites.values())
        sites.sort()
        
        sites_names = sites 
        sites_names.sort()

        data = np.load(self.out_root + '/tsne_out.npy')
        campcolor = np.load(self.out_root + '/site_index.npy')

        ########################################################
        ###### Outliers  

        palette = sns.color_palette('hls',n_colors=len(sites_names))   # n_colors = number of sites you are analizing

        colors_camps = np.empty((len(sites_names),1,3))

        for i in range(len(colors_camps)):
            colors_camps[i] = np.array(palette[i])


        figfigfig, axaxax = plt.subplots(2,5,sharex=True, sharey=True, figsize=(15,9))

        for i, camp in enumerate(sites):
            print(camp)
            id_x, id_y = i//5, i%5
            d = data[campcolor==i]
            d2 = data[campcolor!=i]
            x, y = d[:,0], d[:,1]
            x2, y2 = d2[:,0], d2[:,1]
            color = colors_camps[i]

            c = tuple(color[0])
            xmin, xmax = data[:, 0].min(), data[:,0].max()  # chack
            ymin, ymax = data[:, 1].min(), data[:,1].max()  # chack

            print('computing SVM')

            algorithm = svm.OneClassSVM(nu=.2,kernel="rbf", gamma=0.01)
            print('SVM done')

            xx, yy = np.meshgrid(np.linspace(xmin-2, xmax+2, 400), np.linspace(ymin-2, ymax+2, 400)) 


            inliers = algorithm.fit(d).predict(d)
            Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])

            # This plots all the points in a gray cloud
            axaxax[id_x,id_y].scatter(x2, y2, c='gray', s=.9, label=camp, alpha=.5, marker='o')

            df = algorithm.decision_function(np.c_[xx.ravel(), yy.ravel()])
            df = df.reshape(xx.shape)
            Z = Z.reshape(xx.shape)

            # This plots the points and surfaces site by site
            axaxax[id_x,id_y].contourf(xx, yy, df, levels=[-.9, df.max()], colors=color, alpha = .3) # alpha .6
            axaxax[id_x,id_y].scatter(x,y,c=color, s=1, label=camp,
                       alpha=1, marker='o')
            axaxax[id_x, id_y].set_title(sites_names[i], fontdict={'family':'serif', 'fontsize': 11,
                'fontweight' : 'normal',
                'verticalalignment': 'baseline',
                'horizontalalignment': 'center'}, pad=2.5)

            plt.setp( axaxax[id_x,id_y].get_xticklabels(), visible=False)
            plt.setp( axaxax[id_x, id_y].get_yticklabels(), visible=False)
            plt.xticks([]),plt.yticks([])

            plt.subplots_adjust(hspace=0.2)

        plt.savefig(os.path.join(self.out_root,'vgg19-tsne-site-colors.png'), format='png', bbox_inches='tight')

        plt.show(block=True)
    
    def plot_joint_single_space(self):
        index_site_name = self.out_root + '/sit_ind_sit_name.json'
        with open(index_site_name) as jopen_file:
            sites = ast.literal_eval(json.load(jopen_file))
        sites = list(sites.values())
        sites.sort()
        
        sites_names = sites 
        sites_names.sort()

        data = np.load(self.out_root + '/tsne_out.npy')
        campcolor = np.load(self.out_root + '/site_index.npy')

        ########################################################
        ###### Outliers  

        palette = sns.color_palette('hls',n_colors=len(sites_names))   # n_colors = number of sites you are analizing

        colors_camps = np.empty((len(sites_names),1,3))

        for i in range(len(colors_camps)):
            colors_camps[i] = np.array(palette[i])


        figfigfig, axaxax = plt.subplots(figsize=(10,10)) # 2,5,sharex=True, sharey=True,
        markers = ['o', '*', '^','<','>','v','p','+','d','1']
        cm_list = ["#FF00FF","#8A2BE2","#00FF00","#00FFFF","#0000FF","#000080","#DAA520","#2F4F4F","#FFD700","#FF0000"] 

        for i, camp in enumerate(sites):
            print(camp)
            # id_x, id_y = i//5, i%5
            d = data[campcolor==i]
            # d2 = data[campcolor!=i]
            x, y = d[:,0], d[:,1]
            # x2, y2 = d2[:,0], d2[:,1]
            color = colors_camps[i]

            c = tuple(color[0])
            xmin, xmax = data[:, 0].min(), data[:,0].max()  # chack
            ymin, ymax = data[:, 1].min(), data[:,1].max()  # chack

            print('computing SVM')

            algorithm = svm.OneClassSVM(nu=.2,kernel="rbf", gamma=0.01)
            print('SVM done')

            xx, yy = np.meshgrid(np.linspace(xmin-2, xmax+2, 400), np.linspace(ymin-2, ymax+2, 400)) 


            inliers = algorithm.fit(d).predict(d)
            Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])

            # This plots all the points in a gray cloud
            # axaxax[id_x,id_y].scatter(x2, y2, c='gray', s=.9, label=camp, alpha=.5, marker='o')

            df = algorithm.decision_function(np.c_[xx.ravel(), yy.ravel()])
            df = df.reshape(xx.shape)
            Z = Z.reshape(xx.shape)

            # This plots the points and surfaces site by site
            # axaxax.contourf(xx, yy, df, levels=[-.9, df.max()], colors=cm_list[i], alpha = .3) # alpha .6
            axaxax.scatter(x,y,c=cm_list[i], s=5, label=camp, alpha=1, marker=markers[i])
            # axaxax[id_x, id_y].set_title(sites_names[i], fontdict={'family':'serif', 'fontsize': 11,
            #     'fontweight' : 'normal',
            #     'verticalalignment': 'baseline',
            #     'horizontalalignment': 'center'}, pad=2.5)

        plt.setp(axaxax.get_xticklabels(), visible=False)
        plt.setp(axaxax.get_yticklabels(), visible=False)
        plt.ylabel('PCA-1')
        plt.xlabel('PCA-2')
        plt.xticks([]),plt.yticks([])
        plt.subplots_adjust(hspace=0.2)
        plt.legend()
        plt.savefig(os.path.join(self.out_root,'all-in-one_tsne-site-colors.png'), format='png', bbox_inches='tight')
        plt.show(block=True)
        
    def generate_mask(self):
        print('Generating masks!...')
        sit = self.out_root + '/sit_ind_sit_name.json'
        with open(sit) as jopen_file:
            sites = ast.literal_eval(json.load(jopen_file))
        sites = list(sites.values())
        sites.sort()
    
        sites_names = sites
        sites_names.sort()

        data = np.load(self.out_root + '/tsne_out.npy')
        campcolor = np.load(self.out_root + '/site_index.npy')
        
        # c = tuple(color[0])
        xmin, xmax = data[:, 0].min(), data[:,0].max()  # chack
        ymin, ymax = data[:, 1].min(), data[:,1].max()  # chack

    
        ###### Outliers detection ###

        palette = sns.color_palette('hls',n_colors=len(sites))   # n_colors = number of sites you are analizing

        colors_camps = np.empty((len(sites),1,3))

        for i in range(len(colors_camps)):
            colors_camps[i] = np.array(palette[i])

        fig = plt.figure()


        for i, camp in enumerate(sites):
            d = data[campcolor==i]
            x, y = d[:,0], d[:,1]
            color = colors_camps[i]

            c = tuple(color[0])

            algorithm = svm.OneClassSVM(nu=.2,kernel="rbf", gamma=.01)

            xx, yy = np.meshgrid(np.linspace(xmin-2, xmax+2, 400), np.linspace(ymin-2, ymax+2, 400))
            print(f'SVM done for camp {camp}')

            inliers = algorithm.fit(d).predict(d)
            Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])

            df = algorithm.decision_function(np.c_[xx.ravel(), yy.ravel()])
            df = df.reshape(xx.shape)
            print(df.shape)
            Z = Z.reshape(xx.shape)


            mask_camp = np.where(df>=-1, 1, 0)
        #     mask_city = np.where(df>=-.9, 1, 0)
            save_dir = self.out_root + '/masks'  
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            np.save(save_dir + '/mask_' + camp, mask_camp)
            classes, counts = np.unique(mask_camp, return_counts=True)  # depends on platform
            print(classes, counts)

            plt.imshow(255*mask_camp, cmap='gray',interpolation='none')

            plt.show(block=True)

        plt.show()
    
    def compute_simmilarity(self):
        print(f'computing simmilarity!')
        sit = self.out_root + '/sit_ind_sit_name.json'
        with open(sit) as jopen_file:
            sites = ast.literal_eval(json.load(jopen_file)) # modified
        sites = list(sites.values())
        sites.sort()
    

        sites_names = sites
        sites_names.sort()
        mask_path = self.out_root + '/masks/mask_{}.npy' 

        mat1 = np.zeros(shape=(len(sites), len(sites)))

        for i, camp1 in enumerate(sites):
            for j, camp2 in enumerate(sites):
                mask1 = np.load(mask_path.format(camp1))
                mask2 = np.load(mask_path.format(camp2))
                iou = compute_iou(mask1, mask2)
                mat1[i,j] = iou
                # print('iou',city1,city2,'=', iou)

        cm1 = pd.DataFrame(mat1, index=sites, columns=sites)
        fig = plt.figure(figsize=(4,4))
        cmap = sns.cubehelix_palette(as_cmap=True)
        ax = sns.heatmap(cm1, cmap="Blues", vmin=0, vmax=1, linewidths=.3, square=True, cbar_kws={"shrink": .7})
        top, bottom = ax.get_ylim()
        cbar_ax = ax.figure.axes[-1]
        cbar_labels = cbar_ax.get_yticklabels()
        cbar_ax.set_yticklabels(cbar_labels, fontsize='x-small', fontdict={'family': 'serif'})

        # st()
        ax.set_ylim(top + 0.5, bottom - 0.5,)
        ax.set_xticklabels(sites_names, rotation=80, fontsize='small', va='top', fontdict={'family':'serif'})
        ax.set_yticklabels(sites_names, fontsize='small', fontdict={'family':'serif'})

        # ax.set_aspect('equal')

        fig.tight_layout()

        plt.xlabel('Sites', fontdict={'family':'serif', 'fontsize': 11,
            'fontweight' : 'normal',
            'verticalalignment': 'baseline',
            'horizontalalignment': 'center'}, labelpad=11)
        plt.ylabel('Sites', fontdict={'family':'serif', 'fontsize': 11,
            'fontweight' : 'normal',
            'verticalalignment': 'baseline',
            'horizontalalignment': 'center'}, labelpad=-2)
        cm1.to_csv(self.out_root + '/iou.csv')

        mat = np.zeros(shape=(len(sites), len(sites)))

        for i, camp1 in enumerate(sites):
            for j, camp2 in enumerate(sites):
                mask1 = np.load(mask_path.format(camp1))
                mask2 = np.load(mask_path.format(camp2))
                iot = compute_iot(mask1, mask2)
                mat[i,j] = iot
                # print('iou',city1,city2,'=', iou)

        cm = pd.DataFrame(mat, index=sites, columns=sites)
        fig = plt.figure(figsize=(4,4))
        cmap = sns.cubehelix_palette(as_cmap=True)
        ax = sns.heatmap(cm, cmap="Blues", vmin=0, vmax=1, linewidths=.3, square=True, cbar_kws={"shrink": .7})
        top, bottom = ax.get_ylim()
        cbar_ax = ax.figure.axes[-1]
        cbar_labels = cbar_ax.get_yticklabels()
        cbar_ax.set_yticklabels(cbar_labels, fontsize='x-small', fontdict={'family': 'serif'})

        ax.set_ylim(top + 0.5, bottom - 0.5,)
        ax.set_xticklabels(sites_names, rotation=80, fontsize='small', va='top', fontdict={'family':'serif'})
        ax.set_yticklabels(sites_names, fontsize='small', fontdict={'family':'serif'})

        # ax.set_aspect('equal')

        fig.tight_layout()

        plt.xlabel('Sites', fontdict={'family':'serif', 'fontsize': 11,
            'fontweight' : 'normal',
            'verticalalignment': 'baseline',
            'horizontalalignment': 'center'}, labelpad=11)
        plt.ylabel('Sites', fontdict={'family':'serif', 'fontsize': 11,
            'fontweight' : 'normal',
            'verticalalignment': 'baseline',
            'horizontalalignment': 'center'}, labelpad=-2)
        
        cm1.to_csv(self.out_root + '/iot.csv')


        fig, (ax1, cbar_ax, ax2) = plt.subplots(1,3, figsize=(9, 4), gridspec_kw={'width_ratios': [4, .2, 4]})


        sns.heatmap(cm1, cmap="Blues", vmin=0, vmax=1, linewidths=.3, square=True, ax=ax1, cbar_kws={"shrink": .9}, cbar_ax=cbar_ax)
        top1, bottom1 = ax1.get_ylim()

        sns.heatmap(cm, cmap="Blues", vmin=0, vmax=1, linewidths=.3, square=True, ax=ax2, cbar=False)
        top, bottom = ax2.get_ylim()

        cbar_labels = cbar_ax.get_yticklabels()
        cbar_ax.set_yticklabels(cbar_labels, fontsize='xx-small', fontdict={'family': 'serif'})

        ax1.set_ylim(top1 + 0.5, bottom1 - 0.5,)
        ax1.set_xticklabels(sites_names, rotation=80, fontsize='small', va='top', fontdict={'family':'serif'})
        ax1.set_yticklabels(sites_names, fontsize='small', fontdict={'family':'serif'})
        ax1.set_title('IoU score', fontdict={'family':'serif'}, fontsize=12, pad=8)

        ax2.set_ylim(top + 0.5, bottom - 0.5,)
        ax2.tick_params(rotation=0)
        ax2.set_xticklabels(sites_names, rotation=80, fontsize='small', va='top', fontdict={'family':'serif'})
        ax2.set_yticklabels(sites_names, fontsize='small', fontdict={'family':'serif'})
        ax2.yaxis.tick_right()
        ax2.set_title('IoT score', fontdict={'family':'serif'}, fontsize=12, pad=8)

        plt.subplots_adjust(hspace=.01)

        fig.tight_layout()

        plt.savefig(self.out_root + '/IoU-IoT-train-test.png', format='png', bbox_inches='tight')

        plt.show(block=True)
        
    
    def compute_simmilarity_GIOU(self):
        print(f'computing simmilarity!')
        sit = self.out_root + '/sit_ind_sit_name.json'
        with open(sit) as jopen_file:
            sites = ast.literal_eval(json.load(jopen_file)) # modified
        sites = list(sites.values())
        sites.sort()
    

        sites_names = sites
        sites_names.sort()
        mask_path = self.out_root + '/masks/mask_{}.npy' 

        mat = np.zeros(shape=(len(sites), len(sites)))

        for i, camp1 in enumerate(sites):
            for j, camp2 in enumerate(sites):
                mask1 = np.load(mask_path.format(camp1))
                mask2 = np.load(mask_path.format(camp2))
                # iou = compute_iou(mask1, mask2)
                intersection = np.sum(mask1*mask2)
                union = np.sum(mask1) + np.sum(mask2) - intersection
                iou = intersection/union
                hul = ConvexHull().compute(mask1, mask2)
                giou = iou-(hul-union)/hul
                mat[i,j] = giou
                # print('giou',camp,camp,'=', iou)

        cm1 = pd.DataFrame(mat, index=sites, columns=sites)
        fig = plt.figure(figsize=(4,4))
        cmap = sns.cubehelix_palette(as_cmap=True)
        ax = sns.heatmap(cm1, cmap="RdYlGn", vmin=-1, vmax=1, linewidths=.3, square=True, cbar_kws={"shrink": .7})
        top, bottom = ax.get_ylim()
        cbar_ax = ax.figure.axes[-1]
        cbar_labels = cbar_ax.get_yticklabels()
        cbar_ax.set_yticklabels(cbar_labels, fontsize='x-small', fontdict={'family': 'serif'})

        # st()
        # ax.set_ylim(top + 0.5, bottom - 0.5,)
        ax.set_xticklabels(sites_names, rotation=80, fontsize='small', va='top', fontdict={'family':'serif'})
        ax.set_yticklabels(sites_names, fontsize='small', fontdict={'family':'serif'})

        # ax.set_aspect('equal')

        fig.tight_layout()

        plt.xlabel('Sites', fontdict={'family':'serif', 'fontsize': 11,
            'fontweight' : 'normal',
            'verticalalignment': 'baseline',
            'horizontalalignment': 'center'}, labelpad=11)
        plt.ylabel('Sites', fontdict={'family':'serif', 'fontsize': 11,
            'fontweight' : 'normal',
            'verticalalignment': 'baseline',
            'horizontalalignment': 'center'}, labelpad=-2)
        
        cm1.to_csv(self.out_root + '/GIOU.csv')
        
        plt.savefig(self.out_root + '/GIOU-train-test.png', format='png', bbox_inches='tight')
        plt.show(block=True)

        

    def crossfeature_space_plot(self, one2one=False):
        print('Generating cross feature space plots')
        sit = self.out_root + '/sit_ind_sit_name.json'
        with open(sit) as jopen_file:
            names = ast.literal_eval(json.load(jopen_file)) # modified
        names = list(names.values())
        names.sort()
        
        data = np.load(self.out_root + '/tsne_out.npy')
        indexes = np.load(self.out_root + '/site_index.npy')
        uniq_ind = list(np.unique(indexes))
        if one2one:
            save_dir = self.out_root + '/crossplot'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            for ind in uniq_ind:
                arr1 = data[indexes == ind]
                print('first array shape: {arr1.shape}')
                for ref_ind in uniq_ind:
                    if ind == ref_ind:
                        pass
                    else:
                        ref_arr = data[indexes == ref_ind]
                        print('ref array shape {ref_arr.shape}')
                        plt.scatter(arr1[:,0], arr1[:,1], label = names[ind])
                        plt.scatter(ref_arr[:,0], ref_arr[:,1], label = names[ref_ind])
                        plt.xlabel('component 1')
                        plt.ylabel('component 2')
                        plt.legend()
                        plt.savefig("{}/{}_wiz_{}.png".format(save_dir, names[ind], names[ref_ind]))
                        plt.show()
            print(f'Cross ploting done and saved in {save_dir}')
        else:
            for ind in uniq_ind:
                arr1 = data[indexes == ind]
                plt.scatter(arr1[:,0], arr1[:,1], label = names[ind])
            plt.xlabel('component 1')
            plt.ylabel('component 2')
            plt.legend()
            plt.savefig("{}/combined_feature_space.png".format(self.out_root))
            plt.show()

def argumentParser():
    parser = argparse.ArgumentParser(description = 'Deep feature space embeding plot')
    parser.add_argument('--data_root', help='data folder, can be either with single task or multi task', type=str, required=False)
    parser.add_argument('--mtype', help='type of a model, segmentation or detection', type=str, default='seg', required=False)
    parser.add_argument('--save_dir', help = 'main root to save the test result', type = str,required=False)
    parser.add_argument('--weights', help='Trained model checkpoint', type=str, required=False)
    parser.add_argument('--config', help='configuration file', type=str, required=False)
    parser.add_argument('--builtin', help='Whether to use builting pretrained model', type=str, required=False, default='yes')
    
    arg = parser.parse_args()
    return arg

if __name__ == "__main__":
    args = argumentParser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mtype == 'seg':
        if args.builtin=='no':
            pass
        else:
            model = models.resnet50(pretrained=True)
#         new_model = copy.deepcopy(model.model.encoder)
        
        model.eval()
        generator = GeneratEmbedFeatures(data_root=args.data_root,out_root=args.save_dir,model=model.to(device))
        generator.generate_tsne_features()
        generator.plot_joint_feature_space()
        generator.plot_joint_single_space() # new
        generator.generate_mask()
        generator.compute_simmilarity()
        generator.compute_simmilarity_GIOU() # new
        generator.crossfeature_space_plot()
        
    elif args.mtype == 'det': # if needed  t se trained model weight on custm data
        print('Segmentation featre space plot on progress...')
        config = mmcv.Config.fromfile(args.config)
        model = build_detector(config.model)
        #model.load_state_dict(torchload(args.weights))
        model.eval()
        generator = GeneratEmbedFeatures(data_root=args.data_root, out_root=args.save_dir,model=model)
        generator.generate_tsne_features()
        generator.plot_joint_feature_space()
        generator.plot_joint_single_space() # new
        generator.generate_mask()
        generator.compute_simmilarity()
        generator.compute_simmilarity_GIOU() # new
        generator.crossfeature_space_plot()
    else:
        raise ValueError('Specified model type {args.mtype} is not known')
