import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot, set_random_seed
from mmdet.models import build_detector
from mmdet.datasets import build_dataset, build_dataloader
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch
import numpy as np
import argparse
from itertools import cycle
import json
import os
import copy
import ot

class FeatureLoss(nn.Module): # after review this function is not used 
    def __init__(self, reduction='sum', alpha=1):
        super(FeatureLoss, self).__init__()
        self.reduction = reduction
        self.alpha = torch.Tensor([alpha]).cuda() # alpha for feature space weighting

    def forward(self, x1, x2):
        assert x1.shape == x2.shape, 'the embedings are not the same'
        if len(x1.shape) !=2:
            x1 = x1.reshape(x1.shape[0],-1)
            x2 = x2.reshape(x2.shape[0],-1)
        b = x1.shape[0]
        A = torch.from_numpy(ot.unif(b)).cuda()
        B = torch.from_numpy(ot.unif(b)).cuda()
        cost = ot.dist(x1, x2, metric='euclidean')
        print('FF: ', A.min(), A.max(), x1.max(), x2.max(), cost.min(), cost.max())
        if not cost.is_cuda:
            cost.cuda()
        gamma = ot.emd(A, B, cost)
        loss = self.alpha*gamma*cost
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f'loss reduction type {self.reduction} is not known')


def compute_lamda_stepwise(n_epoch, dstep, d_epoch, len_loader):
    '''
    n_epoch: total number of training epochs
    dstep: trainning step within a single epoch
    d_epoch: specific epoch within training epochs
    len_loader: length of data loader
    '''
    p = float(dstep + d_epoch * len_loader) / n_epoch / len_loader
    alpha = 2. / (1. + np.exp(-10 * p)) - 1

    return alpha


class FeatureAlignment:
    def __init__(self, reduction='sum', alpha=1):
        '''Computes the domain alignment at the feature space'''
        self.alpha = alpha
        self.reduction = reduction

    def compute(self, x1, x2,reduce_sdim=True, mapping=True):
        assert x1.shape == x2.shape, f'the source and target shapes {x1.shape} and {x2.shape} respectivel are not teh same'
        if reduce_sdim:
            x1 = torch.nanmean(x1, dim=(2,3)) # gives [b, #f] and reduces the spatial dimension
            x2 = torch.nanmean(x2, dim=(2,3))
        a = x1.shape[0]
        b = x2.shape[0]
        #print('inf_nan ff: ', torch.isnan(x1).any(), torch.isnan(x1).any(), torch.isinf(x1).any(), torch.isinf(x1).any())
        A = torch.from_numpy(ot.unif(a)).cuda()
        B = torch.from_numpy(ot.unif(b)).cuda()
        cost = ot.dist(x1.view(a, -1), x2.view(b, -1), metric='euclidean')
        #print('FF: ', [A.min(), A.max()], [B.min(), B.max()], [x1.min(), x1.max()], [x2.min(), x2.max()], [cost.min(), cost.max()])
        if not cost.is_cuda:
            cost.cuda()
        gamma = ot.emd(A, B, cost)

        if mapping:
            sorting = torch.argmax(gamma, dim=0).detach().cpu().numpy().tolist() # if we need to sort the source ground truth that are mmaped with target images
            #sorting = torch.argmax(gamma, dim=1).detach().cpu().numpy().tolist() # if we need to sort the target input to label space alignment(unified masks predicted from the mask branch)
        else:
            sorting = None

        loss = self.alpha*gamma*cost
        if self.reduction == 'mean':
            return loss.mean(), sorting
        elif self.reduction == 'sum':
            return loss.sum()/a, sorting
        else:
            raise ValueError(f'The reduction type {self.reduction} not known!')

class LabeAlignment:
    def __init__(self, beta=0.1, reduction='sum'):
        self.beta = beta
        self.reduction=reduction
        '''Computes label space distance'''
    def compute(self, x1, x2):
        assert x1.shape == x2.shape, f'the source and target shapes {x1.shape} and {x2.shape} respectivel are not teh same'
        a = x1.shape[0]
        b = x2.shape[0]
        #print('inf_nan ll: ', torch.isnan(x1).any(), torch.isnan(x1).any(), torch.isinf(x1).any(), torch.isinf(x1).any())
        A = torch.from_numpy(ot.unif(a)).cuda()
        B = torch.from_numpy(ot.unif(b)).cuda()
        cost = ot.dist(x1.view(a, -1), x2.view(b, -1).float().cuda(), metric= 'euclidean')
    
        if not cost.is_cuda:
            cost.cuda()
        gamma = ot.emd(A, B, cost)
        if not gamma.is_cuda:
            gamma.cuda()
        label_loss = self.beta*gamma*cost
        if self.reduction=='sum':
            return label_loss.sum()/a
        elif self.reduction=='mean':
            return label_loss.mean() #.mean()  # sum can also be considered
        else:
            raise ValueError(f'The reduction type {self.reduction} not known!')


# step wise lamda compute
def compute_alpha(n_epoch, dstep, d_epoch, len_loader):
    '''
    n_epoch: total number of training epochs
    dstep: trainning step within a single epoch
    d_epoch: specific epoch within training epochs
    len_loader: length of data loader
    '''
    p = float(dstep + d_epoch * len_loader) / n_epoch / len_loader
    alpha = 2. / (1. + np.exp(-10 * p)) - 1

    return alpha

# step wise learning rate adjuster
def compute_adapt_lr(muno_lr, n_epoch, dstep, d_epoch, len_loader):
    p = float(dstep + d_epoch * len_loader) / n_epoch / len_loader
    step_lr = muno_lr/(1+10*p)**0.75
    return step_lr

def order(x, inds): # to sortout source ground truth for label space alignment
    return [x[ind] for ind in inds]

def train_OT(Config, checkpoint_dir,source_data, target_data, epochs=65, learning_rate=0.001, alpha=2, beta=10, checkpoint=None):
    
    print(f'epoch: {epochs} lr : {learning_rate}  beta: {beta}  alpha: {alpha}')
    config = mmcv.Config.fromfile(Config)
    config.model.bbox_head.num_classes = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # #     device ='cuda:0'
    model = build_detector(config.model)
    checkpoint = None
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint)["state_dict"])
        print(f'Pretrained model weight loaded from {checkpoint}')
    else:
        print(f'Model weight is None, training will strat from scratch')
    model.CLASSES = ('bacground','dwelling',)
    f_alignment = FeatureAlignment(alpha=alpha) # this are not any more torch modules, just classes
    l_alignment = LabeAlignment(beta=beta)


    scaler = GradScaler()

    model.train()  # Convert the model into evaluation mode
    model.to(device)
    #f_alignment.to(device) # as they are now their inherritence from nn.Module is removed 
    #l_alignment.to(device)

    aconfig = copy.deepcopy(config.data.train)
    bconfig = copy.deepcopy(config.data.test) # only collects the images
    cconfig = copy.deepcopy(config.data.train)

    # change target and source data path
    aconfig['ann_file'] = source_data + '/train/train_annotations.json'
    aconfig['img_prefix'] = source_data + '/train/'

    bconfig['ann_file'] = target_data + '/train/train_annotations.json'
    bconfig['img_prefix'] = target_data + '/train/'

    cconfig['ann_file'] = source_data + '/valid/valid_annotations.json'
    cconfig['img_prefix'] = source_data + '/valid/'

    datasetA = build_dataset(aconfig)
    loaderA = build_dataloader(datasetA, config.data.imgs_per_gpu, config.data.workers_per_gpu, dist=False)
    
    #print("Images per gpu:---->", config.data.imgs_per_gpu)

    datasetB = build_dataset(bconfig)
    loaderB = build_dataloader(datasetB, config.data.imgs_per_gpu, config.data.workers_per_gpu, dist=False)
    datasetC = build_dataset(cconfig)
    loaderC = build_dataloader(datasetC, config.data.imgs_per_gpu, config.data.workers_per_gpu, dist=False)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate) #  momentum=0.9, weight_decay=0.0001

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30], gamma=0.1)

    ins_ls_s = []        # instance loss
    cat_ls_s = []        # category loss
    feat_ls_u = []       # domain loss
    ins_ls_u = []
    valid_loss = [] # validation loss
    epoch = []
    step = []
    best = 0
    control=0 # for early stoping

    #insteps = max(len(loaderA), len(loaderB))
    max_data_length = max(len(loaderA), len(loaderB))
    for i in range(1, epochs+1):
        for j, (da, db) in enumerate(zip(cycle(loaderA), loaderB)):
            img_meta = da['img_meta']
            imeta = img_meta.data[0]   # metdata information after data transformed

            gt_bboxes = da['gt_bboxes']
            bbox = gt_bboxes.data[0]
            bbox = [bb.to(device) for bb in bbox] # list comprehension

            gt_labels = da['gt_labels']
            gt_lbl = gt_labels.data[0]

            gt_lbl = [gt.to(device) for gt in gt_lbl] # list compression

            gt_masks = da['gt_masks']
            gt_msk = gt_masks.data[0]
            
            lamda_step = compute_lamda_stepwise(n_epoch=epochs, dstep=j, d_epoch=i, len_loader=max_data_length)
            
            optimizer.zero_grad()
            lamda_step = 0.01
            with torch.cuda.amp.autocast():
                a, b, c  = model.forward_train(da['img'].data[0].to(device)) # forwar pass results in category pred and kernel pred
                d, e, f  = model.forward_train(db['img'][0].to(device)) # forward pass target
                # deep features from last Resnet Layer
                feat_a = a[-1]
                feat_b = d[-1]
                feat_loss, sind = f_alignment.compute(feat_a, feat_b)
 
                loss_inputs = b + (c, bbox, gt_lbl, gt_msk, imeta, model.train_cfg)
                #loss_inputs_unsup = e + (f, bbox, gt_lbl, gt_msk, imeta, model.train_cfg)  # this is trick to prepare masks as solov2 format
                loss_inputs_unsup = e + (f, order(bbox, sind), order(gt_lbl, sind), order(gt_msk, sind), imeta, model.train_cfg)  # the ordering made as per gamm transport plan 
                ins, cate = model.bbox_head.loss(*loss_inputs_unsup, gt_bboxes_ignore=None, return_raw=True)
                ins_ref = ins['ref']
                ins_pred = ins['pred']    
            
                sup_loss = model.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=None, return_raw=False)  # supervised segmenter loss_cate
                inst_align_loss = l_alignment.compute(ins_pred, ins_ref)
            
                #print(ins_pred.shape, ins_ref.shape)

            sup_ins_loss = sup_loss['loss_ins'] # scaler.scale(sup_loss['loss_ins'])
            sup_cat_loss = sup_loss['loss_cate'] # scaler.scale(sup_loss['loss_cate'])
            sp_tot_loss = sup_ins_loss + sup_cat_loss # scaler.scale(sup_loss['loss_ins'] + sup_loss['loss_cate'])
            total_loss = sp_tot_loss + lamda_step*(feat_loss + inst_align_loss)

            #print('Losses: ', feat_loss.item(), inst_align_loss.item(), sp_tot_loss.item(), total_loss.item())

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=40, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()

            if i == 1:
                vls = 0
            else:
                vls = valid_loss[-1]
            print(f'Epoch: {i}, step: {j}, u_fet: {feat_loss.item()}, u_in: {inst_align_loss.item()}, s_in: {sup_ins_loss.item()}, tot_l: {total_loss.item()}, val_tot: {vls}')

            ins_ls_s.append(sup_ins_loss.item())       # instance loss
            cat_ls_s.append(sup_cat_loss.item())        # category loss
            ins_ls_u.append(inst_align_loss.item())
            feat_ls_u.append(feat_loss.item())
            epoch.append(i)
            step.append(j)

        # Validate and save the optimal weights
        lr_scheduler.step()
        v_loss = 0.0
        for m, valid_data in enumerate(loaderC):
            with torch.no_grad():
                img_meta_v = valid_data['img_meta']
                imeta_v= img_meta_v.data[0]   #metdata information after data transformed
                gt_bboxes_v = valid_data['gt_bboxes']
                bbox_v = gt_bboxes_v.data[0]
                bbox_v[0] = bbox_v[0].to(device)
                bbox_v[1] = bbox_v[1].to(device)

                gt_labels_v = valid_data['gt_labels']
                gt_lbl_v = gt_labels_v.data[0]
                gt_lbl_v[0] = gt_lbl_v[0].to(device)
                gt_lbl_v[1] = gt_lbl_v[1].to(device)

                gt_masks_v = valid_data['gt_masks']
                gt_msk_v = gt_masks_v.data[0]

                aa, bb, cc  = model.forward_train(valid_data['img'].data[0].to(device))
                loss_input_valid = bb + (cc, bbox_v, gt_lbl_v, gt_msk_v, imeta_v, model.train_cfg)
                val_loss = model.bbox_head.loss(*loss_input_valid, gt_bboxes_ignore=None, return_raw=False)
                v_loss+=val_loss['loss_ins'].cpu().numpy() + val_loss['loss_cate'].cpu().numpy()
                control+=1

        mean_valid_loss = v_loss/control
        valid_loss.append(mean_valid_loss)

        #if i%5 == 0 or i in [1, epoch]:
        #    name = 'checkpoint.pth'
        #    chpoint = os.path.join(checkpoint_dir, name)
        #    torch.save(model.state_dict(), chpoint)


        if i == 1:
            name = 'checkpoint.pth'
            chpoint = os.path.join(checkpoint_dir, name)
            torch.save(model.state_dict(), chpoint) # save the checkpoint
            best = mean_valid_loss
        elif i>1:
            if mean_valid_loss <= best:
                best = mean_valid_loss
                name = 'checkpoint.pth'
                chpoint = os.path.join(checkpoint_dir, name)
                torch.save(model.state_dict(), chpoint) # save the last check point incase the final checkpoint
                control=0  # early stopping 
            else:
                control+=1
        elif control>5: # early stopping
            break

    metric = dict(STEP = step, EPOCH = epoch, INS_S=ins_ls_s, CAT_S=cat_ls_s, INS_U=ins_ls_u, FEAT_U=feat_ls_u, VL=valid_loss)
    save_name = os.path.join(checkpoint_dir, 'summary.json')
    with open(save_name, 'w') as fp:
        json.dump(str(metric), fp)

def argumentParser():
    parser = argparse.ArgumentParser(description="Sample mixed model trainer parameters")
    parser.add_argument("--config", help="Directory for model configuration",type=str, default = './SOLO/configs/solov2/train_configs.py')
    parser.add_argument('--save_dir', help="Directory to save the checkpoint", type=str, default = 'path 2 save directory')
    parser.add_argument('--epochs', help='number of epochs', type=int, default=65)
    parser.add_argument('--data_dir', help='main data directory', type=str, required=False, default = 'path 2 directory that contain all training and validation data folders')
    parser.add_argument('--weight', help='The pretrained weight used as an input', type=str, required=False)
    parser.add_argument('--alpha', help = 'The alpha value for the unsupervised domain loss', default = 1, type=float, required=False)
    parser.add_argument('--beta', help = 'The beta value for the unsupervised instance loss', default = 10, type=float, required=False)
    parser.add_argument('--lr', help = 'Learning rate', default = 0.10, type=float, required=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argumentParser()
    target_folds = os.listdir(args.data_dir)   #
    base_weights ='path to model weight from supervised training on source datset' # which is optional
    source_folds = os.listdir(args.data_dir)
    for src_fold in source_folds:
        source_data = args.data_dir + '/' + src_fold # args.src_fold
        pretrained = base_weights.format(src_fold)
        print('Weight will be loaded from: ', pretrained)
        for tfold in target_folds:
            if src_fold !=tfold:
                target_data = args.data_dir + '/' + tfold
                out_dir = args.save_dir + '/' + src_fold + '/' + tfold
                print(f'Source: {src_fold} Target: {tfold}')
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                train_OT(args.config, out_dir, source_data, target_data, args.epochs, args.lr, args.alpha, args.beta, checkpoint=pretrained)
