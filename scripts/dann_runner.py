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

# domain classifier
class DomainNetL(nn.Module):
    def __init__(self, input_dim=2048):
        super(DomainNetL, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv2d(self.input_dim, 256, 3, 1) # input, outpu, stride
        self.conv2 = nn.Conv2d(256, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 64, 3, 1)
        self.fc1 = nn.Linear(5184, 128)
        self.fc2 = nn.Linear(128, 2)
        self.pooling = nn.MaxPool2d(2)
        self.softMax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softMax(x)
        return x


class DomainNetS(nn.Module):
    def __init__(self, input_dim=2048):
        super(DomainNetS, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv2d(self.input_dim, 256, 3, 1) # input, outpu, stride
        self.conv2 = nn.Conv2d(256, 128, 3, 1)
        self.fc1 = nn.Linear(128, 2)
        self.pooling = nn.MaxPool2d(2)
        self.softMax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.softMax(x)
        return x


# gradient reversal layer
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# step wise lamda compute
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

# step wise lamda compute
def compute_lamda_epochwise(d_epoch, n_epoch):
    '''
    n_epoch: total number of training epochs
    dstep: trainning step within a single epoch
    d_epoch: specific epoch within training epochs
    len_loader: length of data loader
    '''
    p = float(d_epoch) / float(n_epoch)
    alpha = 2. / (1. + np.exp(-10 * p)) - 1

    return alpha

# step wise learning rate adjuster
def compute_adapt_lr_stepwise(muno_lr, n_epoch, dstep, d_epoch, len_loader):
    p = float(dstep + d_epoch * len_loader) / n_epoch / len_loader
    step_lr = muno_lr/(1+10*p)**0.75
    return step_lr

def compute_adapt_lr_epochwise(muno_lr, n_epoch, d_epoch):
    p = float(d_epoch) / float(n_epoch)
    step_lr = muno_lr/(1+10*p)**0.75
    return step_lr

# Domain target generator
def domain_target(batch_size=(2,2), soure_first=True):
    '''
    Generates dimain labels for adversarial learning of domains
    batch_size: The batch size per terget or source
    soure_first: The arrangement which come first
    '''
    source = torch.zeros(batch_size[0])
    target = torch.ones(batch_size[0])

    if soure_first:
        return torch.cat([source, target]).long()
    else:
        return torch.cat([target, source]).long()

def train_DNN(Config, checkpoint_dir, source_dir, target_dir, epochs=65, learning_rate=0.001,checkpoint=None): # resume_from=None
    torch.cuda.empty_cache() # to freed memory catch especially for batch training
    config = mmcv.Config.fromfile(Config)
    config.model.bbox_head.num_classes = 2 # mae sure nclass = classes +1 (dwelling  +1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # #     device ='cuda:0'
    model = build_detector(config.model)
    model.CLASSES = ('dwelling',) # ('background', 'dwelling',)
    
    model.load_state_dict(torch.load(checkpoint)["state_dict"])
    print(f'The weight was loaded from: {checkpoint}')
    model.train()  # Convert the model into evaluation mode
    model.to(device)

    aconfig = copy.deepcopy(config.data.train)
    bconfig = copy.deepcopy(config.data.test) # only collects the images
    cconfig = copy.deepcopy(config.data.train)

    # change target and source data path
    aconfig['ann_file'] = source_dir + '/train/train_annotations.json'
    aconfig['img_prefix'] = source_dir + '/train/'

    bconfig['ann_file'] = target_dir + '/train/train_annotations.json'
    bconfig['img_prefix'] = target_dir + '/train/'

    cconfig['ann_file'] = source_dir + '/valid/valid_annotations.json'
    cconfig['img_prefix'] = source_dir + '/valid/'

    # build individual datasets
    datasetA = build_dataset(aconfig)
    loaderA = build_dataloader(datasetA, config.data.imgs_per_gpu, config.data.workers_per_gpu, dist=False)   # source data with labels
    datasetB = build_dataset(bconfig)
    loaderB = build_dataloader(datasetB, config.data.imgs_per_gpu, config.data.workers_per_gpu, dist=False)    # target data  labels
    datasetC = build_dataset(cconfig)
    loaderC = build_dataloader(datasetC, config.data.imgs_per_gpu, config.data.workers_per_gpu, dist=False)   # validation data

    scaler = GradScaler()
    domain_classifier = DomainNetS() # DomainNetL()
    domain_classifier.to(device)
    domain_loss = torch.nn.NLLLoss()  # loss for final regression layer
    domain_loss.to(device)

    optimizer = optim.SGD([{'params':model.parameters()},{'params':domain_classifier.parameters()}],lr=learning_rate, momentum=0.9, weight_decay=0.0001)

    learning_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,30], gamma=0.1)

    ins_ls = []        # instance loss
    cat_ls = []        # category loss
    tot_uls = []    # total sypervised loss
    d_ls = []       # domain loss
    aln_ls = []
    tot_ls = []     # grand loss
    valid_loss = [] # validation loss
    step = []
    epoch = []
    max_data_length = max(len(loaderA), len(loaderB))   # used for computing alpha
    #if restart_epoch == 1:
        #start = start = 1
    #else:
        #start = restart_epoch
        #print(f'training started from epoch {start}')
    for i in range(1, epochs+1):
        for j, (da, db) in enumerate(zip(cycle(loaderA), loaderB)):
            img_meta = da['img_meta']
            imeta = img_meta.data[0]   # metdata information after data transformed

            gt_bboxes = da['gt_bboxes']
            bbox = gt_bboxes.data[0]
            bbox[0] = bbox[0].to(device)
            bbox[1] = bbox[1].to(device)

            gt_labels = da['gt_labels']
            gt_lbl = gt_labels.data[0]
            gt_lbl[0] = gt_lbl[0].to(device)
            gt_lbl[1] = gt_lbl[1].to(device)

            gt_masks = da['gt_masks']
            gt_msk = gt_masks.data[0]
            domain_labels = domain_target((da['img'].data[0].shape[0], db['img'][0].shape[0]))
            domain_labels.to(device)

            lamda_step = compute_lamda_stepwise(n_epoch=epochs, dstep=j, d_epoch=i, len_loader=max_data_length)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                a, b, c  = model.forward_train(da['img'].data[0].to(device)) # forwar pass results in category pred and kernel pred
                d, e, f  = model.forward_train(db['img'][0].to(device)) # forward pass target
                loss_inputs = b + (c, bbox, gt_lbl, gt_msk, imeta, model.train_cfg)
                sup_loss = model.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=None, return_raw=False)  # supervised segmenter loss  # return_raw=False

                feat_a = a[-1]
                feat_b = d[-1]
                features = torch.cat([feat_a, feat_b])
                rev_feat = ReverseLayerF.apply(features, lamda_step) # gradient reversal and step wise lamda parameters to be implemented 
                domain_classifier.input_dim = features.shape[1]
                domain_logs = domain_classifier(rev_feat)
                d_loss = domain_loss(domain_logs, domain_labels.to(device))

            sp_tot_loss = sup_loss['loss_ins'] + sup_loss['loss_cate']

            total_loss = sp_tot_loss + lamda_step*d_loss  # lamda is weight for domain loss
            scaler.scale(total_loss).backward() # scale back the loss to 32 bit floating points
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=35, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()
            learning_scheduler.step()

            with torch.no_grad():
                ins_loss = np.round(sup_loss['loss_ins'].item(),6)
                cat_loss = np.round(sup_loss['loss_cate'].item(),6)
                tot_loss = np.round(sp_tot_loss.item(),6)
                tot_dm_loss = np.round(d_loss.item(),6)
                commul_loss = np.round(total_loss.item(),6)

                ins_ls.append(ins_loss)
                cat_ls.append(cat_loss)
                tot_uls.append(tot_loss)
                d_ls.append(tot_dm_loss)
                tot_ls.append(commul_loss)
                epoch.append(i)
                step.append(j)
            if i == 1:
                vls = 0
            else:
                vls = valid_loss[-1]
            print(f'epoch: {i},step: {j}, cat_ls: {cat_loss}, ins_ls: {ins_loss}, us_ls: {tot_loss}, dmn_ls: {tot_dm_loss}, tot_ls: {commul_loss}, valid_los: {vls}')


        v_loss = 0.0
        control = 0.0
        for m, valid_data in enumerate(loaderC): # validate the model
            with torch.no_grad():
                img_meta_v = valid_data['img_meta']
                imeta_v = img_meta_v.data[0]
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
                val_loss = model.bbox_head.loss(*loss_input_valid, gt_bboxes_ignore=None)
                v_loss+=val_loss['loss_ins'].cpu().numpy() + val_loss['loss_cate'].cpu().numpy()
                control+=1

            mean_valid_loss = v_loss/control
            valid_loss.append(mean_valid_loss)

        
        if i == 1:
            name = 'checkpoint.pth'
            chpoint = os.path.join(checkpoint_dir, name)
            torch.save(model.state_dict(), chpoint) # save the initial checkpoint
        elif valid_loss[-1] <= valid_loss[-2]:
            name = 'checkpoint.pth'
            chpoint = os.path.join(checkpoint_dir, name)
            torch.save(model.state_dict(), chpoint) # save the last check if validation results is better than previous epochs

    metric = dict(STEP = step, EPOCH = epoch, INS_L=ins_ls, CAT_L=cat_ls, TOT_ULS=tot_uls, DMN_L=d_ls, TOT_L=tot_ls, VAL_LS=valid_loss)  ## needs re-adjustment
    save_name = os.path.join(checkpoint_dir, 'summary.json')
    if os.path.exists(save_name):
        save_name = os.path.join(checkpoint_dir, 'summary_copy.json')
    with open(save_name, 'w') as fp:
        json.dump(str(metric), fp)

def argumentParser():
    parser = argparse.ArgumentParser(description="Sample mixed model trainer parameters")
    parser.add_argument("--config", help="Directory for model configuration",type=str, default = 'path 2 training configuration file')
    parser.add_argument('--save_dir', help="Directory to save the checkpoint", type=str, default = 'path 2 save logs')
    parser.add_argument('--epochs', help='number of epochs', type=int, default=65)
    parser.add_argument('--data_dir', help='main data directory', type=str, required=False, default = 'path 2 a folder containing all data')
    parser.add_argument('--lr', help='The initial learning rate for domain classifier', default=0.00125, type=float, required=False)
    parser.add_argument('--weight', help='The pretrained weight used as an input', type=str, required=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argumentParser()  # '/home/getch/ssl/UDA/OUTS/nguyen_june_2018/epoch_65.pth' 
    target_folds = os.listdir(args.data_dir)   #
    base_weights = 'path 2 supervised weight on source datset'  # optional
    source_folds = os.listdir(args.data_dir)
    for src_fold in source_folds:
        source_data = args.data_dir + '/' + src_fold # args.src_fold
        pretrained = base_weights.format(src_fold)
        print('Weight will be loaded from: ', pretrained)
        for tfold in target_folds:
            target_data = args.data_dir + '/' + tfold
            out_dir = args.save_dir + '/' + src_fold + '/' + tfold
            if not os.path.exists(final_out_dir):
                os.makedirs(final_out_dir, exist_ok=True)
            train_DNN(args.config, out_dir, source_data, target_data, args.epochs, args.lr, checkpoint = pretrained)

