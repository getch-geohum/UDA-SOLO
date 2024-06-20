import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot, set_random_seed
from mmdet.models import build_detector
from mmdet.datasets import build_dataset, build_dataloader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from itertools import cycle
import torch
import numpy as np
import argparse
import json
import os
import copy

def base_line_trainer(CONFIG, checkpoint_dir,data_dir, epochs = 65):
    config = mmcv.Config.fromfile(CONFIG)
    config.model.bbox_head.num_classes = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # #     device ='cuda:0'
    model = build_detector(config.model)
    model.train()  # Convert the model into evaluation mode
    model.CLASSES = ('background', 'dwelling',)
    model.to(device)
    
    tconfig = copy.deepcopy(config.data.train)
    vconfig = copy.deepcopy(config.data.train)

    # change target and source data path
    tconfig['ann_file'] = data_dir + '/train/train_annotations.json'
    tconfig['img_prefix'] = data_dir + '/train/'

    vconfig['ann_file'] = data_dir + '/valid/valid_annotations.json'
    vconfig['img_prefix'] = data_dir + '/valid/'


    datasets = build_dataset(tconfig)
    loader = build_dataloader(datasets, config.data.imgs_per_gpu, config.data.workers_per_gpu, dist=False)

    
    v_datasets = build_dataset(vconfig)
    v_loader = build_dataloader(v_datasets,config.data.imgs_per_gpu, config.data.workers_per_gpu, dist=False)

    optimizer = optim.SGD(model.parameters(),lr=0.001, momentum=0.9, weight_decay=0.0001)
    ml_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 13], 0.1)
    scaler = GradScaler()

    ins = []
    cat = []
    tot = []
    valid_loss = []
    step = []
    epoch = []
    best_loss = 0.0

    for j in range(1, epochs+1):
        for i, data in enumerate(loader):
            img_meta = data['img_meta']
            imeta = img_meta.data[0]   #metdata information after data transformed

            gt_bboxes = data['gt_bboxes']
            bbox = gt_bboxes.data[0]
            bbox[0] = bbox[0].to(device)
            bbox[1] = bbox[1].to(device)

            gt_labels = data['gt_labels']
            gt_lbl = gt_labels.data[0]
            gt_lbl[0] = gt_lbl[0].to(device)
            gt_lbl[1] = gt_lbl[1].to(device)

            gt_masks = data['gt_masks']
            gt_msk = gt_masks.data[0]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                _, b, c  = model.forward_train(data['img'].data[0].to(device)) # forwar pass results in category pred and kernel pred
                loss_inputs = b + (c, bbox, gt_lbl, gt_msk, imeta, model.train_cfg)
                losses = model.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=None)
            tot_loss = losses['loss_ins'] + losses['loss_cate']
            cat, ins = losses['ins'], losses['cat']
            print('+++', cat.shape, ins.shape, '+++')
            scaler.scale(tot_loss).backward() # scale the loss to 16bit float
            scaler.unscale_(optimizer)   # return the sacel to normal value
            clip_grad_norm_(model.parameters(), max_norm=35, norm_type=2.0)   # gradient clip, to reduce explosion or vanishing
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                ins_loss = np.round(losses['loss_ins'].cpu().numpy(), 4)
                cat_loss = np.round(losses['loss_cate'].cpu().numpy(), 4)
                tot_loss = np.round(tot_loss.cpu().numpy(), 4)

                ins.append(ins_loss)
                cat.append(cat_loss)
                tot.append(tot_loss)
                step.append(i)
                epoch.append(j)
                
                if j == 1:
                    vv = 0
                else:
                    vv = valid_loss[-1]

                print(f'epoch: {j} iter: {i} cat_loss: {cat_loss} ins_loss: {ins_loss} loss: {tot_loss}, vlos: {vv}')
        ml_scheduler.step() # multiscale scheduler

        v_loss = 0
        control = 0

        with torch.no_grad():
            for v_data in v_loader:
                v_img_meta = v_data['img_meta']
                v_imeta = v_img_meta.data[0]   #metdata information after data transformed

                v_gt_bboxes = v_data['gt_bboxes']
                v_bbox = v_gt_bboxes.data[0]
                v_bbox[0] = v_bbox[0].to(device)
                v_bbox[1] = v_bbox[1].to(device)

                v_gt_labels = v_data['gt_labels']
                v_gt_lbl = v_gt_labels.data[0]
                v_gt_lbl[0] = v_gt_lbl[0].to(device)
                v_gt_lbl[1] = v_gt_lbl[1].to(device)

                v_gt_masks = v_data['gt_masks']
                v_gt_msk = v_gt_masks.data[0]

                _, bb, cc  = model.forward_train(v_data['img'].data[0].to(device))
                loss_input_valid = bb + (cc, v_bbox, v_gt_lbl, v_gt_msk, v_imeta, model.train_cfg)
                val_loss = model.bbox_head.loss(*loss_input_valid, gt_bboxes_ignore=None, return_raw=False)
                v_loss+=val_loss['loss_ins'].cpu().numpy() + val_loss['loss_cate'].cpu().numpy()
                control+=1

            mean_valid_loss = v_loss/control
            valid_loss.append(mean_valid_loss)
            if j == 1:
                best_loss = mean_valid_loss
                name = 'checkpoint.pth'
                chpoint = os.path.join(checkpoint_dir, name)
                torch.save(model.state_dict(), chpoint) # save the checkpoint
            elif j>1:
                if mean_valid_loss < best_loss:
                    best_loss = mean_valid_loss
                    name = 'checkpoint.pth'
                    chpoint = os.path.join(checkpoint_dir, name)
                    torch.save(model.state_dict(), chpoint) # best checkpoint
                else:
                    pass

    metric = dict(STEP = step, EPOCH = epoch, INS=ins, CAT=cat, TOT=tot, VAL=valid_loss)
    save_name = os.path.join(checkpoint_dir, 'summary.json')
    with open(save_name, 'w') as fp:
        json.dump(str(metric), fp)


def argumentParser():
    parser = argparse.ArgumentParser(description="region mixed model trainer")
    parser.add_argument("--config", help="directory for model configuration",type=str default='./SOLO/configs/solov2/train_configs.py')
    parser.add_argument('--save_dir', help="directory to save the checkpoint", type=str, default='/')
    parser.add_argument('--epochs', help='The number of epochs to train the model', default=65, type=int, required=False)
    parser.add_argument('--data_dir', help='The data directory', type=str, required=False default='/')
    parsed = parser.parse_args()
    return parsed 


if __name__ == '__main__':
    args = argumentParser()
    folders = sorted(os.listdir(args.data_dir)) 
    for folder in folders:
        data_root = args.data_dir + '/' + folder
        out_dir = args.save_dir + '/' + folder
        print(f'processing for folder{folder}')
        for i in range(1, 2):
            freq_fold = '0' + str(i)
            final_out_dir = out_dir + '/' + freq_fold
            if not os.path.exists(final_out_dir):
                os.makedirs(final_out_dir, exist_ok=True)
            base_line_trainer(args.config, final_out_dir, data_root, args.epochs)
