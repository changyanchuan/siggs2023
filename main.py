# temporarily added for experiments, need polish

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import gc
import sys
import time
import logging
import argparse
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

from config import Config
import tools
from model.unet_model import UNet
from model.u2net import U2NETP, U2NET
from model.loss import DiceLoss
from data_loader import read_datasets, PairDataset
from data_loader import read_test_dataset, output_testresults


'''
nohup python  &> result &
'''


bce_loss = torch.nn.BCELoss(size_average=True).to(Config.device)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)
	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	return loss0, loss

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn


def train_model(ds_train, ds_eval):
    _time_func = time.time()
    logging.info("[Training] Start. @={:.0f}".format(_time_func))
    
    dl_train = DataLoader(ds_train, batch_size = Config.training_batch_size, shuffle = True)
    
    if Config.backbone_str == 'unet':
        model = UNet(n_channels = 3, n_classes = 2)
    elif Config.backbone_str == 'u2net':
        model = U2NET()
    model.to(Config.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = Config.training_lr, weight_decay = Config.training_weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = Config.training_lr_degrade_step, gamma = Config.training_lr_degrade_gamma)
    
    if Config.backbone_str == 'unet':
        criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor([1.0, Config.training_cel_weight]).to(Config.device))
        dice_loss = DiceLoss().to(Config.device)
        
    best_loss_train = 1e10
    best_epoch = 0
    best_model_state_dict = None
    bad_counter = 0
    bad_patience = Config.training_bad_patience
    
    for i_ep in range(Config.training_epochs):
        _time_ep = time.time()
        loss_ep, gpu_ep, ram_ep = [], [], []
        optimizer.zero_grad()
        
        model.train()
        
        _time_batch = time.time()
        for i_batch, batch in enumerate(dl_train):
            optimizer.zero_grad()
            
            imgs, labels = batch
            imgs = imgs.to(Config.device) # [n, 3, h, w]
            labels = labels.squeeze(1).to(Config.device)  # [n, h, w]

            pred = model(imgs)
            if Config.backbone_str == 'unet':
                    loss = criterion(pred, labels)
                    bce_weight = Config.training_celdice_loss_weight
                    dlose = dice_loss(F.softmax(pred, dim=1).float(),
                                        F.one_hot(labels, 2).permute(0, 3, 1, 2).float())
                    loss = loss * bce_weight + (1 - bce_weight) * dlose
            elif Config.backbone_str == 'u2net':
                _, loss = muti_bce_loss_fusion(*pred, labels.float().unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())
            gpu_ep.append(tools.GPUInfo.mem()[0])
            ram_ep.append(tools.RAMInfo.mem())
            
            if i_batch % 100 == 0 and i_batch:
                logging.info("[Training-Batch END]  ep-batch={}-{}, loss={:.3f}, @={:.2f}, gpu={}, ram={}" \
                            .format(i_ep, i_batch, loss.item(), time.time() - _time_batch,
                                    tools.GPUInfo.mem(), tools.RAMInfo.mem()))

        scheduler.step() # decay before optimizer when pytorch < 1.1

        loss_ep_avg = tools.mean(loss_ep)
        f1 = eval_model(model, ds_eval)
        logging.info("[Training-Ep END] ep={}: train_loss={:.3f}, eval_f1={:.3f}, @={:.2f}, gpu={:.0f}, ram={:.0f}" \
                    .format(i_ep, loss_ep_avg, f1, time.time() - _time_ep, 
                            tools.mean(gpu_ep), tools.mean(ram_ep)))

        # early stopping
        if loss_ep_avg < best_loss_train:
            best_epoch = i_ep
            best_loss_train = loss_ep_avg
            bad_counter = 0
            best_model_state_dict = model.state_dict()
        else:
            bad_counter += 1

        if bad_counter == bad_patience or (i_ep + 1) == Config.training_epochs:
            torch.save({'model_state_dict': best_model_state_dict}, Config.checkpoint_file)
            model.load_state_dict(best_model_state_dict)
            logging.info("[Training] END! @={:.2f}, best_epoch={}, best_loss_train={:.6f}" \
                        .format(time.time() - _time_func, best_epoch, best_loss_train))
            break
            
    return model


@torch.no_grad()
def eval_model(model, ds_eval):
    logging.debug("[Eval] Start.")
    dl_eval = DataLoader(ds_eval, batch_size = Config.training_batch_size, shuffle = True)
    
    model.eval()
    
    _time_batch = time.time()
    lst_preds = []
    lst_labels = []
    for i_batch, batch in enumerate(dl_eval):

        imgs, labels = batch
        imgs = imgs.to(Config.device) # [n, 3, h, w]
        labels = labels.squeeze(1).cpu().detach().numpy()  # [n, h, w]
        lst_labels.append(labels)

        if Config.backbone_str == 'unet':
            pred = model(imgs) # [n, 2, h, w]
            pred = torch.argmax(pred, dim = 1).cpu().detach().numpy() # [n, h, w]
        elif Config.backbone_str == 'u2net':
            pred = normPRED(model(imgs)[0]) # [n, 1, h, w]
            pred[pred >= Config.training_u2net_threshold] = 1
            pred[pred < Config.training_u2net_threshold] = 0
            pred = pred.cpu().detach().numpy()
            
        lst_preds.append(pred)
        
    f1 = tools.f1(np.concatenate(lst_labels).flatten(), 
                    np.concatenate(lst_preds).flatten())
    
    
    logging.debug("[Eval] Done. @={:.2f}, f1={:.3f}" \
                .format(time.time() - _time_batch, f1))
    return f1


@torch.no_grad()
def test(model, dic_test):
    logging.debug("[Test] Start.")
    _time = time.time()
    # dl_test = DataLoader(ds_test, batch_size = Config.training_batch_size, shuffle = False)
    
    model.eval()
    
    dic_preds = {}
    
    for tif_filename, dic_regions in dic_test.items():
        dic_preds[tif_filename] = {}
        for region_id, region_meta in dic_regions.items():
            lst_img = region_meta[0] # list of [3, sidelen, sidelen]
            img_region_rows, img_region_cols = region_meta[1], region_meta[2]
            dl_test = DataLoader(lst_img, batch_size = Config.training_batch_size, shuffle = False)
            
            lst_preds = []
            for i_batch, batch in enumerate(dl_test):
                imgs = batch
                imgs = imgs.to(Config.device) # [n, 3, h, w]

                if Config.backbone_str == 'unet':
                    preds = model(imgs) # [n, 2, h, w]
                    preds = torch.argmax(preds, dim = 1).cpu().detach() # [n, h, w]
                elif Config.backbone_str == 'u2net':
                    preds = normPRED(model(imgs)[0]).squeeze(1) # [n,  h, w]
                    preds[preds >= Config.training_u2net_threshold] = 1
                    preds[preds < Config.training_u2net_threshold] = 0
                    preds = preds.to(dtype = torch.uint8).cpu().detach()

                lst_preds.append(preds)
            
            preds = torch.cat(lst_preds, 0).unsqueeze(1)  # [big n, 1, h, w]
            preds = tools.fold4D_depadding(preds, Config.image_side_len, \
                                            img_region_rows[1]-img_region_rows[0]+1, \
                                            img_region_cols[1]-img_region_cols[0]+1) # [1, 1, realheight, realwidth]
            preds = preds.squeeze(0).squeeze(0) # [realheight, realwidth]
            assert preds.shape == (img_region_rows[1]-img_region_rows[0]+1, img_region_cols[1]-img_region_cols[0]+1)
            dic_preds[tif_filename][region_id] = (preds, img_region_rows, img_region_cols)

    logging.info("[Test] Done. @={:.2f}, #regions={}, #imgs={}".format( \
                    time.time() - _time, sum([len(v) for v in dic_preds.values()]),
                    sum([len(vv[0]) for v in dic_preds.values() for vv in v.values()])  ))
    return dic_preds



def parse_args():
    parser = argparse.ArgumentParser(description = '#TODO')
    
    parser.add_argument('--dumpfile_uniqueid', type = str, help = '') # see config.py
    parser.add_argument('--seed', type = int, help = '')
    parser.add_argument('--debug', dest = 'debug', action='store_true')
    parser.add_argument('--load_checkpoint', dest = 'load_checkpoint', action='store_true')
    
    parser.add_argument('--backbone_str', type = str, help = '') # see config.py
    parser.add_argument('--training_cel_weight', type = int, help = '')
    parser.add_argument('--training_celdice_loss_weight', type = float, help = '')
    parser.add_argument('--training_u2net_threshold', type = float, help = '')
    parser.add_argument('--image_augment', dest = 'image_augment', action='store_true')
    
    
    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))


if __name__ == '__main__':
    Config.update(parse_args())

    logging.basicConfig(level = logging.DEBUG if Config.debug else logging.INFO,
            format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
            handlers = [logging.FileHandler(Config.root_dir+'/exp/log/'+tools.log_file_name(), mode = 'w'), 
                        logging.StreamHandler()] )

    logging.info('python ' + ' '.join(sys.argv))
    logging.info('=================================')
    logging.info(Config.to_str())
    logging.info('=================================')
    
    _time = time.time()
    
    # 1. prepare data and dataloader
    # 2. train the model
    # 3. label the test dataset, and dump to file 
    model = None
    if not Config.load_checkpoint:
        lst_img, lst_label = read_datasets()
        n = len(lst_img)
        lst_img_train, lst_img_eval = lst_img[:int(n*0.8)], lst_img[int(n*0.8):]
        lst_label_train, lst_label_eval = lst_label[:int(n*0.8)], lst_label[int(n*0.8):]
        ds_train = PairDataset(lst_img_train, lst_label_train)
        ds_eval = PairDataset(lst_img_eval, lst_label_eval)
        
        model = train_model(ds_train, ds_eval)
        
        del lst_img_train, lst_img_eval, lst_label_train, lst_label_eval, ds_train, ds_eval
        gc.collect()
        
    else: # skip model training. load from checkpoint
        logging.info('Load model from cp file.')
        if Config.backbone_str == 'unet':
            model = UNet(n_channels = 3, n_classes = 2)
        elif Config.backbone_str == 'u2net':
            model = U2NET()
        cp = torch.load(Config.checkpoint_file)
        model.load_state_dict(cp['model_state_dict'])
        model.to(Config.device)

    dic_test = read_test_dataset()
    dic_preds = test(model, dic_test)
    output_testresults(dic_preds)
    
    logging.info('All done. @={:.2f}'.format(time.time() - _time))

