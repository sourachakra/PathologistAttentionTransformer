
import argparse
import os
import random

import numpy as np
import cv2
import datetime
import torch
import torch.nn.functional as F

from sptransformer.builder import build
from common.config import JsonConfig
from common.losses import focal_loss
from common.losses import cc
from common.utils import (
    transform_fixations, )
from sptransformer.evaluation import evaluate
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
import gc
from memory_profiler import profile
from collections import OrderedDict
 
SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def parse_args():
    """Parse args."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams',
                        type=str,
                        help='hyper parameters config file path')
    parser.add_argument('--dataset-root', type=str, help='dataset root path')
    parser.add_argument('--model', choices=['HAT', 'FOM'], default='HAT', help='model type')
    parser.add_argument('--eval-only',
                        action='store_true',
                        help='perform evaluation only')
    parser.add_argument(
        '--split',
        type=int,
        default=1,
        help='dataset split for MIT1003/CAT2000 only (default=1)')
    parser.add_argument(
        '--eval-mode',
        choices=['greedy', 'sample'],
        type=str,
        default='greedy',
        help=
        'whether to sample scanapth or greedily predict scanpath during evaluation (default=greedy)'
    )
    parser.add_argument('--disable-saliency',
                        action='store_true',
                        help='do not calculate saliency metrics')

    parser.add_argument('--gpu-id',
                        type=int,
                        default=0,
                        help='gpu id (default=0)')
    return parser.parse_args()


def log_dict(writer, scalars, step, prefix):
    for k, v in scalars.items():
        writer.add_scalar(prefix + "/" + k, v, step)

        
def plot_scanpath(img, xs, ys, save_path):
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    cir_rad_min, cir_rad_max = 30, 60
    min_T, max_T = np.min(1), np.max(1)

    for i in range(len(xs)):
        if i > 0:
            plt.arrow(xs[i - 1], ys[i - 1], xs[i] - xs[i - 1],
                      ys[i] - ys[i - 1], width=3, color='yellow', alpha=0.5)

    for i in range(len(xs)):
        cir_rad = int(7) # + rad_per_T * (ts[i] - min_T))
        circle = plt.Circle((xs[i], ys[i]),
                            radius=cir_rad,
                            edgecolor='red',
                            facecolor='yellow',
                            alpha=0.5)
        ax.add_patch(circle)
        plt.annotate("{}".format(
            i+1), xy=(xs[i], ys[i]+3), fontsize=10, ha="center", va="center")

    ax.axis('off')
    plt.savefig(save_path,bbox_inches='tight',pad_inches=0)
    
    
def train_iter(model, optimizer, feat_path, batch, losses, loss_weights, loss_funcs, pa, num):
    assert len(losses) > 0, "no loss func assigned!"
    model.train()

    fix_losses, aux_losses = 0, 0

    task_ids = batch['task_id'].to(device)
    is_last = batch['is_last'].to(device)
    mag = batch['mag'].to(device)
    next_mag = batch['next_mag'].to(device)#.float()
    next_fix = batch['next_normalized_fixations']
    act_len = batch['act_len']
    
    if next_mag.dtype != torch.long:
        next_mag = next_mag.long()
    
    IOR_weight_map = batch['IOR_weight_map'].to(device)
    is_fv = batch['is_freeview'].to(device)
    not_fv = torch.logical_not(is_fv)

    for i in range(len(batch['img_name'])):
        name1 = batch['img_name'][i]

        patatt_2x = np.load(feat_path+'/2x/'+name1[:-4]+'-01Z-00-DX1.npy',allow_pickle=True)[0]
        
        patatt_10x = np.load(feat_path+'/10x/'+name1[:-4]+'-01Z-00-DX1.npy',allow_pickle=True)[0]
        
        
        low_res_embed2 = patatt_2x
        high_res_embed2 = patatt_10x
        
        low_res_embed2 = torch.from_numpy(low_res_embed2).unsqueeze(0)
        low_res_embed2 = torch.permute(low_res_embed2,(0,3,1,2))
        low_res_embed2 = F.interpolate(low_res_embed2, [10,16], mode='bilinear', align_corners=True) #[10,16]
        low_res_embed1 = torch.permute(low_res_embed2,(0,2,3,1))
        
        high_res_embed2 = torch.from_numpy(high_res_embed2).unsqueeze(0)
        high_res_embed2 = torch.permute(high_res_embed2,(0,3,1,2))
        high_res_embed2 = F.interpolate(high_res_embed2, [80,128], mode='bilinear', align_corners=True) #[80,128]
        high_res_embed1 = torch.permute(high_res_embed2,(0,2,3,1))

        if i == 0:
            low_res_embed = low_res_embed1
            high_res_embed = high_res_embed1
        else:
            low_res_embed = torch.cat((low_res_embed,low_res_embed1),0)
            high_res_embed = torch.cat((high_res_embed,high_res_embed1),0)
    
    inp_seq, inp_seq_high = transform_fixations(batch['normalized_fixations'],
                                                batch['is_padding'],
                                                hparams.Data,
                                                low_res_embed.shape,
                                                high_res_embed.shape,
                                                False,
                                                return_highres=True)
                                                
    inp_seq = inp_seq.to(device)
    inp_padding_mask = (inp_seq == pa.pad_idx)
    
    logits = model(inp_seq, inp_padding_mask, inp_seq_high.to(device), mag, batch['img_name'],low_res_embed,high_res_embed,next_mag, act_len)
    
    del low_res_embed,high_res_embed

    optimizer.zero_grad()

    bs = inp_seq.size(0)

    optimizer.zero_grad()
    loss_dict = {}
    if "next_fix_pred" in losses:
        # Next fixation prediction
        non_term_mask = True
        if non_term_mask:
            pred_fix_map = logits['pred_fixation_map'][torch.arange(bs), task_ids]
            
            if use_focal_loss:
                pred_fix_map = torch.sigmoid(pred_fix_map)

            tgt_fix_map = batch['target_fix_map'].to(device)
            
            pred_fix_map = pred_fix_map[non_term_mask]
            tgt_fix_map = tgt_fix_map[non_term_mask]

            
            loss_dict['next_fix_pred'] = loss_funcs['next_fix_pred'](
                pred_fix_map,
                tgt_fix_map,
                alpha=1,
                beta=4,
                weights=IOR_weight_map[non_term_mask])  

        else:
            loss_dict['next_fix_pred'] = 0

                                                         
    if "mag_pred" in losses:
        pred_magnification = logits['pred_magnification']
        pred_magnification = pred_magnification.view(-1, pred_magnification.size(-1))
        loss_dict['mag_pred'] = loss_funcs['mag_pred'](pred_magnification,next_mag)


    loss = 0
    for k, v in loss_dict.items():
        loss += v * loss_weights[k]
    #print('loss:',loss)
    loss.backward()
    
    optimizer.step()
    
    for k in loss_dict:
        try:
            loss_dict[k] = loss_dict[k].item()
        except:
            loss_dict[k] = loss_dict[k]

    del pred_magnification, pred_fix_map, loss, logits
    torch.cuda.empty_cache()
    gc.collect()
    
    return loss_dict


def run_evaluation(max_nss, max_auc,i_epoch,model):
    # Perform evaluation
    rst_tp, rst_ta, rst_fv = None, None, None
    pred_tp = pred_ta = pred_fv = None
    if hparams.Data.TAP in ['FV', 'ALL']:
        rst_fv = evaluate(
            max_nss, 
            max_auc,
            i_epoch,
            model,
            device,
            valid_gaze_loader_fv,
            hparams.Data,
            human_cdf,
            prior_maps_fv,
            sss_strings,
            dataset_root,
            sps_test_fv,
            sample_action=sample_action,
            output_saliency_metrics=output_saliency_metrics,
            center_initial=hparams.Data.name
            in ['COCO-Search18', 'COCO-Freeview'],
            log_dir=log_dir)
    return rst_tp, rst_ta, rst_fv, max_nss, max_auc

def log_memory_usage(step_description):
    print(f"--- {step_description} ---")
    print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")
    print(f"Reserved memory: {torch.cuda.memory_reserved() / (1024 ** 2)} MB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2)} MB")
    print(f"Max memory reserved: {torch.cuda.max_memory_reserved() / (1024 ** 2)} MB")
    print()

    
if __name__ == '__main__':
    args = parse_args()
    print(args.hparams)
    hparams = JsonConfig(args.hparams)
    hparams.Model.name = args.model
    dir = os.path.dirname(args.hparams)

    hparams_fv = JsonConfig(os.path.join(dir, 'coco_search18_FV.json'))
    
    dataset_root = args.dataset_root
    output_saliency_metrics = not args.disable_saliency #not 
    device = torch.device(f'cuda:{args.gpu_id}')
    sample_action = args.eval_mode == 'sample'
    if hparams.Data.name in ['MIT1003', 'CAT2000']:
        hparams.Train.log_dir += f'_split{args.split}'

    max_auc, max_nss = 0,0
            
    model, optimizer, train_gaze_loader, valid_gaze_loader_fv, \
        global_step, human_cdf, prior_maps_fv,  \
        sss_strings,  \
        sps_test_fv = build(
        hparams, dataset_root, device, args.eval_only, args.split)

    log_dir = hparams.Train.log_dir
    
    if args.eval_only:
        dataset_split_num = 1
        load_dict_path = './checkpoints/split'+str(dataset_split_num)+'.pt'
        state_dict = torch.load(load_dict_path)['model']
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k  # Add 'module.' prefix
            new_state_dict[name] = v
    
        model.load_state_dict(new_state_dict)
        run_evaluation(0,0,global_step,model)
    else:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        print("Log dir:", log_dir)
        log_folder_runs = "./runs/{}".format(log_dir.split('/')[-1])
        if not os.path.exists(log_folder_runs):
            os.system(f"mkdir -p {log_folder_runs}")

        # Write configuration file to the log dir
        hparams.dump(log_dir, 'config.json')

        print_every = 200
        dataset_split_num = 1
        feat_path = './input_features/split'+str(dataset_split_num)
        max_iters = hparams.Train.max_iters
        save_every = hparams.Train.checkpoint_every
        eval_every = hparams.Train.evaluate_every
        pad_idx = hparams.Data.pad_idx
        use_focal_loss = hparams.Train.use_focal_loss
        
        n_samples_per_class = torch.tensor([2615,6364,12545,14863,3148,364]) # for weighted cross entropy, data derived from the number of instances in collected dataset
        N = n_samples_per_class.sum()  # Total number of samples
        C = len(n_samples_per_class)   # Number of classes
        class_weights = N / (C * n_samples_per_class)
        class_weights = class_weights.cuda()

        loss_funcs = {
            "next_fix_pred":focal_loss if use_focal_loss else torch.nn.BCEWithLogitsLoss(),
            "mag_pred":torch.nn.CrossEntropyLoss(weight=class_weights), #weight=class_weights
        }

        loss_weights = {
            "next_fix_pred": 1.0,
            "mag_pred": hparams.Train.mag_pred_weight,
        }
        losses = hparams.Train.losses
        loss_dict_avg = dict(zip(losses, [0] * len(losses)))
        print("loss weights:", loss_weights)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=hparams.Train.lr_steps, gamma=0.1)

        s_epoch = int(global_step / len(train_gaze_loader))

        tot_len = len(train_gaze_loader)
        last_time = datetime.datetime.now()
        for i_epoch in range(s_epoch, int(1e5)):
            scheduler.step()
            for i_batch, batch in enumerate(train_gaze_loader):
                
                loss_dict = train_iter(model, optimizer, feat_path, batch, losses,loss_weights, loss_funcs, hparams.Data,i_batch)
                
                for k in loss_dict:
                    loss_dict_avg[k] += loss_dict[k]

                if global_step % print_every == print_every - 1:
                    for k in loss_dict_avg:
                        loss_dict_avg[k] /= print_every

                    time = datetime.datetime.now()
                    eta = str((time - last_time) / print_every *
                              (max_iters - global_step))
                    last_time = time
                    time = str(time)
                    log_msg = "[{}], eta: {}, iter: {}, progress: {:.2f}%, epoch: {}, total loss: {:.3f}".format(
                        time[time.rfind(' ') + 1:time.rfind('.')],
                        eta[:eta.rfind('.')],
                        global_step,
                        (i_batch / tot_len) * 100,
                        i_epoch,
                        np.sum(list(loss_dict_avg.values()))
                    )

                    for k, v in loss_dict_avg.items():
                        log_msg += " {}_loss: {:.3f}".format(k, v)

                    print(log_msg)

                    for k in loss_dict_avg:
                        loss_dict_avg[k] = 0

                torch.cuda.empty_cache()
                gc.collect()
    
                # Evaluate
                if global_step % eval_every == 0 and global_step > eval_every-2:
                    _, _, _, _, _ = run_evaluation(max_nss, max_auc,global_step,model)


                if global_step % save_every == 0:
                    save_path = os.path.join(log_dir, f"ckp_{global_step}.pt")
                    if isinstance(model, torch.nn.DataParallel):
                        model_weights = model.module.state_dict()
                    else:
                        model_weights = model.state_dict()
                    torch.save(
                        {
                            'model': model_weights,
                            'optimizer': optimizer.state_dict(),
                            'step': global_step + 1,
                        },
                        save_path,
                    )
                    print(f"Saved checkpoint to {save_path}.")
                global_step += 1
                if global_step >= max_iters:
                    print("Exit training!")
                    break
                    
                del loss_dict
            else:
                continue
            break  # Break outer loop

        # Copy to log file to ./runs
        os.system(f"cp {log_dir}/events* {log_folder_runs}")
