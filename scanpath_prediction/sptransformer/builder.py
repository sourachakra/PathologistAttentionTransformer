import sys

sys.path.append('../common')

from common.dataset import process_data
from .models import Im2SpDenseTransformer
from common.config import JsonConfig
from common.utils import get_prior_maps, cutFixOnTarget
import json
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader
import types


def build(hparams, dataset_root, device, is_eval=False, split=1):
    dataset_name = hparams.Data.name

    # bounding box of the target object (for search efficiency evaluation)
    bbox_annos = np.load(
        join(dataset_root, 'bbox_annos.npy'),allow_pickle=True).item() if dataset_name == 'COCO-Search18' else {}

    # load ground-truth human scanpaths
    if dataset_name in ['WSI']:
        with open(
                join(dataset_root,'all_WSIs_fix_data_standarddim2_recent_1.json'), 'r'
        ) as json_file:
            human_scanpaths = json.load(json_file)

        n_tasks = 1
        #human_scanpaths = list(filter(lambda x: x['correct'] == 1, human_scanpaths))  
        #human_scanpaths = {k: v for k, v in human_scanpaths.items() if v['expertise'] == 2 and v['correct'] == 1}
        human_scanpaths = {k: v for k, v in human_scanpaths.items() if v['correct'] == 1}
    else:
        print(f"dataset {dataset_name} not supported!")
        raise NotImplementedError
        
    human_scanpaths_all = human_scanpaths
    print('len scanpath:',len(human_scanpaths_all))
    human_scanpaths_fv = human_scanpaths

    n_tasks = 1

    if hparams.Data.subject > -1:
        print(f"excluding subject {hparams.Data.subject} data!")
        human_scanpaths = list(
            filter(lambda x: x['subject'] != hparams.Data.subject,
                   human_scanpaths))

    # process fixation data
    dataset = process_data(
        human_scanpaths,
        dataset_root,
        bbox_annos,
        hparams,
        human_scanpaths_all,
        sample_scanpath=False,
        use_coco_annotation="centermap_pred" in hparams.Train.losses
        and (not is_eval))

    batch_size = hparams.Train.batch_size
    n_workers = hparams.Train.n_workers

    train_HG_loader = DataLoader(dataset['gaze_train'],
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=n_workers,
                                 drop_last=True,
                                 pin_memory=True)
    print('num of training batches =', len(train_HG_loader))

    train_img_loader = DataLoader(dataset['img_train'],
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=n_workers,
                                  drop_last=True,
                                  pin_memory=True)
    valid_img_loader_FV = DataLoader(dataset['img_valid_FV'],
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=n_workers,
                                     drop_last=False,
                                     pin_memory=True)
    valid_HG_loader_FV = DataLoader(dataset['gaze_valid_FV'],
                                    batch_size=1, #*2
                                    shuffle=False,
                                    num_workers=n_workers,
                                    drop_last=False,
                                    pin_memory=True)

    # Create model
    emb_size = hparams.Model.embedding_dim
    n_heads = hparams.Model.n_heads
    hidden_size = hparams.Model.hidden_dim
    tgt_vocab_size = hparams.Data.patch_count + len(
        hparams.Data.special_symbols)
    if hparams.Train.use_sinkhorn:
        assert hparams.Model.separate_fix_arch, "sinkhorn requires the model to be separate!"

    if hparams.Model.name == 'HAT':
        model = Im2SpDenseTransformer(
            hparams.Data,
            num_decoder_layers=hparams.Model.n_dec_layers,
            hidden_dim=emb_size,
            nhead=n_heads,
            ntask=n_tasks,
            tgt_vocab_size=tgt_vocab_size,
            num_output_layers=hparams.Model.num_output_layers,
            separate_fix_arch=hparams.Model.separate_fix_arch,
            train_encoder=hparams.Train.train_backbone,
            use_dino=hparams.Train.use_dino_pretrained_model,
            dropout=hparams.Train.dropout,
            dim_feedforward=hidden_size,
            parallel_arch=hparams.Model.parallel_arch,
            dorsal_source=hparams.Model.dorsal_source,
            num_encoder_layers=hparams.Model.n_enc_layers,
            output_centermap="centermap_pred" in hparams.Train.losses,
            output_saliency="saliency_pred" in hparams.Train.losses,
            output_target_map="target_map_pred" in hparams.Train.losses)
    else:
        print(f"No {hparams.Model.name} model implemented!")
        raise NotImplementedError
        
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=hparams.Train.adam_lr,
                                  betas=hparams.Train.adam_betas)

    # Load weights from checkpoint when available
    if len(hparams.Model.checkpoint) > 0:
        ckp = torch.load(join(hparams.Train.log_dir, hparams.Model.checkpoint))
        model.load_state_dict(ckp['model'])
        optimizer.load_state_dict(ckp['optimizer'])
        global_step = ckp['step']
        print(f"loaded weights from {hparams.Model.checkpoint}.")
    else:
        global_step = 0

    if hparams.Train.parallel:
        model = torch.nn.DataParallel(model)

    human_cdf = dataset['human_cdf']

    if len(human_scanpaths_fv) > 0:
        prior_maps_fv = get_prior_maps(human_scanpaths_fv, hparams.Data.im_w,
                                       hparams.Data.im_h)
        keys = list(prior_maps_fv.keys())
        for k in keys:
            prior_maps_fv[k] = torch.tensor(prior_maps_fv.pop(k)).to(device)

        for k in keys:
            prior_maps_fv[k] = prior_maps_fv['all']
    else:
        prior_maps_fv = None

    if dataset_name == 'COCO-Search18' or dataset_name == 'COCO-Freeview':
        sss_strings = np.load(join(dataset_root, hparams.Data.sem_seq_dir,
                                   'test.pkl'),
                              allow_pickle=True)
    else:
        sss_strings = None

    human_scanpaths_fv = list(human_scanpaths_fv.values())
    sps_test_fv = list(
        filter(lambda x: x['split'] == 'test', human_scanpaths_fv))

    is_lasts = [x[5] for x in dataset['gaze_train'].fix_labels]

    return (model, optimizer, train_HG_loader, valid_HG_loader_FV, 
            global_step, human_cdf, 
            prior_maps_fv, sss_strings, 
            sps_test_fv) 
