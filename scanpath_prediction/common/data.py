import torch
import torchvision.transforms as T
from . import utils
from detectron2.data.detection_utils import read_image
from .coco_det import COCODetHelper
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import scipy.ndimage as filters
from os import listdir
from os.path import join
import warnings
import torch.utils.data
import torchvision
import torch.multiprocessing
import cv2
import random
from .cort_magnif_tfm import (radial_exp_isotrop_gridfun,
                              radial_quad_isotrop_gridfun,
                              img_cortical_magnif_tsr)



class FFN_IRL(Dataset):
    def __init__(self,
                 root_dir,
                 initial_fix,
                 img_info,
                 annos,
                 transform,
                 pa,
                 #catIds,
                 coco_annos=None):
        self.img_info = img_info
        self.root_dir = root_dir
        self.img_dir = join(root_dir, 'images')
        self.transform = transform
        self.pa = pa
        self.bboxes = annos
        self.initial_fix = initial_fix
        #self.catIds = catIds
        # self.prior_maps = torch.load(join(root_dir, pa.prior_maps_dir))
        if self.pa.use_DCB_target:
            self.coco_thing_classes = np.load(join(root_dir,
                                                   'coco_thing_classes.npy'),
                                              allow_pickle=True).item()
        if coco_annos:
            self.coco_helper = COCODetHelper(coco_annos)
        else:
            self.coco_helper = None
        self.fv_tid = 0 if self.pa.TAP == 'FV' else len(self.catIds)

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        imgId = self.img_info[idx]
        img_name, condition = imgId.split('*')
        if imgId in self.bboxes.keys():
            bbox = self.bboxes[imgId]
        else:
            bbox = None

        im_path = "{}/{}".format(self.img_dir, img_name)

        im = Image.open(im_path[:-4]+'-01Z-00-DX1.png').convert('RGB')
        im_tensor = self.transform(im)

        if bbox is not None:
            coding = utils.multi_hot_coding(bbox, self.pa.patch_size,
                                            self.pa.patch_num)
            coding = torch.from_numpy(coding).view(1, -1)
        else:
            coding = torch.zeros(1, self.pa.patch_count)

        # create action mask
        action_mask = np.zeros(self.pa.patch_num[0] * self.pa.patch_num[1],
                               dtype=np.uint8)

        is_fv = condition == 'freeview'
        ret = {
            'task_id': self.fv_tid if is_fv else self.catIds[cat_name],
            'img_name': img_name,
            #'cat_name': cat_name,
            'im_tensor': im_tensor,
            'label_coding': coding,
            'condition': condition,
            # 'prior_map': self.prior_maps[cat_name],
            'action_mask': torch.from_numpy(action_mask)
        }
        if self.coco_helper:
            centermaps = self.coco_helper.create_centermap_target(
                img_name, self.pa.patch_num[1], self.pa.patch_num[0])
            ret['centermaps'] = centermaps

        if self.pa.use_DCB_target:
            DCBs = torch.load(
                join(self.pa.DCB_dir, cat_name.replace(' ', '_'),
                     img_name[:-3] + 'pth.tar'))
            ret['DCB_target_map'] = DCBs[self.coco_thing_classes[cat_name]]

        return ret


class FFN_Human_Gaze(Dataset):
    """
    Human gaze data in state-action pairs for foveal feature net
    """
    def __init__(self,
                 root_dir,
                 fix_labels,
                 bbox_annos,
                 scene_annos,
                 pa,
                 transform,
                 catIds,
                 blur_action=False,
                 acc_foveal=True,
                 coco_annos=None):
        self.root_dir = root_dir
        self.img_dir = join(root_dir, 'images')
        self.pa = pa
        self.transform = transform
        # Remove scanpaths longer than max_traj_length
        self.fix_labels = list(
            filter(lambda x: len(x[3]) <= pa.max_traj_length + 1, fix_labels))
        self.catIds = catIds
        self.blur_action = blur_action
        self.acc_foveal = acc_foveal
        self.bboxes = bbox_annos
        # self.prior_maps = torch.load(join(root_dir, pa.prior_maps_dir))
        self.scene_labels = scene_annos['labels']
        self.scene_to_id = scene_annos['id_list']

        if self.pa.use_DCB_target:
            self.coco_thing_classes = np.load(join(root_dir,
                                                   'coco_thing_classes.npy'),
                                              allow_pickle=True).item()
        if coco_annos:
            self.coco_helper = COCODetHelper(coco_annos)
        else:
            self.coco_helper = None
        self.fv_tid = 0 if self.pa.TAP == 'FV' else len(self.catIds)

    def __len__(self):
        return len(self.fix_labels)

    def __getitem__(self, idx):
        img_name, cat_name, condition, fixs, action, is_last, sid, dura = self.fix_labels[
            idx]
        imgId = cat_name + '_' + img_name
        if imgId in self.bboxes.keys():
            bbox = self.bboxes[imgId]
        else:
            bbox = None

        if cat_name == 'none':
            im_path = "{}/{}".format(self.img_dir, img_name)
        else:
            c = cat_name.replace(' ', '_')
            im_path = "{}/{}/{}".format(self.img_dir, c, img_name)

        im = Image.open(im_path)
        im_tensor = self.transform(im)

        if bbox is not None:
            coding = utils.get_center_keypoint_map(
                bbox, self.pa.patch_num[::-1],
                box_size_dependent=False).view(1, -1)
        else:
            coding = torch.zeros(1, self.pa.patch_count)

        scanpath_length = len(fixs)
        # Pad fixations to max_traj_lenght
        fixs = fixs + [fixs[-1]] * (self.pa.max_traj_length - len(fixs) + 1)
        is_padding = torch.zeros(self.pa.max_traj_length + 1)
        is_padding[scanpath_length:] = 1

        fixs_tensor = torch.tensor(fixs)
        # Discretize fixations
        fixs_tensor = fixs_tensor // torch.tensor([self.pa.patch_size])
        # Normalize to 0-1
        fixs_tensor /= torch.tensor(self.pa.patch_num)

        next_fixs_tensor = fixs_tensor.clone()
        if action < self.pa.patch_count:
            x, y = utils.action_to_pos(action, self.pa.patch_size,
                                       self.pa.patch_num)
            next_fix = (torch.tensor([x, y]) // torch.tensor(
                [self.pa.patch_size])) / torch.tensor(self.pa.patch_num)
            next_fixs_tensor[scanpath_length:] = next_fix

        is_fv = condition == 'freeview'
        ret = {
            "task_id": self.fv_tid if is_fv else self.catIds[cat_name],
            "condition": condition,
            "true_state": im_tensor,
            "true_action": torch.tensor([action], dtype=torch.long),
            'label_coding': coding,
            'img_name': img_name,
            'task_name': cat_name,
            'normalized_fixations': fixs_tensor,
            'next_normalized_fixations': next_fixs_tensor,
            'is_TP': condition == 'present',
            'is_last': is_last,
            'is_padding': is_padding,
            'true_or_fake': 1.0,
            # 'prior_map': self.prior_maps[cat_name],
            'scanpath_length': scanpath_length,
            'duration': dura,
            'subj_id': sid - 1  # sid ranges from [1, 10]
        }

        if self.coco_helper:
            centermaps = self.coco_helper.create_centermap_target(
                img_name, self.pa.patch_num[1], self.pa.patch_num[0])
            ret['centermaps'] = centermaps

        # precomputed panoptic-FPN target map
        if self.pa.use_DCB_target:
            DCBs = torch.load(
                join(self.pa.DCB_dir, cat_name.replace(' ', '_'),
                     img_name[:-3] + 'pth.tar'))
            ret['DCB_target_map'] = DCBs[self.coco_thing_classes[cat_name]]
            ret['DCBs'] = DCBs

        # compute the map of last fixation
        if self.pa.use_action_map:
            action_map = np.zeros((self.pa.patch_num[1], self.pa.patch_num[0]),
                                  dtype=np.float32)
            action_map[ind_y[-1].item(), ind_x[-1].item()] = 1
            action_map = filters.gaussian_filter(action_map, sigma=1)
            ret['last_fixation_map'] = action_map

        if self.blur_action:
            action_map = np.zeros(self.pa.patch_count, dtype=np.float32)
            if action < self.pa.patch_count:
                action_map[action] = 1
                action_map = action_map.reshape(self.pa.patch_num[1], -1)
                # action_map = filters.gaussian_filter(action_map, sigma=1)
            else:
                action_map = action_map.reshape(self.pa.patch_num[1], -1)
            ret['action_map'] = action_map
        return ret


def normalize_im(im):
    if np.max(im) > 0:
        im = (im-np.min(im))/(np.max(im)-np.min(im))
    else:
        im = im
    return im
    
class SPTrans_Human_Gaze(Dataset):
    """
    Human gaze data for two-pathway dense transformer
    """
    def __init__(self,
                 root_dir,
                 fix_labels,
                 bbox_annos,
                 scene_annos,
                 pa,
                 transform,
                 #catIds,
                 blur_action=False,
                 #acc_foveal=True,
                 coco_annos=None):
        self.root_dir = root_dir
        self.img_dir = join(root_dir, 'images')
        self.pa = pa
        self.transform = transform
        self.to_tensor = T.ToTensor()

        self.fix_labels = fix_labels

        self.blur_action = blur_action
        self.bboxes = bbox_annos

        self.coco_helper = None

        self.grid_func = None

        self.resize = T.Resize(size=(pa.im_h // 2, pa.im_w // 2))
        self.resize2 = T.Resize(size=(pa.im_h // 4, pa.im_w // 4))

        self.fv_tid = 0 if self.pa.TAP == 'FV' else len(self.catIds)

    def __len__(self):
        return len(self.fix_labels)

    def __getitem__(self, idx):
        img_name, condition, fixs, action, next_mag, is_last, sid, dura, mags, height, width, other_wsis = self.fix_labels[idx]
        
        imgId = img_name

        IOR_weight_map = np.zeros((height,width), dtype=np.float32)
        IOR_weight_map += 1  # Set base weight to 1

        scanpath_length = len(fixs)
        if scanpath_length == 0:
            fixs = [(0, 0)]
        # Pad fixations to max_traj_length
        fixs = fixs + [fixs[-1]] * (self.pa.max_traj_length - len(fixs))
        is_padding = torch.zeros(self.pa.max_traj_length)
        is_padding[scanpath_length:] = 1

        act_len = len(mags)
        mags = mags + [mags[-1]] * (self.pa.max_traj_length - len(mags))

        fixs_tensor = torch.FloatTensor(fixs)[0:self.pa.max_traj_length]
        mags_tensor = torch.LongTensor(mags)[0:self.pa.max_traj_length]
        next_mag = torch.tensor(next_mag, dtype=torch.int32)

        fixs_tensor = fixs_tensor/torch.FloatTensor([height,width])
        next_fixs_tensor = fixs_tensor.clone()
        if not is_last:
            x, y = utils.action_to_pos(action, [1, 1], [height,width])
            next_fix = torch.FloatTensor([x, y]) / torch.FloatTensor([height,width])
            next_fixs_tensor[scanpath_length:] = next_fix
            
        mag1 = next_mag.detach().cpu().numpy()
        sigma0 = 14*(1/(mag1+1))
            
        target_fix_map = np.zeros((height*width),dtype=np.float32)
        if not is_last:
            target_fix_map[action] = 1
            target_fix_map = target_fix_map.reshape(height, -1)
            target_fix_map = filters.gaussian_filter(target_fix_map, sigma=sigma0) #self.pa.target_fix_map_sigma
            target_fix_map /= target_fix_map.max()
        else:
            target_fix_map = target_fix_map.reshape(height, -1)
            
            
        is_fv = condition == 'freeview'

        list_val = []
        for i in range(len(other_wsis)):
            traj = other_wsis[i]
            if len(traj['X']) > act_len:
                list_val.append(traj)

        if len(list_val) > 0:
            r1 = random.randint(0,len(list_val)-1)
            x1 = float(list_val[r1]['X'][act_len])/80
            y1 = float(list_val[r1]['Y'][act_len])/128
            m1 = list_val[r1]['M'][act_len]
            mag_list = [1,2,4,10,20,40]
            m1 = mag_list.index(m1)
        else:
            x1 = 0.5
            y1 = 0.5
            m1 = 0
        
        ret = {
            "task_id": self.fv_tid if is_fv else self.catIds[cat_name],
            "is_freeview": is_fv,
            "target_fix_map": target_fix_map,
            "true_action": torch.tensor([action], dtype=torch.long),
            'img_name': img_name,
            'is_TP': condition == 'present',
            'is_last': is_last,
            'normalized_fixations': fixs_tensor,
            'next_normalized_fixations': next_fixs_tensor,
            'mag': mags_tensor,
            'next_mag': next_mag,
            'is_padding': is_padding,
            'true_or_fake': 1.0,
            'IOR_weight_map': IOR_weight_map,
            'scanpath_length': scanpath_length,
            'duration': dura,
            'subj_id': sid - 1,  # sid ranges from [1, 10]
            'act_len':act_len,
            'random_point':[x1,y1,m1]
        }

        return ret
