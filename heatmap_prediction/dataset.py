import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
import os
import cv2
import numpy as np
from torchvision import datasets, models, transforms
import glob
import torch.nn.functional as F

def normalize_im(s_map):
	norm_s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0)
	return norm_s_map
    
    
class ListDataset(Dataset):
    def __init__(self, split_id,mode,method,mag,grd_size):
        self.pack = []
        self.grid_sze = grd_size
        train_npys = sorted(glob.glob('/home/schakraborty/classify_WSI_prostate/training_codes/train_labels/'+method+'/'+str(mag)+'x/*.npy')) # contains 
        #val_npys = sorted(glob.glob('../train_labels/'+method+'/'+str(mag)+'x/val/*.npy'))
        all_npys0 = sorted(train_npys) #+val_npys)
        print(len(all_npys0))
        
        len1 = len(all_npys0)/2
        print(len1)
        
        if split_id == 1:
            if mode == 'val':
                all_npys = all_npys0[0:25*2]
            else:
                all_npys1 = all_npys0[0:25*2]
                all_npys = list(filter(lambda x: x not in all_npys1, all_npys0))
                
        elif split_id == 2:
            if mode == 'val':
                all_npys = all_npys0[25*2:50*2]
            else:
                all_npys1 = all_npys0[25*2:50*2]
                all_npys = list(filter(lambda x: x not in all_npys1, all_npys0))
            
        elif split_id == 3:
            if mode == 'val':
                all_npys = all_npys0[50*2:75*2]
            else:
                all_npys1 = all_npys0[50*2:75*2]
                all_npys = list(filter(lambda x: x not in all_npys1, all_npys0))
                
        elif split_id == 4:
            if mode == 'val':
                all_npys = all_npys0[75*2:100*2]
            else:
                all_npys1 = all_npys0[75*2:100*2]
                all_npys = list(filter(lambda x: x not in all_npys1, all_npys0))
            
        elif split_id == 5:
            if mode == 'val':
                all_npys = all_npys0[100*2:123*2]
            else:
                all_npys1 = all_npys0[100*2:123*2]
                all_npys = list(filter(lambda x: x not in all_npys1, all_npys0))
            
            
            
        for i in range(0,int(len(all_npys)),2):
            
            dat = np.load(all_npys[i],allow_pickle=True)
           
            lab = np.load(all_npys[i+1],allow_pickle=True)/255

            self.pack.append([dat,lab,all_npys[i].split('/')[-1][:-8],lab.shape])
            
        

    def __len__(self):
        return len(self.pack)
    
    def __getitem__(self, index):
        line = self.pack[index]
        grid_width = self.grid_sze 
        
        im_embed, lab, filen, file_shape = line[0], line[1], line[2], line[3]
        im_embed = torch.from_numpy(im_embed).cuda().float()
        im_embed = torch.permute(im_embed.unsqueeze(0), (0, 3, 1, 2)).cuda()
        im_embed = F.interpolate(im_embed, [grid_width, grid_width], mode='bilinear', align_corners=True)
        im_embed = im_embed.squeeze(0)
        
        lab = torch.from_numpy(lab).cuda() #/255
        lab = F.interpolate(lab.unsqueeze(0).unsqueeze(0), [grid_width,grid_width], mode='bilinear', align_corners=True)
        lab = lab.squeeze(0)

        return im_embed, lab, filen, file_shape


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, imgs, trg_y, pad=0):
        self.imgs = Variable(imgs.cuda(), requires_grad=False)
        self.src_mask = Variable(torch.from_numpy(np.ones([imgs.size(0), 1, label_len], dtype=np.bool)).cuda())
        if trg is not None:
            self.trg = Variable(trg.cuda(), requires_grad=False)
            self.trg_y = Variable(trg_y.cuda(), requires_grad=False)
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return Variable(tgt_mask.cuda(), requires_grad=False)


if __name__=='__main__':
    listdataset = ListDataset('your-lines')
    dataloader = torch.utils.data.DataLoader(listdataset, batch_size=2, shuffle=False, num_workers=0)
    for epoch in range(1):
        for batch_i, (imgs, labels_y, labels) in enumerate(dataloader):
            continue


















