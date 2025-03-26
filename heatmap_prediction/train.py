import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
from dataset import ListDataset
from dataset import Batch
from model import make_model
import os

import torchvision.models as models
import cv2
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def normalize_im(s_map):
	# normalize the salience map (as done in MIT code)
	norm_s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0)
	return norm_s_map
    

def cc(s_map, gt):
    # Avoid division by zero during normalization
    s_map_std = torch.std(s_map)
    gt_std = torch.std(gt)
    
    if s_map_std == 0 or gt_std == 0:
        return 0  # Or return a fallback value like 1 or 0
    
    # Normalize maps
    s_map_norm = (s_map - torch.mean(s_map)) / s_map_std
    gt_norm = (gt - torch.mean(gt)) / gt_std
    
    # Compute correlation
    a = s_map_norm
    b = gt_norm
    numerator = torch.sum(a * b)
    denominator = torch.sqrt(torch.sum(a * a) * torch.sum(b * b))
    
    # Handle potential zero denominator
    if denominator == 0:
        return 0  # Or return a fallback value
    
    r = numerator / denominator
    return 1 - r

def run_epoch(dataloader, model, optim_algo): #train_dataloader, model, nn.MSELoss(), model_opt
    total_tokens = 0
    total_loss = 0
    tokens = 0
    outs = []
    filesn = []
    shapes = []
    for i, (imgs, labels_y, files, shape1) in enumerate(dataloader):
    
        if optim_algo is not None:
            optim_algo.zero_grad()
        
        out, enc = model(imgs)
        
        outs.append(out)
        filesn.append(files)
        shapes.append(shape1)
        
            
        loss = cc(out.float(),labels_y.cuda().float()) #.float()
        
        total_loss += loss
        total_tokens += 1
        
        if optim_algo is not None:
            loss.backward()
            optim_algo.step()   

    #print(total_tokens)
    return total_loss/total_tokens,outs,filesn, shapes



def train():
    batch_size = 16
    mag = 2
    #split_id = 1
    grd_size = 10 #10, 20, 50, 100   #10,20,36,40
    #train_path = '../train_labels/dino_v2/'+str(mag)+'x/train/'
    #val_path = '../train_labels/dino_v2/'+str(mag)+'x/val/'
    for split_id in range(1,2):
        train_dataloader = torch.utils.data.DataLoader(ListDataset(split_id,'train','dino',mag,grd_size), batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(ListDataset(split_id,'val','dino',mag,grd_size), batch_size=1, shuffle=False)
        
        model = make_model(grd_size)
        model.cuda()
        
        model_opt = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9,weight_decay = 0.00005)
        
        val_loss = 100000
        for epoch in range(50):
            print('epoch:',epoch)
            
            model.train()
            train_loss,_,_,_ = run_epoch(train_dataloader, model, model_opt)
            print("train_loss", train_loss.item())
            
            with torch.no_grad():
                model.eval()
                test_loss, preds, file_names, shapes = run_epoch(val_dataloader, model, None)
                print("test_loss", test_loss.item())
                
                if test_loss.item() < val_loss:
                    print('Checkpoint saved with loss:',test_loss.item())
                    torch.save(model.state_dict(), './checkpoint/best_'+str(mag)+'_'+str(split_id)+'.pth')
                    val_loss = test_loss.item()
                    print(len(preds),len(file_names))
                    for i in range(len(preds)):
                        he = cv2.imread('../../wsi_new/'+file_names[i][0]+'.png')
                        output = np.uint8(normalize_im(preds[i].detach().cpu().numpy()[0][0])*255)
                        output = cv2.resize(output,(he.shape[1],he.shape[0]))
                        out1 = cv2.applyColorMap(output, cv2.COLORMAP_JET)
                        out1 = cv2.addWeighted(out1, 0.7, he, 0.3, 0)
                        
                        gt = cv2.resize(cv2.imread('../../gt_maps/'+file_names[i][0]+'/'+str(mag)+'x.png',0),(output.shape[1],output.shape[0]))
                        gt1 = cv2.applyColorMap(gt, cv2.COLORMAP_JET)
                        gt1 = cv2.addWeighted(gt1, 0.7, he, 0.3, 0)
                
                        output_c = np.concatenate((out1,gt1),1)
                        output_c2 = np.concatenate((output,gt),1)
                        output_c = np.concatenate((he,output_c),1)
            
        print('Done!!!!!!')
if __name__=='__main__':
    train()





