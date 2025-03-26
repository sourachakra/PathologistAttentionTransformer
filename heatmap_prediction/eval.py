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
	norm_s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0)
	return norm_s_map
    
def cc(s_map,gt):
	s_map_norm = (s_map - torch.mean(s_map))/torch.std(s_map)
	gt_norm = (gt - torch.mean(gt))/torch.std(gt)
	a = s_map_norm
	b= gt_norm
	r = torch.sum(a*b) / torch.sqrt(torch.sum(a*a) * torch.sum(b*b));
	return 1-r

def run_epoch(dataloader, model, optim_algo): #train_dataloader, model, nn.MSELoss(), model_opt
    total_tokens = 0
    total_loss = 0
    tokens = 0
    outs = []
    filesn = []
    shapes = []
    encoded_feats = []
    for i, (imgs, labels_y, files, shape1) in enumerate(dataloader):
    
        if optim_algo is not None:
            optim_algo.zero_grad()
        
        out, enc = model(imgs)
        
        outs.append(out)
        filesn.append(files)
        shapes.append(shape1)
        encoded_feats.append(enc.detach().cpu())
        

        loss = cc(out.float(),labels_y.cuda().float()) #.float()
        total_loss += loss
        total_tokens += 1
        
        if optim_algo is not None:
            loss.backward()
            optim_algo.step()   

    #print(total_tokens)
    return total_loss/total_tokens,outs,filesn, shapes, encoded_feats



def train():
    batch_size = 16
    mag = 2 # Other values: 4,10,20
    #split_id = 1
    grd_size = 10 #10 (2x), 20 (4x), 50 (10x), 60 (20x)
    #train_path = '../train_labels/dino_v2/'+str(mag)+'x/train/'
    #val_path = '../train_labels/dino_v2/'+str(mag)+'x/val/'
    for split_id in range(1,2):
        train_dataloader = torch.utils.data.DataLoader(ListDataset(split_id,'train','dino',mag,grd_size), batch_size=1, shuffle=False)
        val_dataloader = torch.utils.data.DataLoader(ListDataset(split_id,'val','dino',mag,grd_size), batch_size=1, shuffle=False)
        
        model = make_model(grd_size)
        state_dict = torch.load('/home/schakraborty/classify_WSI_prostate/training_codes/heatmap_prediction/checkpoint/best_'+str(mag)+'_'+str(split_id)+'.pth')
        model.load_state_dict(state_dict)
        model.cuda()

        with torch.no_grad():
            model.eval()
            test_loss, preds, file_names, shapes, encoded_feats = run_epoch(val_dataloader, model, None)
            
            for i in range(len(preds)):
                x = encoded_feats[i]
                #print('encoded_feats:',file_names[i][0],x.shape)
                
                # saving the encoded feature representations
                np.save('./feature_encodings/split'+str(split_id)+'/'+str(mag)+'x/'+'val/'+file_names[i][0]+'.npy', x)
                #cv2.imwrite('./our2_preds/split'+str(split_id)+'/pred_'+str(mag)+'x/'+file_names[i][0]+'.png',output_c)
               
                output_c2 = preds[i][0][0].detach().cpu().numpy()
                output_c2 = np.uint8(normalize_im(output_c2)*255)
                cv2.imwrite('./predicted_heatmaps/split'+str(split_id)+'/'+str(mag)+'x/'+file_names[i][0]+'_gray.png',output_c2)
            
        print('Done!!!!!!')
if __name__=='__main__':
    train()





