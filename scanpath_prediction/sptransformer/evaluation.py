import sys
sys.path.append('../common')

from common import utils, metrics
from torch.distributions import Categorical
import torch, json
from tqdm import tqdm
import numpy as np
from os.path import join
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
import gc
from memory_profiler import profile
import json
from scipy.ndimage import gaussian_filter


os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda')

    
def create_gaussian_map(center_x, center_y, sigma=10, width=128, height=80):
    # Create coordinate grids
    x = np.arange(0, width)
    y = np.arange(0, height)
    x_grid, y_grid = np.meshgrid(x, y)

    # Calculate the Gaussian map
    gaussian_map = np.exp(-((x_grid - center_x)**2 + (y_grid - center_y)**2) / (2 * sigma**2))
    return gaussian_map
    

def normalize_im(im):
    if np.max(im) > 0:
        im = (im-np.min(im))/(np.max(im)-np.min(im))
    else:
        im = im
    return im
        
            
def get_IOR_mask(norm_x, norm_y, h, w, r):
    bs = len(norm_x)
    x, y = norm_x * h, norm_y * w
    #print('x,y:',X.size(),Y.size())
    X, Y = np.ogrid[:h, :w]
    X = X.reshape(1, h, 1)
    Y = Y.reshape(1, 1, w)
    x = x.reshape(bs, 1, 1)
    y = y.reshape(bs, 1, 1)
    dist = np.sqrt((X - x)**2 + (Y - y)**2)
    mask = dist <= r
    return torch.from_numpy(mask.reshape(bs, -1))
    


def actions2scanpaths(norm_fixs, patch_num, im_h, im_w, mags1):
    # convert actions to scanpaths
    scanpaths = []
    mag_list = [1,2,4,10,20,40]
    for traj in norm_fixs:
        img_name, condition, fixs = traj
        fixs = fixs.numpy()
        #print('mags2:',mags[0])
        scanpaths.append({
            'X': fixs[:, 0] * pa.im_h,
            'Y': fixs[:, 1] * pa.im_w,
            'M': [mag_list[i] for i in mags1[0]],
            'name': img_name,
            'condition': condition
        })
    #print('scanpaths:',scanpaths)
    return scanpaths
    
def log_memory_usage(step_description):
    print(f"--- {step_description} ---")
    print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")
    print(f"Reserved memory: {torch.cuda.memory_reserved() / (1024 ** 2)} MB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2)} MB")
    print(f"Max memory reserved: {torch.cuda.max_memory_reserved() / (1024 ** 2)} MB")
    print()


def plot_scanpath(img, xs, ys, gt_fix, mags, next_mag, save_path):
    fig, ax = plt.subplots()
    ax.imshow(img)
    
    gxs,gys = gt_fix[1],gt_fix[0]
    
    colors = ['RoyalBlue', 'cyan', 'blue', 'green', 'yellow', 'red']
    color_labels = ['1x', '2x', '4x', '10x', '20x', '40x']
    
    next_mag = next_mag.cpu().numpy()[0]

    # Plot arrows between points
    for i in range(len(xs)):
        if i > 0:
            plt.arrow(xs[i - 1], ys[i - 1], xs[i] - xs[i - 1],
                      ys[i] - ys[i - 1], width=3, color='yellow', alpha=0.5)
    
    # Plot circles at points
    for i in range(len(mags)):
        if i < len(mags) - 1:
            cir_rad = int(25)  # radius for intermediate points
            alpha1 = 0.5
        else:
            cir_rad = int(50)  # radius for the last point
            alpha1 = 0.8
        circle = plt.Circle((xs[i], ys[i]),
                            radius=cir_rad,
                            edgecolor='red',
                            facecolor=colors[mags[i]],
                            alpha=alpha1)
        ax.add_patch(circle)
        plt.annotate("{}".format(i + 1), xy=(xs[i], ys[i] + 3), fontsize=6, ha="center", va="center")
    
    circle = plt.Circle((gxs, gys),
                            radius=cir_rad,
                            edgecolor='red',
                            facecolor=colors[next_mag],
                            alpha=alpha1)
    plt.annotate("GT", xy=(gxs, gys + 3), fontsize=6, ha="center", va="center")
    ax.add_patch(circle)
        
    # Create legend patches
    legend_patches = [mpatches.Patch(color=colors[i], label=color_labels[i]) for i in range(len(colors))]
    
    # Add legend to the plot
    ax.legend(handles=legend_patches, loc='lower right', bbox_to_anchor=(1,0), fontsize='small', markerscale=0.5, framealpha=0.8)
    
    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    
    plt.show()
    
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    
    
def plot_scanpath2(img, xs, ys, mags, save_path):
    fig, ax = plt.subplots()
    ax.imshow(img)
    colors = ['RoyalBlue','cyan','blue','green','yellow','red']
    for i in range(len(xs)):
        if i > 0:
            plt.arrow(xs[i - 1], ys[i - 1], xs[i] - xs[i - 1],
                      ys[i] - ys[i - 1], width=3, color='yellow', alpha=0.5)
    for i in range(len(mags)):
        if i < len(mags)-1:
            cir_rad = int(25) # + rad_per_T * (ts[i] - min_T))
        else:
            cir_rad = int(50)
        circle = plt.Circle((xs[i], ys[i]),
                            radius=cir_rad,
                            edgecolor='red',
                            facecolor=colors[mags[i]],
                            alpha=0.5)
        ax.add_patch(circle)
        plt.annotate("{}".format(
            i+1), xy=(xs[i], ys[i]+3), fontsize=6, ha="center", va="center")

    ax.axis('off')
    plt.savefig(save_path,bbox_inches='tight',pad_inches=0)

def extract_embeds_point(fix_x,fix_y,fix_m,name1,dict_2x,dict_4x,dict_10x,dict_20x):

    name_embs = dict()
    embs = np.zeros((len(fix_x),384))
    name1 = name1+'-01Z-00-DX1'
    for j in range(len(fix_x)):
        mag = fix_m[j]
        #print('mg:',mag)
        if mag == 1:
            arr = dict_2x[name1]
        if mag == 2:
            arr = dict_2x[name1]
        if mag == 4:
            arr = dict_4x[name1]
        if mag == 10:
            arr = dict_10x[name1]
        if mag == 20:
            arr = dict_20x[name1]
        if mag == 40:
            arr = dict_20x[name1]
            
        #print('arr:',arr.shape)
        h,w = arr.shape[1],arr.shape[2]
        
        h_m = int((fix_x[j]/80)*h)
        w_m = int((fix_y[j]/128)*w)
        
        embs[j,0:384] = arr[0,h_m,w_m,:]
        
    return embs
    
def cosine_similarity_loss(embedding1, embedding2):
    # Normalize the embeddings
    embedding1 = F.normalize(embedding1, p=2, dim=-1)
    embedding2 = F.normalize(embedding2, p=2, dim=-1)
    
    # Compute the cosine similarity (between -1 and 1)
    similarity = torch.sum(embedding1 * embedding2, dim=-1)
    
    # Define a loss as the negative of cosine similarity (to maximize similarity)
    loss = similarity
    return loss.mean()

def visualize_angle_on_image(image, prev_point, pred_point, gt_point, angle,name,count):
    # Scale normalized points
    scale = 4
    prev_point = (int(prev_point[1] * 128*scale),int(prev_point[0] * 80*scale))
    pred_point = (int(pred_point[1] * 128*scale),int(pred_point[0] * 80*scale))
    gt_point = (int(gt_point[1] * 128*scale),int(gt_point[0] * 80*scale))

    # Plot points on image
    img_copy = image.copy()
    cv2.circle(img_copy, prev_point, 5, (255, 0, 0), -1)  # Previous point in blue
    cv2.circle(img_copy, pred_point, 5, (0, 255, 0), -1)  # Predicted point in green
    cv2.circle(img_copy, gt_point, 5, (0, 0, 255), -1)    # Ground truth point in red

    # Draw lines
    cv2.line(img_copy, prev_point, pred_point, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.line(img_copy, prev_point, gt_point, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    # Annotate angle
    text_position = (prev_point[0] + 10, prev_point[1] - 10)
    cv2.putText(img_copy, f'Angle: {angle:.2f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 1, cv2.LINE_AA)

    # Show the image
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Angle Visualization with angle = "+str(angle))
    plt.savefig('./angles/'+name+'_'+str(count)+'.png',bbox_inches='tight', pad_inches=0)


def compute_angle(last_point, predicted_point, ground_truth_point):
    # Convert points to vectors
    vector_pred = np.array(predicted_point) - np.array(last_point)
    vector_gt = np.array(ground_truth_point) - np.array(last_point)
    
    # Compute magnitudes
    magnitude_pred = np.linalg.norm(vector_pred)
    magnitude_gt = np.linalg.norm(vector_gt)
    
    # Check if either vector is a zero vector to avoid division by zero
    if magnitude_pred == 0 or magnitude_gt == 0:
        return -1

    # Compute dot product and angle
    dot_product = np.dot(vector_pred, vector_gt)
    angle_radians = np.arccos(dot_product / (magnitude_pred * magnitude_gt))
    angle_degrees = np.degrees(angle_radians)
    
    if np.isnan(angle_degrees):
        return -1
    return angle_degrees
    
    
def compute_sim(pred_1x, gt_1x, mag):
    # Move tensors to GPU
    pred_1x = torch.tensor(pred_1x).cuda()
    gt_1x = torch.tensor(gt_1x).cuda()
    
    sum1 = 0
    for i in range(len(pred_1x)):
        for j in range(len(gt_1x)):
            sum1 += cosine_sim(pred_1x[i], gt_1x[j])
    
    if len(pred_1x) == 0 or len(gt_1x) == 0:
        sum1 = 0
    else:
        sum1 /= len(pred_1x) * len(gt_1x)
    
    #print(f'Avg. cosine similarity of patch embeddings at {mag}:', sum1)
    return sum1,len(gt_1x)
    
def cosine_sim(vector_a, vector_b):
    dot_product = torch.dot(vector_a, vector_b)
    norm_a = torch.norm(vector_a)
    norm_b = torch.norm(vector_b)
    
    if norm_a == 0 or norm_b == 0:
        cosine_similarity = 0
    else:
        cosine_similarity = dot_product / (norm_a * norm_b)
        
    return cosine_similarity
   
    
def generate_heatmap(points, img_size, sigma):
    # Initialize an empty array for the heatmap
    heatmap = np.zeros(img_size)

    # Place a value at each fixation point location
    #print(points)
    for point in points:
        x,y = point[0],point[1]
        if 0 <= x < img_size[0] and 0 <= y < img_size[1]:  # Ensure point is within bounds
            heatmap[int(x), int(y)] += 1

    # Apply Gaussian filter to create smooth heatmap
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    heatmap = (heatmap-np.min(heatmap))/(np.max(heatmap)-np.min(heatmap))
    return heatmap
    
def generate_center_bias_map(img_shape, sigma):
    height, width = img_shape
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)
    x_center = width / 2
    y_center = height / 2
    center_bias_map = np.exp(-((x_grid - x_center)**2 + (y_grid - y_center)**2) / (2 * sigma**2))
    return center_bias_map
    
def compute_scan_metrics(path_pred,path_gt):
    with open(path_pred, 'r') as json_file:
        pred_scan = json.load(json_file)
    with open(path_gt, 'r') as json_file:
        gt_scan = json.load(json_file)

    
    g = glob.glob('./agg_maps/*.png')
    for k in range(len(g)):
        map1 = cv2.imread(g[k],0)
        if k == 0:
            map2 = cv2.resize(map1,(128,80)).astype(np.float32)
        else:
            map2 += cv2.resize(map1,(128,80)).astype(np.float32)
    # cb2 = map2/len(g)
    # cb2 = (cb2-np.min(cb2))/(np.max(cb2)-np.min(cb2))
    # cv2.imwrite('./cb_overall.png',np.uint8(cb2*255))
    
    all_nss,all_ig,all_auc,all_scanmatch = [],[],[],[]
    for key in pred_scan.keys(): # per pred image

        name1 = key
    
        list_xy = [[x, y] for x, y in zip(pred_scan[key]['X'], pred_scan[key]['Y'])]
        list_xym = [(x, y) for x, y in zip(pred_scan[key]['X'], pred_scan[key]['Y'])]
        pred_map = generate_heatmap(list_xy, (80,128), 6)

        avg_nss,avg_auc,count = 0,0,0
        avg_ig = 0
        list_xy3 = []
        #avg_scanmatch = 0
        for key1 in gt_scan.keys():
            #print(gt_scan[key1]['name'][:4],name1)
            if gt_scan[key1]['name'][:-4] == name1:
                #list_xy2 = np.array([[x, y] for x, y in zip(gt_scan[key1]['X'], gt_scan[key1]['Y'])])
                list_xy3.extend([[x, y] for x, y in zip(gt_scan[key1]['X'], gt_scan[key1]['Y'])])
                list_newm = [(x, y) for x, y in zip(gt_scan[key1]['X'], gt_scan[key1]['Y'])]
                
                #m = metrics.multimatch(pred_scan[key], gt_scan[key1], [128,80])
                #m = metrics.scanmatch(list_xym, list_newm)
                #avg_scanmatch += m
                count += 1
                
        list_xy3 = np.array(list_xy3)
        avg_nss = metrics.NSS(pred_map, list_xy3)
        avg_auc = metrics.AUC(pred_map, list_xy3)
        avg_ig = metrics.info_gain(pred_map, list_xy3, cb2)
        #avg_ig2 = metrics.info_gain(cb3, list_xy3, cb2)
        #avg_ig = avg_ig1/avg_ig2
                
        # avg_nss /= count
        # avg_ig /= count
        all_nss.append([name1,avg_nss])
        all_ig.append([name1,avg_ig])
        #all_scanmatch.append([name1,avg_scanmatch/count])
        all_auc.append([name1,avg_auc])
        
    #print('all_list:',all_nss)
    return all_nss,all_ig,all_auc #,all_scanmatch
        
def visualize_ims(pred,save_path,save_path2,name1,fixes,gt_fix,mags,next_mag,count,dict_2x,dict_4x,dict_10x,dict_20x):

    mag_cacc_1,mag_cacc_2,mag_cacc_4,mag_cacc_10,mag_cacc_20,mag_cacc_40 = 0,0,0,0,0,0
    l2_1,l2_2,l2_4,l2_10,l2_20,l2_40 = 0,0,0,0,0,0
    
    prev_fixes = np.array(fixes[0]).astype(np.float32)

    prev_fixes[:,0] = prev_fixes[:,0].astype(np.float32)*(1/80) #80 orig.shape[0]
    prev_fixes[:,1] = prev_fixes[:,1].astype(np.float32)*(1/128) #128 orig.shape[1]
    
    gt_fix = np.array(gt_fix).astype(np.float32)
    gt_fix[0] = gt_fix[0].astype(np.float32)*(1/80) #80 orig.shape[0]
    gt_fix[1] = gt_fix[1].astype(np.float32)*(1/128) #128 orig.shape[1]
    
    gt_mg = next_mag[0].cpu().numpy()
    
    dist_l2 = np.sqrt((prev_fixes[-1,0]-gt_fix[0])**2+(prev_fixes[-1,1]-gt_fix[1])**2)
    if gt_mg == 0:
        l2_1 = dist_l2
    elif gt_mg == 1:
        l2_2 = dist_l2
    elif gt_mg == 2:
        l2_4 = dist_l2
    elif gt_mg == 3:
        l2_10 = dist_l2
    elif gt_mg == 4:
        l2_20 = dist_l2
    elif gt_mg == 5:
        l2_40 = dist_l2    
    

    prev_point = prev_fixes[-2,:]
    pred_point = prev_fixes[-1,:]
    gt_point = gt_fix[:]
    
    angle = compute_angle(prev_point, pred_point, gt_point)
    #image = cv2.imread('/home/soura/scanpath_prediction_all-main2/datasets/WSIs/images/'+name1+'-01Z-00-DX1.png')
    #image = cv2.resize(image,(128*4,80*4))
    #visualize_angle_on_image(image, prev_point, pred_point, gt_point, angle, name1,count)
    
    
    #print('gt,pred:',gt_mg,mags[-1],mags)
    mag_list = [1,2,4,10,20,40]
    #print(mags)
    m1 = [mag_list[int(np.round(mags[-1]))]]
    pred_emb = extract_embeds_point([int(pred_point[0]*80)],[int(pred_point[1]*128)],m1,name1,dict_2x,dict_4x,dict_10x,dict_20x)
    m2 = [mag_list[int(np.round(gt_mg))]]
    gt_emb = extract_embeds_point([int(gt_point[0]*80)],[int(gt_point[1]*128)],m2,name1,dict_2x,dict_4x,dict_10x,dict_20x)
    cos_score = cosine_similarity_loss(torch.tensor(pred_emb),torch.tensor(gt_emb)).detach().cpu().numpy()
    
    #info_gain += metrics.compute_info_gain(probs, gt_next_fixs, prior_maps)
    #print(pred.shape,torch.tensor([int(gt_point[0]*80),int(gt_point[1]*128)]).shape)
    pred_tens = torch.tensor(pred.astype(np.float32)/255).unsqueeze(0)
    pt_tens = torch.tensor([int(gt_point[0]*80),int(gt_point[1]*128)]).unsqueeze(0)
    
    #nss_score = metrics.compute_NSS(pred_tens,pt_tens)
    #auc_score = metrics.compute_cAUC(pred_tens,pt_tens)
    #print(nss_score,auc_score)
        
    mag_acc = 1 if gt_mg == mags[-1] else 0
    
    mag_acc_1 = 1 if gt_mg == mags[-1] and gt_mg == 0 else 0
    mag_acc_2 = 1 if gt_mg == mags[-1] and gt_mg == 1 else 0
    mag_acc_4 = 1 if gt_mg == mags[-1] and gt_mg == 2 else 0
    mag_acc_10 = 1 if gt_mg == mags[-1] and gt_mg == 3 else 0
    mag_acc_20 = 1 if gt_mg == mags[-1] and gt_mg == 4 else 0
    mag_acc_40 = 1 if gt_mg == mags[-1] and gt_mg == 5 else 0
    
    mag_change_acc = -1
    if mags[-2] != next_mag[0].cpu():
        mag_change_acc = 1 if gt_mg == mags[-1] else 0
        mag_cacc_1 = 1 if gt_mg == mags[-1] and gt_mg == 0 else 0
        mag_cacc_2 = 1 if gt_mg == mags[-1] and gt_mg == 1 else 0
        mag_cacc_4 = 1 if gt_mg == mags[-1] and gt_mg == 2 else 0
        mag_cacc_10 = 1 if gt_mg == mags[-1] and gt_mg == 3 else 0
        mag_cacc_20 = 1 if gt_mg == mags[-1] and gt_mg == 4 else 0
        mag_cacc_40 = 1 if gt_mg == mags[-1] and gt_mg == 5 else 0
    
    #scan = plot_scanpath(orig, prev_fixes[:,1], prev_fixes[:,0], gt_fix, mags, next_mag, save_path2)
    
    del gt_fix, prev_fixes, fixes
    
    return dist_l2,l2_1,l2_2,l2_4,l2_10,l2_20,l2_40,mag_acc,mag_change_acc,mag_acc_1,mag_acc_2,mag_acc_4,mag_acc_10,mag_acc_20,mag_acc_40,mag_cacc_1,mag_cacc_2,mag_cacc_4,mag_cacc_10,mag_cacc_20,mag_cacc_40,angle,cos_score#,nss_score,auc_score
    
def scanpath_decode(model, dict_2x,dict_4x,dict_10x,dict_20x, sn, img_names_batch, i_epoch, count2, batch, task_ids, pa, sample_action=False, center_initial=True):

    bs = 1
    name1 = img_names_batch[0]
    mags_seq = batch['mag'].to(device)
    next_mags = batch['next_mag'].to(device)
    act_len = batch['act_len']
    normalized_fixes = batch['normalized_fixations'] #.to(device)
    next_normalized_fixes = batch['next_normalized_fixations'] #.to(device)
    sid = batch['subj_id']
    random_point = batch['random_point']
    
    patatt_2x = np.load('./input_features/split'+str(sn)+'/2x/'+name1[:-4]+'-01Z-00-DX1.npy',allow_pickle=True)[0]
    patatt_10x = np.load('./input_features/split'+str(sn)+'/10x/'+name1[:-4]+'-01Z-00-DX1.npy',allow_pickle=True)[0]
    
    low_res_embed = patatt_2x #all_res_embeds[1] #res_embeds_2x #
    high_res_embed = patatt_10x 
    
    low_res_embed = torch.from_numpy(low_res_embed).unsqueeze(0)
    low_res_embed = torch.permute(low_res_embed,(0,3,1,2))
    high_res_embed = torch.from_numpy(high_res_embed).unsqueeze(0)
    high_res_embed = torch.permute(high_res_embed,(0,3,1,2))
    low_res_embed = F.interpolate(low_res_embed, [int(pa.im_h/8),int(pa.im_w/8)], mode='bilinear', align_corners=True)
    high_res_embed = F.interpolate(high_res_embed, [pa.im_h,pa.im_w], mode='bilinear', align_corners=True)
    low_res_embed = torch.permute(low_res_embed,(0,2,3,1)).squeeze(0)
    high_res_embed = torch.permute(high_res_embed,(0,2,3,1)).squeeze(0)
    low_res_embed = low_res_embed.numpy()
    high_res_embed = high_res_embed.numpy()
            
    #log_memory_usage("Evaluation: After input")
    trans_mat = [[0.286,0.277,0.274,0.241,0.549,0.804],[1,0.825,0.710,0.418,0.168,0]]
    with torch.no_grad():
        dorsal_embs, dorsal_pos, dorsal_mask, high_res_featmaps = model.encode(name1,low_res_embed,high_res_embed)

        action_mask = torch.zeros(bs, pa.im_h * pa.im_w)
        #log_memory_usage("Evaluation: After encoder")
        
    stop_flags,mags,mag_pred = [],[],[]
    dist,magacc,magcacc = 0,0,0
    
    mag_acc_1,mag_acc_2,mag_acc_4,mag_acc_10,mag_acc_20,mag_acc_40 = 0,0,0,0,0,0
    mag_cacc_1,mag_cacc_2,mag_cacc_4,mag_cacc_10,mag_cacc_20,mag_cacc_40 = 0,0,0,0,0,0
    l2_1,l2_2,l2_4,l2_10,l2_20,l2_40 = 0,0,0,0,0,0
    angle,cos_score = 0,0
    nss_score,auc_score = 0,0

    for i in range(1):
        with torch.no_grad():
            if i == 0 and not center_initial:
                ys = ys_high = torch.zeros(bs, 1).to(torch.long)
                padding = torch.ones(bs, 1).bool().to(img.device)
            else:
                ys, ys_high = utils.transform_fixations(normalized_fixes, None, pa, [1,int(pa.im_h/8),int(pa.im_w/8)], [1,pa.im_h,pa.im_w], False,return_highres=True)
                padding = None

            #print('vals:',ys.shape,normalized_fixes.shape)
            out = model.decode_and_predict(#map_2x,map_4x,map_10x,map_20x,
                low_res_embed,high_res_embed,dorsal_embs, dorsal_pos, dorsal_mask, high_res_featmaps,
                ys.to(device), padding, ys_high.to(device), mags_seq, act_len, 'validate', task_ids) #.detach()
                

            prob, mag= out['pred_fixation_map'], out['pred_magnification'] #, out['pred_next_point']

            # prob2 = torch.reshape(prob,(1,80,128))
            # xy = normalized_fixes[0][0:act_len][-1].cpu().numpy()
            # center_y, center_x = int(xy[0]*80), int(xy[1]*128)  # center point in (width, height)
            # sigma = 1 #3
            # width, height = 128, 80
            # gaussian_map = create_gaussian_map(center_x, center_y, sigma, width, height)
            # mod_prob = (prob2[0]*(torch.tensor(gaussian_map).cuda())).unsqueeze(0)
            # prob = mod_prob.view(1, -1) 

            del out

            mags.append(mag)
            
            
            # temperature = 0.15
            # mag2 = torch.softmax(mag[0] / temperature, dim=0)
            # distribution = torch.distributions.Categorical(mag2)
            # mag1 = distribution.sample().cpu().numpy()#-1+mags_seq[0][-1]
            #mag_pred.append(mag1)
            
            #======================
            mag1 = mags_seq[0][-1]     
            temperature = 0.5 #4 for split1, 1 for split2, 8 for split5,,,,,,0.5 for scanfix (point pred)
            probabilities = F.softmax(mag[0]/temperature, dim=-1)
            if mag1 == 0:
                m = torch.distributions.Categorical(probabilities[0:2])
                mag = mag1 + m.sample().item()
            elif mag1 > 0 and mag1 < 5:
                m = torch.distributions.Categorical(probabilities[mag1-1:mag1+2])
                mag = mag1 + m.sample().item()-1
            elif mag1 == 5:
                m = torch.distributions.Categorical(probabilities[4:6])
                mag = mag1 + m.sample().item()-1
            mag1 = mag.cpu().numpy()
            #====================
            
            #mag1 = torch.argmax(mag[0])
            #mag1 = mag1.cpu().numpy()
            
            #====================
            #mag1 = random.randint(0, 4)
            #mag1 = random_point[2].cpu().numpy()
            # mag1 = min(max(mag1,0),5)
            # mag_pred.append(mag1)
            
            #del mag
            del high_res_embed,low_res_embed
            del dorsal_embs, dorsal_pos, dorsal_mask

        if sample_action:
            m = Categorical(prob)
            next_word = m.sample()
        else:
            _, next_word = torch.max(prob, dim=1)
        
        
        next_word = next_word.cpu()
        
        norm_fy = (next_word % pa.im_w) / float(pa.im_w)
        norm_fx = (next_word // pa.im_w) / float(pa.im_h)
        
        act_len = act_len.cpu().numpy()[0]

        normalized_fixes2 = torch.cat([normalized_fixes[0][0:act_len].unsqueeze(0), torch.stack([norm_fx, norm_fy], dim=1).unsqueeze(0)], dim=1)
        
        
        if norm_fx != 0 and norm_fy != 0 and act_len < pa.max_traj_length: #i == pa.max_traj_length-1 and 
            
            fixs_pred = (normalized_fixes2 * torch.Tensor([pa.im_h,pa.im_w])).to(torch.long)
            gt_fix = (next_normalized_fixes[0][act_len]* torch.Tensor([pa.im_h,pa.im_w])).to(torch.long)
            pred_map = torch.reshape(prob,(pa.im_h,pa.im_w)).detach().cpu().numpy()
            prob.detach()
            #mag.detach()
            pred_map = normalize_im(pred_map)*255

            #orig = cv2.imread('/home/soura/scanpath_prediction_all-main2/datasets/WSIs/images/'+name1[:-4]+'-01Z-00-DX1.png')
            #orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            if not os.path.exists('./compare_eval/'+str(i_epoch)):
                os.mkdir('./compare_eval/'+str(i_epoch))
            save_path = './compare_eval/'+str(i_epoch)+'/'+name1[:-4]+'_'+str(i)+'_'+str(count2)+'_1.png'
            save_path2 = './compare_eval/'+str(i_epoch)+'/'+name1[:-4]+'_'+str(i)+'_'+str(count2)+'_2.png'

            tot_mag = mags_seq[0].cpu().tolist()
            tot_mag.append(mag1)
            dist,l2_1,l2_2,l2_4,l2_10,l2_20,l2_40,magacc,magcacc,mag_acc_1,mag_acc_2,mag_acc_4,mag_acc_10,mag_acc_20,mag_acc_40,mag_cacc_1,mag_cacc_2,mag_cacc_4,mag_cacc_10,mag_cacc_20,mag_cacc_40,angle,cos_score = visualize_ims(pred_map, save_path, save_path2, name1[:-4], fixs_pred, gt_fix, tot_mag, next_mags,count2,dict_2x,dict_4x,dict_10x,dict_20x)  #,inp_seq, inp_seq_high
            
            del pred_map, fixs_pred, gt_fix, tot_mag #, out
                
    
    mags = torch.stack(mags, dim=1)
    trajs = []
    for i in range(normalized_fixes.size(0)):
        trajs.append(normalized_fixes[i, :])

    nonstop_trajs = [normalized_fixes[i] for i in range(normalized_fixes.size(0))]
    
    del normalized_fixes, normalized_fixes2, patatt_2x, patatt_10x #all_res_embeds
    torch.cuda.empty_cache()
    gc.collect()
    
    return trajs, nonstop_trajs, mags, dist, l2_1,l2_2,l2_4,l2_10,l2_20,l2_40, magacc, magcacc, mag_acc_1,mag_acc_2,mag_acc_4,mag_acc_10,mag_acc_20,mag_acc_40,mag_cacc_1,mag_cacc_2,mag_cacc_4,mag_cacc_10,mag_cacc_20,mag_cacc_40,angle,cos_score 
    
    
def gen_fixation(model, dataloader, pa, device, sample_action, i_epoch, center_initial=True):

    all_actions, nonstop_actions = [], []
    center_initial = True
    count = 0
    avg_dist,avg_magacc,avg_magcacc,count2,count2_c = 0,0,0,0,0
    avg_dist1,avg_dist2,avg_dist4,avg_dist10,avg_dist20,avg_dist40 = 0,0,0,0,0,0
    avg_magacc1,avg_magacc2,avg_magacc4,avg_magacc10,avg_magacc20,avg_magacc40 = 0,0,0,0,0,0
    avg_magcacc1,avg_magcacc2,avg_magcacc4,avg_magcacc10,avg_magcacc20,avg_magcacc40 = 0,0,0,0,0,0
    angle,avg_angle,count_ang,count_cos,avg_emb = 0,0,0,0,0
    avg_nss_score,avg_auc_score = 0,0
    
    num_split = 1
    dict_2x = dict()
    files = glob.glob('./input_features/split'+str(num_split)+'/2x/*.npy')
    for i in range(len(files)):
        nme = files[i].split('/')[-1][:-4].split('_')[0]
        dict_2x[nme] = np.load(files[i],allow_pickle=True)
      
    dict_4x = dict()
    files = glob.glob('./input_features/split'+str(num_split)+'/4x/*.npy')
    for i in range(len(files)):
        nme = files[i].split('/')[-1][:-4].split('_')[0]
        dict_4x[nme] = np.load(files[i],allow_pickle=True)

    dict_10x = dict()
    files = glob.glob('./input_features/split'+str(num_split)+'/10x/*.npy')
    for i in range(len(files)):
        nme = files[i].split('/')[-1][:-4].split('_')[0]
        dict_10x[nme] = np.load(files[i],allow_pickle=True)

    dict_20x = dict()
    files = glob.glob('./input_features/split'+str(num_split)+'/20x/*.npy')
    for i in range(len(files)):
        nme = files[i].split('/')[-1][:-4].split('_')[0]
        dict_20x[nme] = np.load(files[i],allow_pickle=True)
    
    mags = torch.tensor([[0,0,0,1,1,2]]) #dummy
    for i in range(1):
        count1,count2 = 0,0
        counts = np.zeros((6,6))
        mag_counts = np.zeros((1,6))
        mag_ccounts = np.zeros((1,6))
        for batch in tqdm(dataloader, desc=f'Generate scanpaths [{i}/10]:'):
            #random.seed(15)
            curr_mag1 = batch['mag'][0].cpu().numpy()[-1]
            next_mag1 = batch['next_mag'][0].cpu().numpy()
            t = random.random()

            if (t < 0.18): 
                count1 += 1
                counts[curr_mag1,next_mag1] += 1
                
                mag_counts[0,next_mag1] += 1
                
                task_ids = batch['task_id'].to(device)
                img_names_batch = batch['img_name']
                    
                trajs, nonstop_trajs, mags, dist, l2_1,l2_2,l2_4,l2_10,l2_20,l2_40, magacc, magcacc, mag_acc_1,mag_acc_2,mag_acc_4,mag_acc_10,mag_acc_20,mag_acc_40,mag_cacc_1,mag_cacc_2,mag_cacc_4,mag_cacc_10,mag_cacc_20,mag_cacc_40,angle,cos_score = scanpath_decode(
                    model.module if isinstance(model, torch.nn.DataParallel) else model,dict_2x,dict_4x,dict_10x,dict_20x,num_split,img_names_batch, i_epoch, count, batch, task_ids, pa, sample_action, center_initial)
                
                if dist == 0 and magacc == 0 and magcacc == 0:
                    5==5
                else:
                    avg_dist += dist
                    
                    avg_dist1 += l2_1
                    avg_dist2 += l2_2
                    avg_dist4 += l2_4
                    avg_dist10 += l2_10
                    avg_dist20 += l2_20
                    avg_dist40 += l2_40
                    
                    avg_magacc += magacc
                    
                    avg_magacc1 += mag_acc_1
                    avg_magacc2 += mag_acc_2
                    avg_magacc4 += mag_acc_4
                    avg_magacc10 += mag_acc_10
                    avg_magacc20 += mag_acc_20
                    avg_magacc40 += mag_acc_40
                    
                    count2 += 1
                    
                    if magcacc != -1:
                        avg_magcacc += magcacc
                        count2_c += 1
                        
                        avg_magcacc1 += mag_cacc_1
                        avg_magcacc2 += mag_cacc_2
                        avg_magcacc4 += mag_cacc_4
                        avg_magcacc10 += mag_cacc_10
                        avg_magcacc20 += mag_cacc_20
                        avg_magcacc40 += mag_cacc_40
                        
                        mag_ccounts[0,next_mag1] += 1
                    
                    if angle != -1:
                        avg_angle += angle
                        count_ang += 1
                        
                    avg_emb += cos_score
                    count_cos += 1
                    
                        
                del trajs, nonstop_trajs, dist, magacc, magcacc
                    
            count += 1
        
        print('Evaluation metrics on ',count1,' data points:','total:',count2)
        print('Current Magnification-wise counts:',counts)
        if not sample_action:
            break
            
    
    scanpaths = actions2scanpaths(all_actions, pa.patch_num, pa.im_h, pa.im_w, mags)
    del mags
    try:
        print('Evaluation metrics on ',count1,' data points:')
        print('Spatial loss:',avg_dist/count2)
        print('Spatial loss (magnification-wise):')
        print('1x:',avg_dist1/mag_counts[0,0])
        print('2x:',avg_dist2/mag_counts[0,1])
        print('4x:',avg_dist4/mag_counts[0,2])
        print('10x:',avg_dist10/mag_counts[0,3])
        print('20x:',avg_dist20/mag_counts[0,4])
        print('40x:',avg_dist40/mag_counts[0,5])
        print('Angle deviation:',avg_angle/count_ang,', over #points: ',count_ang)
        print('Embedding score:',avg_emb/count_cos,', over #points: ',count_cos)
        #print('NSS score:',avg_nss_score/count_cos,', over #points: ',count_cos)
        #print('AUC score:',avg_auc_score/count_cos,', over #points: ',count_cos)
        
        print('Magnification accuracy:',avg_magacc/count2)
        print('Magnification-wise prediction accuracy: ')
        print(avg_magacc1/mag_counts[0,0],mag_counts[0,0])
        print(avg_magacc2/mag_counts[0,1],mag_counts[0,1])
        print(avg_magacc4/mag_counts[0,2],mag_counts[0,2])
        print(avg_magacc10/mag_counts[0,3],mag_counts[0,3])
        print(avg_magacc20/mag_counts[0,4],mag_counts[0,4])
        print(avg_magacc40/mag_counts[0,5],mag_counts[0,5])
        
        print('Magnification-wise change prediction accuracy: ')
        print(avg_magcacc1/mag_ccounts[0,0],mag_ccounts[0,0])
        print(avg_magcacc2/mag_ccounts[0,1],mag_ccounts[0,1])
        print(avg_magcacc4/mag_ccounts[0,2],mag_ccounts[0,2])
        print(avg_magcacc10/mag_ccounts[0,3],mag_ccounts[0,3])
        print(avg_magcacc20/mag_ccounts[0,4],mag_ccounts[0,4])
        print(avg_magcacc40/mag_ccounts[0,5],mag_ccounts[0,5])
        
        print('Magnification change accuracy:',avg_magcacc/count2_c,' (',count2_c,')')
        print([avg_dist/count2,avg_angle/count_ang,avg_emb/count_cos,avg_magacc*100/count2,
        avg_magacc1*100/mag_counts[0,0],avg_magacc2*100/mag_counts[0,1],avg_magacc4*100/mag_counts[0,2],
        avg_magacc10*100/mag_counts[0,3],avg_magacc20*100/mag_counts[0,4],avg_magcacc*100/count2_c,avg_magcacc1*100/mag_ccounts[0,0],avg_magcacc2*100/mag_ccounts[0,1],avg_magcacc4*100/mag_ccounts[0,2],avg_magcacc10*100/mag_ccounts[0,3],avg_magcacc20*100/mag_ccounts[0,4]])
        print('============================')
    except:
        print('divide by zero occured!')
    return scanpaths, nonstop_actions
        
            
            
def visualize_ims22(orig,save_path,fixes,mags,name,dict1):
    prev_fixes = np.array(fixes[0])
    
    dict1[name] = {}
    dict1[name]['X'] = prev_fixes[:,0].tolist()
    dict1[name]['Y'] = prev_fixes[:,1].tolist()
    dict1[name]['M'] = mags
    
    prev_fixes[:,0] = prev_fixes[:,0]*(orig.shape[0]/80)
    prev_fixes[:,1] = prev_fixes[:,1]*(orig.shape[1]/128)
    
    scan = plot_scanpath22(orig, prev_fixes[:,1], prev_fixes[:,0], mags, save_path)
    return dict1

def visualize_ims223(orig,save_path,fixes,mags,name):
    prev_fixes = np.array(fixes[0])
    prev_fixes[:,0] = prev_fixes[:,0]*(orig.shape[0]/80)
    prev_fixes[:,1] = prev_fixes[:,1]*(orig.shape[1]/128)
    orig = cv2.cvtColor(orig,cv2.COLOR_BGR2RGB)
    plot_scanpath22(orig, prev_fixes[:,1], prev_fixes[:,0], mags, save_path)
    
    
def plot_scanpath22(img, xs, ys, mags, save_path):
    fig, ax = plt.subplots()
    ax.imshow(img)
    
    colors = ['RoyalBlue', 'cyan', 'blue', 'green', 'yellow', 'red']
    color_labels = ['1x', '2x', '4x', '10x', '20x', '40x']
    
    for i in range(len(xs)):
        if i > 0:
            plt.arrow(xs[i - 1], ys[i - 1], xs[i] - xs[i - 1],
                      ys[i] - ys[i - 1], width=3, color='yellow', alpha=0.5)
    
    # Plot circles at points
    for i in range(len(mags)):
        cir_rad = int(25)  # radius for intermediate points
        alpha1 = 0.5

        circle = plt.Circle((xs[i], ys[i]),
                            radius=cir_rad,
                            edgecolor='red',
                            facecolor=colors[mags[i]],
                            alpha=alpha1)
        ax.add_patch(circle)
        plt.annotate("{}".format(i + 1), xy=(xs[i], ys[i] + 3), fontsize=6, ha="center", va="center")
        
    # Create legend patches
    legend_patches = [mpatches.Patch(color=colors[i], label=color_labels[i]) for i in range(len(colors))]
    
    # Add legend to the plot
    ax.legend(handles=legend_patches, loc='lower right', bbox_to_anchor=(1,0), fontsize='small', markerscale=0.5, framealpha=0.8)
    
    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    
    
def scanpath_decode22(model, num_split, dict1, img_names_batch, i_epoch, count2, batch, task_ids, pa, sample_action=False, center_initial=True):

    bs = 1
    name1 = img_names_batch[0]
    mags_seq = batch['mag'].to(device)
    random_point = batch['random_point']
    
    patatt_2x = np.load('./input_features/split'+str(num_split)+'/2x/'+name1[:-4]+'-01Z-00-DX1.npy',allow_pickle=True)[0]
    patatt_10x = np.load('./input_features/split'+str(num_split)+'/10x/'+name1[:-4]+'-01Z-00-DX1.npy',allow_pickle=True)[0]
            
    low_res_embed = patatt_2x
    high_res_embed = patatt_10x
    
    low_res_embed = torch.from_numpy(low_res_embed).unsqueeze(0)
    low_res_embed = torch.permute(low_res_embed,(0,3,1,2))
    high_res_embed = torch.from_numpy(high_res_embed).unsqueeze(0)
    high_res_embed = torch.permute(high_res_embed,(0,3,1,2))
    low_res_embed = F.interpolate(low_res_embed, [int(pa.im_h/8),int(pa.im_w/8)], mode='bilinear', align_corners=True)
    high_res_embed = F.interpolate(high_res_embed, [pa.im_h,pa.im_w], mode='bilinear', align_corners=True)
    low_res_embed = torch.permute(low_res_embed,(0,2,3,1)).squeeze(0)
    high_res_embed = torch.permute(high_res_embed,(0,2,3,1)).squeeze(0)
    low_res_embed = low_res_embed.numpy()
    high_res_embed = high_res_embed.numpy()
            
    with torch.no_grad():
        dorsal_embs, dorsal_pos, dorsal_mask, high_res_featmaps = model.encode(name1,low_res_embed,high_res_embed)

    normalized_fixs = torch.zeros(bs, 1, 2).fill_(0.5)
    action_mask = get_IOR_mask(np.ones(bs) * 0.5,
                               np.ones(bs) * 0.5,
                               pa.im_h, 
                               pa.im_w, 
                               pa.IOR_radius)
        
    stop_flags,mags,mag_pred = [],[],[]
    
    trans_mat = [[0.286,0.277,0.274,0.241,0.549,0.804],[1,0.825,0.710,0.418,0.168,0]]
    for i in range(60):
        with torch.no_grad():

            ys, ys_high = utils.transform_fixations(normalized_fixs, None, pa, [1,int(pa.im_h/8),int(pa.im_w/8)], [1,pa.im_h,pa.im_w], False,return_highres=True)
            padding = None
            if i == 0:
                mags_seq = torch.zeros(bs, 1).to(torch.long).to(device)
                mags_seq[0,0] = 0

            out = model.decode_and_predict(#map_2x,map_4x,map_10x,map_20x,
                low_res_embed,high_res_embed,dorsal_embs.clone(), dorsal_pos, dorsal_mask, high_res_featmaps.clone(),
                ys.to(device), padding, ys_high.to(device), mags_seq, i+1,'test', task_ids)
                
            prob, mag = out['pred_fixation_map'], out['pred_magnification']
            
            
            # mags.append(mag)
            #mag_pred.append(torch.argmax(mag[0])) #torch.argmax(mag[0])

            #if pa.enforce_IOR:
            batch_idx, visited_locs = torch.where(action_mask==True)
            prob[batch_idx, visited_locs] = 0

        if sample_action:
            m = Categorical(prob)
            next_word = m.sample()
        else:
            _, next_word = torch.max(prob, dim=1)
        
        next_word = next_word.cpu()
        
        norm_fy = (next_word % pa.im_w) / float(pa.im_w)
        norm_fx = (next_word // pa.im_w) / float(pa.im_h)
        
        
        normalized_fixs = torch.cat([normalized_fixs, torch.stack([norm_fx, norm_fy], dim=1).unsqueeze(1)], dim=1)
        
        mags_curr = torch.zeros(1, 1).to(torch.long).to(device)
        
        #mags_curr[0,0] = torch.argmax(mag[0])
        
        # temperature = 4
        # probabilities = F.softmax(mag[0]/temperature, dim=-1)
        # m = torch.distributions.Categorical(probabilities)
        # mags_curr[0,0] = m.sample().item() #-1+mags_seq[0][-1],0)
        
        
        mag1 = mags_seq[-1][0]     
        temperature = 4
        probabilities = F.softmax(mag[0]/temperature, dim=-1)
        if mag1 == 0:
            m = torch.distributions.Categorical(probabilities[0:2])
            mag = mag1 + m.sample().item()
        elif mag1 > 0 and mag1 < 5:
            m = torch.distributions.Categorical(probabilities[mag1-1:mag1+2])
            mag = mag1 + m.sample().item()-1
        elif mag1 == 5:
            m = torch.distributions.Categorical(probabilities[4:6])
            mag = mag1 + m.sample().item()-1
                
        mags_curr[0,0] = mag
        
        
        mags_seq = torch.cat([mags_seq,mags_curr])
        new_mask = get_IOR_mask(norm_fx.numpy(),
                                norm_fy.numpy(),
                                pa.im_h, 
                                pa.im_w, 
                                pa.IOR_radius)
                                
        action_mask = torch.logical_or(action_mask, new_mask)
        
        
    mags_seq = torch.transpose(mags_seq, 0, 1)
    
    fixs_pred = (normalized_fixs * torch.Tensor([pa.im_h,pa.im_w])).to(torch.long)
    orig = cv2.imread('./datasets/WSIs/images/'+name1[:-4]+'-01Z-00-DX1.png')
    orig = cv2.cvtColor(orig,cv2.COLOR_BGR2RGB)
    save_path = './scanpaths_visualized/'+name1[:-4]+'.png'

    tot_mag = mags_seq[0].cpu().tolist()
    dict1 = visualize_ims22(orig, save_path, fixs_pred, tot_mag,name1[:-4],dict1)
            
    mags = 1 #torch.stack(mags, dim=1) dummy mag
    trajs = []
    for i in range(normalized_fixs.size(0)):
        trajs.append(normalized_fixs[i, :])

    nonstop_trajs = [normalized_fixs[i] for i in range(normalized_fixs.size(0))]
    return trajs, nonstop_trajs, mags, dict1
    
def prepare_random_json(path_pred,path_gt,num_split):
    with open(path_pred, 'r') as json_file1:
        pred_scan = json.load(json_file1)
    with open(path_gt, 'r') as json_file2:
        gt_scan = json.load(json_file2)
        
    dict1 = {}
    mag_list = [1,2,4,10,20,40]
    for key in pred_scan.keys():
        list1 = []
        for key1 in gt_scan.keys():
            if key != gt_scan[key1]['name'][:-4]:
                list1.append(key1)
        
        sel = random.randint(0, len(list1)-1)
        dict2 = gt_scan[list1[sel]]
        dict1[key] = {}
        dict1[key]['X'] = dict2['X'].copy()
        dict1[key]['Y'] = dict2['Y'].copy()
        dict1[key]['M'] = dict2['M'].copy()
        for k in range(len(dict1[key]['M'])):
            #print(dict1[key])
            dict1[key]['M'][k] = mag_list.index(dict1[key]['M'][k])
        
    path1 = './prediction_jsons/predictions_split'+str(num_split)+'_random2.json'
    with open(path1, 'w') as fp:
        json.dump(dict1,fp,indent=4)
    return path1
    
def extract_feats_scanpaths(path1,dict_2x,dict_4x,dict_10x,dict_20x,mode):

    with open(path1, 'r') as json_file:
        scanpaths = json.load(json_file)
        
    all_embs = []
    name_embs = dict()
    mags_list = [1,2,4,10,20,40]
    seqs = dict()
    for key in scanpaths.keys():
        embs = np.zeros((len(scanpaths[key]['X']),384+1))
        seq = []
        if mode == 'gt':
            name1 = scanpaths[key]['name'][:-4]+'-01Z-00-DX1'
        else:
            name1 = key+'-01Z-00-DX1'
        
        for j in range(len(scanpaths[key]['X'])):
            mag = scanpaths[key]['M'][j]
            #print(mag,mode)
            if mode == 'gt':
                5==5
            else:
                mag = mags_list[mag]

            if mag == 1:
                arr = dict_2x[name1]
            if mag == 2:
                arr = dict_2x[name1]
            if mag == 4:
                arr = dict_4x[name1]
            if mag == 10:
                arr = dict_10x[name1]
            if mag == 20:
                arr = dict_20x[name1]
            if mag == 40:
                arr = dict_20x[name1]
                
            h,w = arr.shape[1],arr.shape[2]
            h_m = int((scanpaths[key]['X'][j]/80)*h)
            w_m = int((scanpaths[key]['Y'][j]/128)*w)
            embs[j,0:384] = arr[0,h_m,w_m,:]
            embs[j,384] = mag
            
        if name1 not in name_embs.keys():
            name_embs[name1] = [embs]
        else:
            name_embs[name1].append(embs)
            
    return name_embs
    
def measure_performance(path_pred,path_gt,num_split):
    
    dict_2x = dict()
    files = glob.glob('./input_features/split'+str(num_split)+'/2x/*.npy')
    for i in tqdm(range(len(files)), desc="Loading 2x magnification files"):
        nme = files[i].split('/')[-1][:-4].split('_')[0]
        dict_2x[nme] = np.load(files[i],allow_pickle=True)
    print('mag 2x loaded')

    dict_4x = dict()
    files = glob.glob('./input_features/split'+str(num_split)+'/4x/*.npy')
    for i in tqdm(range(len(files)), desc="Loading 4x magnification files"):
        nme = files[i].split('/')[-1][:-4].split('_')[0]
        dict_4x[nme] = np.load(files[i],allow_pickle=True)
    print('mag 4x loaded')

    dict_10x = dict()
    files = glob.glob('./input_features/split'+str(num_split)+'/10x/*.npy')
    for i in tqdm(range(len(files)), desc="Loading 10x magnification files"):
        nme = files[i].split('/')[-1][:-4].split('_')[0]
        dict_10x[nme] = np.load(files[i],allow_pickle=True)
    print('mag 10x loaded')

    dict_20x = dict()
    files = glob.glob('./input_features/split'+str(num_split)+'/20x/*.npy')
    for i in tqdm(range(len(files)), desc="Loading 20x magnification files"):
        nme = files[i].split('/')[-1][:-4].split('_')[0]
        dict_20x[nme] = np.load(files[i],allow_pickle=True) #dict_20x[nme]
    print('mag 20x loaded')
    
    embs_gt = extract_feats_scanpaths(path_gt,dict_2x,dict_4x,dict_10x,dict_20x,'gt')        
    print("--------")
    embs_pred = extract_feats_scanpaths(path_pred,dict_2x,dict_4x,dict_10x,dict_20x,'pred')
    print("--------")
    
    all_nsses,all_igs,all_aucs = compute_scan_metrics(path_pred,path_gt)

    
    avg_1x,avg_2x,avg_4x,avg_10x,avg_20x,avg_nss,avg_ig,avg_scanmatch,avg_auc = 0,0,0,0,0,0,0,0,0
    avg_overall,count_overall = 0,0
    count_1x,count_2x,count_4x,count_10x,count_20x,count_nss = 0,0,0,0,0,0
    for key in embs_pred.keys():
        pred = embs_pred[key]
        gt = embs_gt[key]
        
        pred_1x,pred_2x,pred_4x,pred_10x,pred_20x = [],[],[],[],[]
        gt_1x,gt_2x,gt_4x,gt_10x,gt_20x = [],[],[],[],[]
        
        for i in range(len(pred[0])):
            mag = pred[0][i][384]
            if mag == 1:
                pred_1x.append(pred[0][i][:384].reshape(384,))
            if mag == 2:
                pred_2x.append(pred[0][i][:384].reshape(384,))
            elif mag == 4:
                pred_4x.append(pred[0][i][:384].reshape(384,))
            elif mag == 10:
                pred_10x.append(pred[0][i][:384].reshape(384,))
            elif mag == 20:
                pred_20x.append(pred[0][i][:384].reshape(384,))
                
        for j in range(len(gt)):
            for i in range(len(gt[j])):
                mag = gt[j][i][384]
                if mag == 1:
                    gt_1x.append(gt[j][i][:384].reshape(384,))
                elif mag == 2:
                    gt_2x.append(gt[j][i][:384].reshape(384,))
                elif mag == 4:
                    gt_4x.append(gt[j][i][:384].reshape(384,))
                elif mag == 10:
                    gt_10x.append(gt[j][i][:384].reshape(384,))
                elif mag == 20:
                    gt_20x.append(gt[j][i][:384].reshape(384,))
                

        avg1,len1 = compute_sim(pred_1x,gt_1x,'1x')
        avg2,len2 = compute_sim(pred_2x,gt_2x,'2x')
        avg4,len4 = compute_sim(pred_4x,gt_4x,'4x')
        avg10,len10 = compute_sim(pred_10x,gt_10x,'10x')
        avg20,len20 = compute_sim(pred_20x,gt_20x,'20x')
        
        avg_1x += avg1
        avg_2x += avg2
        avg_4x += avg4
        avg_10x += avg10
        avg_20x += avg20
        
        overall = (avg1*len1 + avg2*len2 + avg4*len4 + avg10*len10 + avg20*len20)/(len1+len2+len4+len10+len20)

        avg_overall += overall
        count_overall += 1
        
        if len(pred_1x) != 0 and len(gt_1x) != 0:
            count_1x += 1
        if len(pred_2x) != 0 and len(gt_2x) != 0:
            count_2x += 1
        if len(pred_4x) != 0 and len(gt_4x) != 0:
            count_4x += 1
        if len(pred_10x) != 0 and len(gt_10x) != 0:
            count_10x += 1
        if len(pred_20x) != 0 and len(gt_20x) != 0:
            count_20x += 1
        
        for j in range(len(all_nsses)):
            #print(key[:-11],all_nsses[j][0])
            if key[:-11] == all_nsses[j][0]:
                avg_nss += all_nsses[j][1]
                avg_ig += all_igs[j][1]
                avg_auc += all_aucs[j][1]
                count_nss += 1
                
    if count_1x != 0:
        print('Average cosine similarity at 1x:',avg_1x/count_1x)
    if count_2x != 0:
        print('Average cosine similarity at 2x:',avg_2x/count_2x)
    if count_4x != 0:
        print('Average cosine similarity at 4x:',avg_4x/count_4x)
    if count_10x != 0:
        print('Average cosine similarity at 10x:',avg_10x/count_10x)
    if count_20x != 0:
        print('Average cosine similarity at 20x:',avg_20x/count_20x)
    print('Average overall cosine similarity:', avg_overall/count_overall,count_overall)
    print('Average NSS:',avg_nss/count_nss,count_nss)
    print('Average IG:',avg_ig/count_nss,count_nss)
    print('Average AUC:',avg_auc/count_nss,count_nss)
    val1 = avg_overall/count_overall
    val2 = avg_1x/count_1x
    val3 = avg_2x/count_2x
    val4 = avg_4x/count_4x
    val5 = avg_10x/count_10x
    val6 = avg_20x/count_20x
    val7 = avg_nss/count_nss
    val8 = avg_auc/count_nss
    
    print(val1.cpu().numpy(),val2.cpu().numpy(),val3.cpu().numpy(),val4.cpu().numpy(),val5.cpu().numpy(),val6.cpu().numpy(),val7,val8)

    
def gen_scanpath(model, dataloader, pa, device, sample_action, i_epoch, center_initial=True):

    all_actions, nonstop_actions = [], []
    center_initial = True
    count = 0
    mags = torch.tensor([[0,0,0,1,1,2]]) #dummy mags
    
    num_split = 1
    count1,count2 = 0,0
    counts = np.zeros((6,6))
    imgs_names = []
    dict1 = dict()
    for batch in tqdm(dataloader, desc=f'Generate scanpaths [1/10]:'):
        curr_mag1 = batch['mag'][0].cpu().numpy()[-1]
        next_mag1 = batch['next_mag'][0].cpu().numpy()

        img_names_batch = batch['img_name']

        if img_names_batch[0] not in imgs_names: 
            count1 += 1
            counts[curr_mag1,next_mag1] += 1

            task_ids = batch['task_id'].to(device)
            
            trajs, nonstop_trajs, mags, dict1 = scanpath_decode22(
                model.module if isinstance(model, torch.nn.DataParallel) else model,num_split,
                dict1,img_names_batch, i_epoch, count, batch, task_ids, pa, sample_action, center_initial)
            
            imgs_names.append(img_names_batch[0])
        count += 1
    
    scanpaths = actions2scanpaths(all_actions, pa.patch_num, pa.im_h, pa.im_w, mags)  
        
    del mags
    
    # path_gt = './datasets/WSIs/all_WSIs_fix_data_standarddim2_recent_'+str(num_split)+'.json'
    # path_pred = './prediction_jsons/predictions_split'+str(num_split)+'_baseline2.json'
    
    #measure_performance(path_pred,path_gt,num_split)
    
    return scanpaths, nonstop_actions
    
def evaluate(max_nss,
             max_auc,
             i_epoch,
             model,
             device,
             gazeloader,
             pa,
             human_cdf,
             task_dep_prior_maps,
             semSS_strings,
             dataset_root,
             human_scanpath_test,
             sample_action=True,
             sample_stop=False,
             output_saliency_metrics=True,
             center_initial=True,
             log_dir=None):
    print("Eval on {} batches of fixations".format(
        len(gazeloader)))
    model.eval()
    center_initial = True

    #scanpaths, nonstop_actions = gen_fixation(model, gazeloader, pa, device, sample_action, i_epoch, center_initial)
          
    scanpaths, nonstop_actions = gen_scanpath(model, gazeloader, pa, device, sample_action, i_epoch, center_initial)

    return scanpaths
    