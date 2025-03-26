from os.path import join
from torchvision import transforms
from torch.nn import Identity
import numpy as np
from pycocotools.coco import COCO
from .utils import compute_search_cdf, preprocess_fixations
from .utils import cutFixOnTarget, get_prior_maps
from .data import FFN_IRL, SPTrans_Human_Gaze


def process_data(target_trajs,
                 dataset_root,
                 target_annos,
                 hparams,
                 target_trajs_all,
                 is_testing=False,
                 sample_scanpath=False,
                 use_coco_annotation=False,
                 out_of_subject_eval=False):

    print("using", hparams.Train.repr, 'dataset:', hparams.Data.name, 'TAP:',
          hparams.Data.TAP)

    coco_annos = None

    # Rescale fixations and images if necessary
    if hparams.Data.name == 'WSI':
        #ori_h, ori_w = 320, 512
        rescale_flag = False # Use rescaled scanpaths
    else:
        print(f"dataset {hparams.Data.name} not supported")
        raise NotImplementedError

    if hparams.Train.repr == 'DCB':
        DCB_HR_dir = join(dataset_root, 'DCBs/HR/')
        DCB_LR_dir = join(dataset_root, 'DCBs/LR/')
    elif hparams.Train.repr == 'CFI' or hparams.Train.repr == 'FFN':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        raise NotImplementedError
        
    target_trajs_all = list(target_trajs_all.values())
    valid_target_trajs_all = list(
        filter(lambda x: x['split'] == 'test', target_trajs_all))
            
    is_coco_dataset = hparams.Data.name == 'COCO-Search18' or hparams.Data.name == 'COCO-Freeview'
    scene_labels = None

    target_init_fixs = {}
    count_e,count_n = 0,0
    for traj in target_trajs_all:
        key = traj['name'] + '*' + traj['condition']
        height = traj['height']
        width = traj['width']
        if is_coco_dataset:
            target_init_fixs[key] = (0.5, 0.5)  
        else:
            target_init_fixs[key] = (traj['X'][0] / height,
                                     traj['Y'][0] / width)

        if len(traj['X']) > 200:
            count_e += 1
        else:
            count_n += 1
            
    print('tot exceeded:',count_e,count_n)
    human_mean_cdf = None
    if is_testing:
        # testing fixation data
        target_trajs = list(target_trajs.values())
        test_target_trajs = list(
            filter(lambda x: x['split'] == 'test', target_trajs))
        assert len(test_target_trajs) > 0, 'no testing data found!'
        test_task_img_pair = np.unique([
             traj['name'] + '*' + traj['condition'] #traj['task'] + '*' +
            for traj in test_target_trajs
        ])

        # print statistics
        traj_lens = list(map(lambda x: x['length'], test_target_trajs))
        avg_traj_len, std_traj_len = np.mean(traj_lens), np.std(traj_lens)
        print('average train scanpath length : {:.3f} (+/-{:.3f})'.format(
            avg_traj_len, std_traj_len))
        print('num of train trajs = {}'.format(len(test_target_trajs)))

        if hparams.Train.repr == 'FFN':
            test_img_dataset = FFN_IRL(dataset_root, target_init_fixs,
                                       test_task_img_pair, target_annos,
                                       transform_test, hparams.Data, catIds)

        return {
            'catIds': catIds,
            'img_test': test_img_dataset,
            'bbox_annos': target_annos,
            'gt_scanpaths': test_target_trajs,
            'fix_clusters': fix_clusters
        }

    else:
        # training fixation data
        target_trajs = list(target_trajs.values())
        train_target_trajs = list(
            filter(lambda x: x['split'] == 'train', target_trajs))
            
        # print statistics
        traj_lens = list(map(lambda x: x['length'], train_target_trajs))
        avg_traj_len, std_traj_len = np.mean(traj_lens), np.std(traj_lens)
        print('average train scanpath length : {:.3f} (+/-{:.3f})'.format(
            avg_traj_len, std_traj_len))
        print('num of train trajs = {}'.format(len(train_target_trajs)))

        train_task_img_pair = np.unique([
            traj['name'] + '*' + traj['condition'] #traj['task'] + '*' + 
            for traj in train_target_trajs
        ])
        
        print('train label:')
        train_fix_labels = preprocess_fixations(
            train_target_trajs,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            has_stop=hparams.Data.has_stop,
            sample_scanpath=sample_scanpath,
            discretize_fix=hparams.Data.discretize_fix,
            remove_return_fixations=hparams.Data.remove_return_fixations,
            is_coco_dataset=is_coco_dataset,
        )

        # validation fixation data
        valid_target_trajs = list(
            filter(lambda x: x['split'] == 'test', target_trajs))
            
        all_target_trajs = list(
            filter(lambda x: 5==5, target_trajs))
            
        # print statistics
        traj_lens = list(map(lambda x: x['length'], valid_target_trajs))
        avg_traj_len, std_traj_len = np.mean(traj_lens), np.std(traj_lens)
        print('average valid scanpath length : {:.3f} (+/-{:.3f})'.format(
            avg_traj_len, std_traj_len))
        print('num of valid trajs = {}'.format(len(valid_target_trajs)))

        valid_task_img_pair = np.unique([
            traj['name'] + '*' + traj['condition']
            for traj in valid_target_trajs
        ])

        print('valid label:')
        valid_fix_labels = preprocess_fixations(
            valid_target_trajs,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            has_stop=hparams.Data.has_stop,
            sample_scanpath=sample_scanpath,
            discretize_fix=hparams.Data.discretize_fix,
            remove_return_fixations=hparams.Data.remove_return_fixations,
            is_coco_dataset=is_coco_dataset,
        )

        print('all label:')
        valid_fix_labels = preprocess_fixations(
            all_target_trajs,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            has_stop=hparams.Data.has_stop,
            sample_scanpath=sample_scanpath,
            discretize_fix=hparams.Data.discretize_fix,
            remove_return_fixations=hparams.Data.remove_return_fixations,
            is_coco_dataset=is_coco_dataset,
        )
        
        valid_target_trajs_FV = list(
            filter(lambda x: x['condition'] == 'freeview',
                    valid_target_trajs_all))
                    
        if len(valid_target_trajs_FV) > 0:
            valid_fix_labels_FV = preprocess_fixations(
                valid_target_trajs_FV,
                hparams.Data.patch_size,
                hparams.Data.patch_num,
                hparams.Data.im_h,
                hparams.Data.im_w,
                has_stop=hparams.Data.has_stop,
                sample_scanpath=sample_scanpath,
                discretize_fix=hparams.Data.discretize_fix,
                is_coco_dataset=is_coco_dataset,
            )
        else:
            valid_fix_labels_FV = []
            
        valid_fix_labels_all = preprocess_fixations(
            valid_target_trajs_all,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            has_stop=hparams.Data.has_stop,
            sample_scanpath=sample_scanpath,
            discretize_fix=hparams.Data.discretize_fix,
            remove_return_fixations=hparams.Data.remove_return_fixations,
            is_coco_dataset=is_coco_dataset,
        )

        valid_task_img_pair_FV = np.unique([
            traj['name'] + '*' + traj['condition']
            for traj in valid_target_trajs_all
            if traj['condition'] == 'freeview'
        ])
        valid_task_img_pair_all = np.unique([
            traj['name'] + '*' + traj['condition']
            for traj in valid_target_trajs_all
        ])

        
        if hparams.Train.repr == 'FFN':
            # load image data
            train_img_dataset = FFN_IRL(dataset_root, None,
                                        train_task_img_pair, target_annos,
                                        transform_train, hparams.Data, #catIds,
                                        coco_annos=coco_annos)
            valid_img_dataset_all = FFN_IRL(dataset_root, None,
                                            valid_task_img_pair_all, target_annos,
                                            transform_test, hparams.Data, #catIds,
                                            coco_annos=None)
            valid_img_dataset_FV = FFN_IRL(dataset_root, None,
                                           valid_task_img_pair_FV,
                                           target_annos, transform_test,
                                           hparams.Data, None)

            if hparams.Model.name == 'HAT':
                gaze_dataset_func = SPTrans_Human_Gaze   
            else:
                raise NotImplementedError

            train_HG_dataset = gaze_dataset_func(dataset_root,
                                                 train_fix_labels,
                                                 target_annos,
                                                 scene_labels,
                                                 hparams.Data,
                                                 transform_train,
                                                 #catIds,
                                                 blur_action=True,
                                                 coco_annos=coco_annos)
            valid_HG_dataset = gaze_dataset_func(dataset_root,
                                                 valid_fix_labels,
                                                 target_annos,
                                                 scene_labels,
                                                 hparams.Data,
                                                 transform_test,
                                                 #catIds,
                                                 blur_action=True,
                                                 coco_annos=None)
            if len(valid_target_trajs_FV) > 0:
                print('valid HG:')
                valid_HG_dataset_FV = gaze_dataset_func(dataset_root,
                                                        valid_fix_labels_FV,
                                                        target_annos,
                                                        scene_labels,
                                                        hparams.Data,
                                                        transform_test,
                                                        #catIds,
                                                        blur_action=True,
                                                        coco_annos=None)
            else:
                valid_HG_dataset_FV = None
                
            valid_HG_dataset_all = gaze_dataset_func(dataset_root,
                                                     valid_fix_labels_all,
                                                     target_annos,
                                                     scene_labels,
                                                     hparams.Data,
                                                     transform_test,
                                                     blur_action=True,
                                                     coco_annos=None)


        return {
            'img_train': train_img_dataset,
            'img_valid_FV': valid_img_dataset_FV,
            'img_valid': valid_img_dataset_all,
            'gaze_train': train_HG_dataset,
            'gaze_valid': valid_HG_dataset,
            'gaze_valid_FV': valid_HG_dataset_FV,
            'bbox_annos': target_annos,
            'valid_scanpaths': valid_target_trajs_all,
            'human_cdf': human_mean_cdf,
        }
