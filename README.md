# Code for our paper "Where and When Pathologists Focus their Visual Attention while Grading Whole Slide Images of Cancer"

The project aims at predicting the spatio-temporal dynamics of pathologists' attention allocation as they view WSIs for tumor grading. In this work, we predict both attention heatmaps and attention scanpaths of pathologists during their WSI viewing.

## Heatmap prediction (PAT-H model)
The directory 'heatmap_prediction' contains code for: 1) predicting attention heatmaps on WSIs, 2) obtaining feature encodings of the WSIs at multiple magnification levels trained using the heatmap prediction task. Follow the steps below to run training and inference codes for predicting attention heatmaps:

1) Enter into the 'heatmap_prediction' directory using: "cd heatmap_prediction".
2) Run "pip install -r requirements.txt" in order to install the libraries required for this task.
3) Download models and data at: https://drive.google.com/file/d/15dHFFgLkwh_OmkdN3KdV8bbrLwDEAZ5B/view?usp=sharing
4) Extract the 'heatmap_drive' directory from the zip file in the link above and plug-in the models and data into the corresponding directories of 'heatmap_prediction'.
5) Run 'train.py' for training a PAT-H model and 'eval.py' for evaluating the performance of a trained PAT-H model.

## Scanpath prediction (PAT-S model)
The directory 'scanpath_prediction' contains code for predicting attention scanpaths on WSIs based on the feature encodings obtained in the previous stage of heatmap prediction. Follow the 
steps below to run training and inference codes for predicting attention heatmaps:

1) Enter into the 'scanpath_prediction' directory using: "cd scanpath_prediction".
2) Run "pip install -r requirements.txt" in order to install the libraries required for this task.
3) Download models and data at: https://drive.google.com/file/d/15dHFFgLkwh_OmkdN3KdV8bbrLwDEAZ5B/view?usp=sharing
4) Extract the 'scanpath_drive' directory from the zip file in the link above and plug-in the models and data into the corresponding directories of 'scanpath_prediction'.
5) Run 'train_sptransformer.py' for training the PAT-S model using the feature encodings from previous stage (heatmap prediction) as the output. Run the same file in evaluation-only model (by adding '--eval_only')
   in order to run inference on the trained models.
   Example commands:
   Training: python train_sptransformer.py --hparams ./configs/coco_freeview_dense_SSL.json --dataset-root ./datasets/WSIs
   Evaluation: python train_sptransformer.py --hparams ./configs/coco_freeview_dense_SSL.json --dataset-root ./datasets/WSIs --eval-only
